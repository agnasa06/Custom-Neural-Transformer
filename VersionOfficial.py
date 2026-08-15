import subprocess
import sys
from torch.profiler import profile, ProfilerActivity

import torch
import torch.nn.functional as F
import pandas as pd
import os
import shutil
import time
from pathlib import Path
import numpy as np
import math
import torch.nn as nn
from torchao.float8 import (
	Float8LinearConfig,
	convert_to_float8_training,
)
from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
from torchao.float8.float8_training_tensor import LinearMMConfig
from dataclasses import replace
from torchao.float8 import Float8GemmConfig


def ensure_single_gpu_job():
	"""Fail fast if another GPU compute process is already active."""
	nvidia_smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
	if not Path(nvidia_smi).is_file():
		raise RuntimeError(
			"nvidia-smi not found; add it to PATH or install the NVIDIA WSL utilities."
		)

	try:
		result = subprocess.run(
			[
				nvidia_smi,
				"--query-compute-apps=pid,process_name,used_memory",
				"--format=csv,noheader",
			],
			capture_output=True,
			text=True,
			check=False,
		)
	except FileNotFoundError as exc:
		raise RuntimeError("nvidia-smi not found; cannot enforce single-GPU training.") from exc

	if result.returncode != 0:
		raise RuntimeError(
			"nvidia-smi failed while checking for active GPU jobs. "
			f"stderr: {result.stderr.strip() or '<empty>'}"
		)

	active_jobs = [line.strip() for line in result.stdout.splitlines() if line.strip()]
	if active_jobs:
		print("GPU is already in use by another process. Exiting before training starts.")
		for job in active_jobs:
			print(f"  - {job}")
		raise SystemExit(1)

	print("No active GPU compute jobs detected by nvidia-smi; starting training.")


ensure_single_gpu_job()

torch.set_float32_matmul_precision('high')
torch._inductor.config.triton.cudagraph_support_input_mutation = True

UNKNOWN_TOKEN = "<unk>"
mask_value = -2e4
device = torch.device("cuda")
torch.manual_seed(12354)
epsilon = 1e-7
FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2

FP8_E4M3_MAX = torch.finfo(FP8_E4M3).max
FP8_E5M2_MAX = torch.finfo(FP8_E5M2).max

character_to_id = {
	"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6,
	"h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
	"n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18,
	"t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24,
	"z": 25,

	"A": 26, "B": 27, "C": 28, "D": 29, "E": 30, "F": 31,
	"G": 32, "H": 33, "I": 34, "J": 35, "K": 36, "L": 37,
	"M": 38, "N": 39, "O": 40, "P": 41, "Q": 42, "R": 43,
	"S": 44, "T": 45, "U": 46, "V": 47, "W": 48, "X": 49,
	"Y": 50, "Z": 51,

	"1": 52, "2": 53, "3": 54, "4": 55, "5": 56,
	"6": 57, "7": 58, "8": 59, "9": 60, "0": 61,

	".": 62, ",": 63, "?": 64, "!": 65, ":": 66, ";": 67,
	"'": 68, "&": 69, "@": 70, "#": 71, "$": 72, "%": 73,
	"*": 74, "+": 75, "-": 76, "_": 77, "(": 78, ")": 79,
	"[": 80, "]": 81, "{": 82, "}": 83, "<": 84, ">": 85,
	"=": 86, "~": 87, "/": 88, "`": 89, "|": 90, '"': 91,
	"\\": 92, " ": 93,

	UNKNOWN_TOKEN: 94,
	"\n": 95,
}

id_to_character = {
	token_id: character
	for character, token_id in character_to_id.items()
}

number_of_ids = len(character_to_id)

def encode_text(text):
	token_ids = [character_to_id[char] if char in character_to_id else character_to_id[UNKNOWN_TOKEN] for char in text]
	return token_ids

def decode_token_ids(token_ids):
	return "".join([id_to_character[token_id] if token_id != character_to_id[UNKNOWN_TOKEN] else UNKNOWN_TOKEN for token_id in token_ids])

validation_size = 100000

dataset_token_ids = torch.load("wiki103_uint8.pt", weights_only=True)
token_ids = dataset_token_ids[:-validation_size]
validation_token_ids = dataset_token_ids[-validation_size:]

# #^ 1. Load the compressed parquet table
# df = pd.read_parquet("train-00000-of-00001.parquet")

# #^ 2. Extract the 'text' column rows and combine them
# dataText = "\n".join(df["text"].astype(str))
# training_text = dataText[:len(dataText) - 50000]
# validation_text = dataText[len(dataText) - 50000:]

# token_ids = torch.tensor(encode_text(training_text))
# validation_token_ids = torch.tensor(encode_text(validation_text))

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

def save_model(path, iteration, best_loss, previous_elapsed_seconds):
	"""
	Save the current tensor parameters and training progress.

	Saves to a temporary file first, then replaces the real checkpoint.
	This avoids corrupting an existing checkpoint if saving is interrupted.
	"""

	path = Path(path)

	if not path.is_absolute():
		path = BASE_DIR / path

	path.parent.mkdir(parents=True, exist_ok=True)

	checkpoint = {
		"iteration": iteration,
		"best_loss": best_loss,
		"previous_elapsed_seconds": previous_elapsed_seconds,

		"number_of_layers": number_of_layers,
		"embedding_size": embedding_size,
		"context_length": CONTEXT_LENGTH,
		"number_of_ids": number_of_ids,
		"ffn_hidden_size": ffn_hidden_size,
		"persistent_learning_rate": persistent_learning_rate,
		"head_count": head_count,
		"pmax": pmax,
		"qkv_initial_weight_rms": qkv_initial_weight_rms.detach().cpu(),
		"qkv_reference_momentum_rms": qkv_reference_momentum_rms.detach().cpu(),
		"ffn_up_initial_weight_rms": ffn_up_initial_weight_rms.detach().cpu(),
		"ffn_up_reference_momentum_rms": ffn_up_reference_momentum_rms.detach().cpu(),
		"ffn_down_initial_weight_rms": ffn_down_initial_weight_rms.detach().cpu(),
		"ffn_down_reference_momentum_rms": ffn_down_reference_momentum_rms.detach().cpu(),
		"Wo_initial_weight_rms": Wo_initial_weight_rms.detach().cpu(),
		"Wo_reference_momentum_rms": Wo_reference_momentum_rms.detach().cpu(),

		"token_embedding_table": token_embedding_table.detach().cpu(),
		"positional_embedding_table": (
			positional_embedding_table.detach().cpu()
		),
		"output_weight_table": output_weight_table.detach().cpu(),
		"attention_gamma_table": attention_gamma_table.detach().cpu(),
		"ffn_gamma_table": ffn_gamma_table.detach().cpu(),

		"momentum_rate": momentum_rate,

		"token_embedding_table_momentum_gradient":
			token_embedding_table_momentum_gradient.detach().cpu(),

		"positional_embedding_table_momentum_gradient":
			positional_embedding_table_momentum_gradient.detach().cpu(),

		"qkv_linear_momentum_gradient":
			qkv_linear_momentum_gradient.detach().cpu(),

		"output_weight_table_momentum_gradient":
			output_weight_table_momentum_gradient.detach().cpu(),

		"attention_gamma_table_momentum_gradient":
			attention_gamma_table_momentum_gradient.detach().cpu(),

		"ffn_gamma_table_momentum_gradient":
			ffn_gamma_table_momentum_gradient.detach().cpu(),
	}
	for layer in range(number_of_layers):
		checkpoint[f"ffn_up_linear_layer{layer+1}"] = ffn_up_linear[layer].weight.detach().cpu()
		checkpoint[f"ffn_down_linear_layer{layer+1}"] = ffn_down_linear[layer].weight.detach().cpu()
		checkpoint[f"Wo_linear{layer+1}"] = Wo_linear[layer].weight.detach().cpu()

		checkpoint[f"ffn_up_linear_momentum_gradient_layer{layer+1}"] = (
			ffn_up_linear_momentum_gradient[layer].detach().cpu()
		)

		checkpoint[f"ffn_down_linear_momentum_gradient_layer{layer+1}"] = (
			ffn_down_linear_momentum_gradient[layer].detach().cpu()
		)

		checkpoint[f"Wo_linear_momentum_gradient_layer{layer+1}"] = (
			Wo_linear_momentum_gradient[layer].detach().cpu()
		)

		checkpoint[f"qkv_linear_weight_layer{layer+1}"] = qkv_linear[layer].weight.detach().cpu()

	temporary_path = path.with_suffix(path.suffix + ".tmp")

	torch.save(checkpoint, temporary_path)

	# Replaces the old file only after torch.save() completed successfully.
	os.replace(temporary_path, path)

	print(f"Model saved to: {path}")


def load_model(path):
	"""
	Load a saved tensor checkpoint into the current global parameter tables.

	Returns:
		iteration, best_loss, persistent_learning_rate, previous_elapsed_seconds
	"""

	global token_embedding_table
	global positional_embedding_table
	global qkv_linear
	global output_weight_table
	global ffn_up_linear
	global ffn_down_linear
	global attention_gamma_table
	global ffn_gamma_table
	global Wo_linear
	global token_embedding_table_momentum_gradient
	global positional_embedding_table_momentum_gradient
	global qkv_linear_momentum_gradient
	global output_weight_table_momentum_gradient
	global ffn_up_linear_momentum_gradient
	global ffn_down_linear_momentum_gradient
	global Wo_linear_momentum_gradient
	global attention_gamma_table_momentum_gradient
	global ffn_gamma_table_momentum_gradient
	global qkv_initial_weight_rms, qkv_reference_momentum_rms
	global ffn_up_initial_weight_rms, ffn_up_reference_momentum_rms
	global ffn_down_initial_weight_rms, ffn_down_reference_momentum_rms
	global Wo_initial_weight_rms, Wo_reference_momentum_rms

	path = Path(path)

	if not path.is_absolute():
		path = BASE_DIR / path

	if not path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {path}")

	if path.stat().st_size == 0:
		raise ValueError(
			f"Checkpoint is empty/corrupted: {path}\n"
			"Delete it and start from scratch or load another checkpoint."
		)

	checkpoint = torch.load(
		path,
		map_location="cpu",
		weights_only=False,
	)

	# Confirm this checkpoint belongs to the current model configuration.
	if checkpoint["embedding_size"] != embedding_size:
		raise ValueError(
			"Cannot load checkpoint: embedding_size differs.\n"
			f"Checkpoint: {checkpoint['embedding_size']}\n"
			f"Current: {embedding_size}"
		)

	if checkpoint["context_length"] != CONTEXT_LENGTH:
		raise ValueError(
			"Cannot load checkpoint: CONTEXT_LENGTH differs.\n"
			f"Checkpoint: {checkpoint['context_length']}\n"
			f"Current: {CONTEXT_LENGTH}"
		)

	if checkpoint["number_of_ids"] != number_of_ids:
		raise ValueError(
			"Cannot load checkpoint: vocabulary size differs.\n"
			f"Checkpoint: {checkpoint['number_of_ids']}\n"
			f"Current: {number_of_ids}"
		)

	if checkpoint["head_count"] != head_count:
		raise ValueError(
			"Cannot load checkpoint: head_count differs.\n"
			f"Checkpoint: {checkpoint['head_count']}\n"
			f"Current: {head_count}"
		)

	if "momentum_rate" not in checkpoint:
		raise ValueError("Cannot load checkpoint: momentum_rate doesn't exist.\n")

	if "pmax" not in checkpoint:
		raise ValueError("Cannot load checkpoint: pmax doesn't exist.\n")

	if "qkv_linear_momentum_gradient" not in checkpoint:
		raise ValueError("Checkpoint does not contain momentum buffers.")

	baseline_keys = ("qkv_initial_weight_rms", "qkv_reference_momentum_rms", "ffn_up_initial_weight_rms", "ffn_up_reference_momentum_rms", "ffn_down_initial_weight_rms", "ffn_down_reference_momentum_rms", "Wo_initial_weight_rms", "Wo_reference_momentum_rms")
	missing_baselines = [key for key in baseline_keys if key not in checkpoint]
	if missing_baselines:
		raise ValueError(f"Checkpoint does not contain optimizer baseline data: {missing_baselines}")

	expected_qkv_shape = (3 * embedding_size, embedding_size)
	expected_wo_shape = (embedding_size, embedding_size)
	for layer in range(number_of_layers):
		qkv_key = f"qkv_linear_weight_layer{layer+1}"
		if qkv_key not in checkpoint:
			raise ValueError(f"Checkpoint does not contain {qkv_key}.")
		if checkpoint[qkv_key].shape != expected_qkv_shape:
			raise ValueError(
				"Cannot load checkpoint: QKV weight shape differs.\n"
				f"Layer: {layer + 1}\n"
				f"Checkpoint: {checkpoint[qkv_key].shape}\n"
				f"Current: {expected_qkv_shape}"
			)

		wo_key = f"Wo_linear{layer+1}"
		if wo_key not in checkpoint:
			raise ValueError(f"Checkpoint does not contain {wo_key}.")
		if checkpoint[wo_key].shape != expected_wo_shape:
			raise ValueError(
				"Cannot load checkpoint: Wo weight shape differs.\n"
				f"Layer: {layer + 1}\n"
				f"Checkpoint: {checkpoint[wo_key].shape}\n"
				f"Current: {expected_wo_shape}"
			)

	if checkpoint["ffn_hidden_size"] != ffn_hidden_size:
		raise ValueError(
			"Cannot load checkpoint: ffn_hidden_size_ size differs.\n"
			f"Checkpoint: {checkpoint['ffn_hidden_size']}\n"
			f"Current: {ffn_hidden_size}"
		)

	if checkpoint["number_of_layers"] != number_of_layers:
		raise ValueError(
			"Cannot load checkpoint: number_of_layers differs.\n"
			f"Checkpoint: {checkpoint['number_of_layers']}\n"
			f"Current: {number_of_layers}"
		)

	if "attention_gamma_table" in checkpoint:
		attention_gamma_table = nn.Parameter(checkpoint["attention_gamma_table"].to(device))

	if "ffn_gamma_table" in checkpoint:
		ffn_gamma_table = nn.Parameter(checkpoint["ffn_gamma_table"].to(device))

	token_embedding_table = nn.Parameter(checkpoint["token_embedding_table"].to(device))
	positional_embedding_table = nn.Parameter(checkpoint["positional_embedding_table"].to(device))
	output_weight_table = nn.Parameter(checkpoint["output_weight_table"].to(device))

	token_embedding_table_momentum_gradient = checkpoint[
		"token_embedding_table_momentum_gradient"
	].to(device)

	positional_embedding_table_momentum_gradient = checkpoint[
		"positional_embedding_table_momentum_gradient"
	].to(device)

	qkv_linear_momentum_gradient = checkpoint[
		"qkv_linear_momentum_gradient"
	].to(device)

	output_weight_table_momentum_gradient = checkpoint[
		"output_weight_table_momentum_gradient"
	].to(device)

	attention_gamma_table_momentum_gradient = checkpoint[
		"attention_gamma_table_momentum_gradient"
	].to(device)

	ffn_gamma_table_momentum_gradient = checkpoint["ffn_gamma_table_momentum_gradient"].to(device)
	qkv_initial_weight_rms = checkpoint["qkv_initial_weight_rms"].to(device)
	qkv_reference_momentum_rms = checkpoint["qkv_reference_momentum_rms"].to(device)
	ffn_up_initial_weight_rms = checkpoint["ffn_up_initial_weight_rms"].to(device)
	ffn_up_reference_momentum_rms = checkpoint["ffn_up_reference_momentum_rms"].to(device)
	ffn_down_initial_weight_rms = checkpoint["ffn_down_initial_weight_rms"].to(device)
	ffn_down_reference_momentum_rms = checkpoint["ffn_down_reference_momentum_rms"].to(device)
	Wo_initial_weight_rms = checkpoint["Wo_initial_weight_rms"].to(device)
	Wo_reference_momentum_rms = checkpoint["Wo_reference_momentum_rms"].to(device)

	with torch.no_grad():
		for layer in range(number_of_layers):
			qkv_linear[layer].weight.copy_(checkpoint[f"qkv_linear_weight_layer{layer+1}"].to(device))
			ffn_up_linear[layer].weight.copy_(checkpoint[f"ffn_up_linear_layer{layer+1}"].to(device))
			ffn_down_linear[layer].weight.copy_(checkpoint[f"ffn_down_linear_layer{layer+1}"].to(device))
			Wo_linear[layer].weight.copy_(checkpoint[f"Wo_linear{layer+1}"].to(device))

			ffn_up_linear_momentum_gradient[layer] = checkpoint[
				f"ffn_up_linear_momentum_gradient_layer{layer+1}"
			].to(device)

			ffn_down_linear_momentum_gradient[layer] = checkpoint[
				f"ffn_down_linear_momentum_gradient_layer{layer+1}"
			].to(device)

			Wo_momentum_key = f"Wo_linear_momentum_gradient_layer{layer+1}"
			if Wo_momentum_key in checkpoint:
				Wo_linear_momentum_gradient[layer] = checkpoint[Wo_momentum_key].to(device)
			else:
				# Older checkpoints did not save Wo momentum buffers.
				Wo_linear_momentum_gradient[layer] = torch.zeros_like(
					Wo_linear[layer].weight
				)

	iteration = checkpoint.get("iteration", 0)
	best_loss = checkpoint.get("best_loss", float("inf"))
	persistent_learning_rate = checkpoint.get("persistent_learning_rate", 1)
	previous_elapsed_seconds = checkpoint.get("previous_elapsed_seconds", 0)

	print(f"Model loaded from: {path}")
	print(f"Resuming from iteration: {iteration}")
	print(f"Previous best loss: {best_loss:.6f}")

	return iteration, best_loss, persistent_learning_rate, previous_elapsed_seconds

def get_window_token_ids(start_positions, source_token_ids):
	offsets = context_indices_plus1
	return source_token_ids[start_positions + offsets].to(device=device, dtype=torch.long)

def run_forward_attention(layer, input_embeddings, normalized_input_embeddings):
	#% Shape: 3 x context_length x embedding_size
	qkv_matrix = qkv_linear[layer](normalized_input_embeddings)

	query_matrix = qkv_matrix[...,0:embedding_size,]
	key_matrix = qkv_matrix[...,embedding_size:2 * embedding_size,]
	value_matrix = qkv_matrix[...,2 * embedding_size:3 * embedding_size,]

	query_matrix = query_matrix.reshape(batch_size, CONTEXT_LENGTH, head_count, embedding_size // head_count).transpose(1, 2)
	key_matrix = key_matrix.reshape(batch_size, CONTEXT_LENGTH, head_count, embedding_size // head_count).transpose(1, 2)
	value_matrix = value_matrix.reshape(batch_size, CONTEXT_LENGTH, head_count, embedding_size // head_count).transpose(1, 2)
	
	attention_output_matrix = F.scaled_dot_product_attention(
		query_matrix, key_matrix, value_matrix,
		dropout_p=0, is_causal=True
	).transpose(1, 2)
	attention_output_matrix = attention_output_matrix.reshape(batch_size, CONTEXT_LENGTH, embedding_size)
	attention_output_matrix = Wo_linear[layer](attention_output_matrix) + input_embeddings

	return attention_output_matrix


def run_forward_pass(start_positions, source_token_ids):
	# Look up one embedding vector for each input token ID.
	window_token_ids = get_window_token_ids(start_positions, source_token_ids).to(device)
	input_token_ids = window_token_ids[:, :-1]
	target_token_ids = window_token_ids[:, 1:]
	# print(input_token_ids.shape)

	#% Shape: context_length × embedding_size
	#* This is converting each token (character) in the input of size CONTEXT_LENGTH (64) to their vector representation from the token_embedding_table
	input_token_embeddings = token_embedding_table[input_token_ids]

	#% Shape: context_length x embedding_size
	#* Retrieves the position vector for each position in this input window. For now, this matches the full positional_embedding_table and is mainly useful for debugging.
	input_position_embeddings = positional_embedding_table

	#% Shape: context_length x embedding_size
	#* Adds the token vector and position vector at each position, producing the final input representation for the transformer.
	combined_input_embeddings = input_token_embeddings + input_position_embeddings

	layer_state = combined_input_embeddings

	#@ Layers 1 - number_of_layers

	for layer in range(0, number_of_layers):
		layer_state_input = layer_state
		root_mean_square = torch.sqrt(layer_state_input.square().mean(dim = -1, keepdim = True) + epsilon)
		attention_inverse_root_mean_square = 1 / root_mean_square
		normalized_layer_state = layer_state_input * attention_inverse_root_mean_square
		scaled_layer_state = normalized_layer_state * attention_gamma_table[layer]
		attention_output_matrix = run_forward_attention(layer, layer_state, scaled_layer_state)

		ffn_input = attention_output_matrix
		root_mean_square = torch.sqrt(ffn_input.square().mean(dim = -1, keepdim = True) + epsilon)
		ffn_inverse_root_mean_square = 1 / root_mean_square
		normalized_ffn_input = ffn_input * ffn_inverse_root_mean_square
		scaled_ffn_input = normalized_ffn_input * ffn_gamma_table[layer]

		ffn_upscale_matrix = ffn_up_linear[layer](scaled_ffn_input)
		
		ffn_upscale_matrix = F.gelu(ffn_upscale_matrix, approximate="tanh")

		layer_state = ffn_down_linear[layer](ffn_upscale_matrix) + attention_output_matrix

	

	logits_matrix = layer_state @ output_weight_table.transpose(-2, -1)

	average_loss = F.cross_entropy(
		logits_matrix.reshape(-1, number_of_ids),
		target_token_ids.reshape(-1),
	)

	return average_loss

def update_weights():
	with torch.no_grad():
		normalized_learning_rate = learning_rate_tensor #** (1 - norm_power)

		#& Token and positional embedding tables
		token_embedding_table_momentum_gradient.mul_(momentum_rate)
		token_embedding_table_momentum_gradient.add_(token_embedding_table.grad, alpha=(1 - momentum_rate))
		token_embedding_table.sub_(token_embedding_table_momentum_gradient * normalized_learning_rate)

		positional_embedding_table_momentum_gradient.mul_(momentum_rate)
		positional_embedding_table_momentum_gradient.add_(positional_embedding_table.grad, alpha=(1 - momentum_rate))
		positional_embedding_table.sub_(positional_embedding_table_momentum_gradient * normalized_learning_rate)

		qkv_log_ratio = torch.zeros(number_of_layers, device=device)
		qkv_momentum_rms = torch.zeros(number_of_layers, device=device)
		qkv_log_average = 0

		for layer in range(number_of_layers):
			qkv_linear_momentum_gradient[layer].mul_(momentum_rate)
			qkv_linear_momentum_gradient[layer].add_(qkv_linear[layer].weight.grad, alpha=(1 - momentum_rate))
			qkv_momentum_rms[layer] = qkv_linear_momentum_gradient[layer].square().mean().sqrt()
			qkv_linear_rms = qkv_linear[layer].weight.square().mean().sqrt()
			qkv_log_ratio[layer] = (torch.log(qkv_linear_rms / (qkv_momentum_rms[layer] + epsilon)))

		qkv_log_average = qkv_log_ratio.mean()
		qkv_std = qkv_log_ratio.std()
		qkv_zscore = (qkv_log_ratio - qkv_log_average) / (qkv_std + epsilon)
		qkv_norm_power = pmax * (1 - 2 ** (-qkv_zscore.abs()))
		qkv_ratio = torch.exp(qkv_log_ratio - qkv_log_average)
		qkv_momentum_ratio = (qkv_reference_momentum_rms / (qkv_momentum_rms + epsilon)) ** global_norm_power

		# Store QKV derived stats and multiplier for external debugging/printing
		qkv_zscore_table.copy_(qkv_zscore)
		qkv_norm_power_table.copy_(qkv_norm_power)
		qkv_ratio_table.copy_(qkv_ratio)
		qkv_momentum_multiplier.copy_(qkv_momentum_ratio)
		qkv_multiplier.copy_(qkv_ratio ** qkv_norm_power * qkv_momentum_ratio)

		#& QKV layers
		for layer in range(number_of_layers):
			qkv_linear[layer].weight.sub_(
				qkv_ratio[layer] ** qkv_norm_power[layer]
				* qkv_momentum_ratio[layer]
				* qkv_linear_momentum_gradient[layer]
				* normalized_learning_rate
			)

		#& Output table
		output_weight_table_momentum_gradient.mul_(momentum_rate)
		output_weight_table_momentum_gradient.add_(output_weight_table.grad, alpha=(1 - momentum_rate))
		output_weight_table.sub_(output_weight_table_momentum_gradient * normalized_learning_rate)

		#& FFN layers
		# Use the same log/std based normalization strategy as QKV for FFN up/down
		ffn_up_log_ratio = torch.zeros(number_of_layers, device=device)
		ffn_down_log_ratio = torch.zeros(number_of_layers, device=device)
		ffn_up_momentum_rms = torch.zeros(number_of_layers, device=device)
		ffn_down_momentum_rms = torch.zeros(number_of_layers, device=device)

		for layer in range(number_of_layers):
			ffn_up_linear_momentum_gradient[layer].mul_(momentum_rate)
			ffn_up_linear_momentum_gradient[layer].add_(ffn_up_linear[layer].weight.grad, alpha=(1 - momentum_rate))
			ffn_up_momentum_rms[layer] = ffn_up_linear_momentum_gradient[layer].square().mean().sqrt()
			ffn_up_linear_rms = ffn_up_linear[layer].weight.square().mean().sqrt()
			ffn_up_log_ratio[layer] = torch.log(ffn_up_linear_rms / (ffn_up_momentum_rms[layer] + epsilon))

			ffn_down_linear_momentum_gradient[layer].mul_(momentum_rate)
			ffn_down_linear_momentum_gradient[layer].add_(ffn_down_linear[layer].weight.grad, alpha=(1 - momentum_rate))
			ffn_down_momentum_rms[layer] = ffn_down_linear_momentum_gradient[layer].square().mean().sqrt()
			ffn_down_linear_rms = ffn_down_linear[layer].weight.square().mean().sqrt()
			ffn_down_log_ratio[layer] = torch.log(ffn_down_linear_rms / (ffn_down_momentum_rms[layer] + epsilon))

		ffn_up_log_average = ffn_up_log_ratio.mean()
		ffn_up_std = ffn_up_log_ratio.std()
		ffn_up_zscore_local = (ffn_up_log_ratio - ffn_up_log_average) / (ffn_up_std + epsilon)
		ffn_up_norm_power_local = pmax * (1 - 2 ** (-ffn_up_zscore_local.abs()))
		ffn_up_ratio_local = torch.exp(ffn_up_log_ratio - ffn_up_log_average)
		ffn_up_momentum_ratio = (ffn_up_reference_momentum_rms / (ffn_up_momentum_rms + epsilon)) ** global_norm_power

		ffn_down_log_average = ffn_down_log_ratio.mean()
		ffn_down_std = ffn_down_log_ratio.std()
		ffn_down_zscore_local = (ffn_down_log_ratio - ffn_down_log_average) / (ffn_down_std + epsilon)
		ffn_down_norm_power_local = pmax * (1 - 2 ** (-ffn_down_zscore_local.abs()))
		ffn_down_ratio_local = torch.exp(ffn_down_log_ratio - ffn_down_log_average)
		ffn_down_momentum_ratio = (ffn_down_reference_momentum_rms / (ffn_down_momentum_rms + epsilon)) ** global_norm_power

		# Store FFN up/down stats and multipliers for debugging/printing (copy from local)
		ffn_up_zscore.copy_(ffn_up_zscore_local)
		ffn_up_norm_power.copy_(ffn_up_norm_power_local)
		ffn_up_ratio.copy_(ffn_up_ratio_local)
		ffn_up_momentum_multiplier.copy_(ffn_up_momentum_ratio)
		ffn_up_multiplier.copy_(ffn_up_ratio_local ** ffn_up_norm_power_local * ffn_up_momentum_ratio)

		ffn_down_zscore.copy_(ffn_down_zscore_local)
		ffn_down_norm_power.copy_(ffn_down_norm_power_local)
		ffn_down_ratio.copy_(ffn_down_ratio_local)
		ffn_down_momentum_multiplier.copy_(ffn_down_momentum_ratio)
		ffn_down_multiplier.copy_(ffn_down_ratio_local ** ffn_down_norm_power_local * ffn_down_momentum_ratio)

		for layer in range(number_of_layers):
			ffn_up_linear[layer].weight.sub_(
				ffn_up_ratio_local[layer] ** ffn_up_norm_power_local[layer]
				* ffn_up_momentum_ratio[layer]
				* ffn_up_linear_momentum_gradient[layer]
				* normalized_learning_rate
			)

			ffn_down_linear[layer].weight.sub_(
				ffn_down_ratio_local[layer] ** ffn_down_norm_power_local[layer]
				* ffn_down_momentum_ratio[layer]
				* ffn_down_linear_momentum_gradient[layer]
				* normalized_learning_rate
			)

		#& Wo layers — apply log/std normalization like QKV
		Wo_log_ratio = torch.zeros(number_of_layers, device=device)
		Wo_momentum_rms = torch.zeros(number_of_layers, device=device)

		for layer in range(number_of_layers):
			Wo_linear_momentum_gradient[layer].mul_(momentum_rate)
			Wo_linear_momentum_gradient[layer].add_(Wo_linear[layer].weight.grad, alpha=(1 - momentum_rate))
			Wo_momentum_rms[layer] = Wo_linear_momentum_gradient[layer].square().mean().sqrt()
			Wo_linear_rms = Wo_linear[layer].weight.square().mean().sqrt()
			Wo_log_ratio[layer] = torch.log(Wo_linear_rms / (Wo_momentum_rms[layer] + epsilon))

		Wo_log_average = Wo_log_ratio.mean()
		Wo_std = Wo_log_ratio.std()
		Wo_zscore_local = (Wo_log_ratio - Wo_log_average) / (Wo_std + epsilon)
		Wo_norm_power_local = pmax * (1 - 2 ** (-Wo_zscore_local.abs()))
		Wo_ratio_local = torch.exp(Wo_log_ratio - Wo_log_average)
		Wo_momentum_ratio = (Wo_reference_momentum_rms / (Wo_momentum_rms + epsilon)) ** global_norm_power

		for layer in range(number_of_layers):
			Wo_linear[layer].weight.sub_(
				Wo_ratio_local[layer] ** Wo_norm_power_local[layer]
				* Wo_momentum_ratio[layer]
				* Wo_linear_momentum_gradient[layer]
				* normalized_learning_rate
			)

		# Store Wo stats and multipliers for debugging/printing (copy from local)
		Wo_zscore.copy_(Wo_zscore_local)
		Wo_norm_power.copy_(Wo_norm_power_local)
		Wo_ratio.copy_(Wo_ratio_local)
		Wo_momentum_multiplier.copy_(Wo_momentum_ratio)
		Wo_multiplier.copy_(Wo_ratio_local ** Wo_norm_power_local * Wo_momentum_ratio)

		#& Gamma tables
		attention_gamma_table_momentum_gradient.mul_(momentum_rate)
		attention_gamma_table_momentum_gradient.add_(attention_gamma_table.grad, alpha=(1 - momentum_rate))
		attention_gamma_table.sub_(attention_gamma_table_momentum_gradient * normalized_learning_rate)

		ffn_gamma_table_momentum_gradient.mul_(momentum_rate)
		ffn_gamma_table_momentum_gradient.add_(ffn_gamma_table.grad, alpha=(1 - momentum_rate))
		ffn_gamma_table.sub_(ffn_gamma_table_momentum_gradient * normalized_learning_rate)

		token_embedding_table.grad = None
		positional_embedding_table.grad = None
		for layer in range(number_of_layers):
			qkv_linear[layer].weight.grad = None
			Wo_linear[layer].weight.grad = None
		output_weight_table.grad = None
		attention_gamma_table.grad = None
		ffn_gamma_table.grad = None
		for layer in range(number_of_layers):
			ffn_up_linear[layer].weight.grad = None
			ffn_down_linear[layer].weight.grad = None

def training_step(start_positions, source_token_ids):
	window_loss = run_forward_pass(start_positions, source_token_ids)
	window_loss.backward()
	return window_loss


def set_learning_rate(loss):
	learning_rate = 0.02
	if (loss < 1.5):
		learning_rate = 0.010
	if (loss < 1.46):
		learning_rate = 0.005
	if (loss < 1.4):
		learning_rate = 0.0025
	if (loss < 1.36):
		learning_rate = 0.0025 * 0.75
	if (loss < 1.3):
		learning_rate = 0.00125
	if (loss < 1.28):
		learning_rate = 0.00125 / 2
	if (loss < 1.25):
		learning_rate = 0.00125 / 4
	learning_rate *= learning_user_scale * persistent_learning_rate
	learning_rate_tensor.fill_(learning_rate)	

torch._dynamo.config.compiled_autograd = True
max_compile = False
compiled_training_step = torch.compile(
	training_step,
	# max-autotune for long uses. default mode for testing
	mode="max-autotune" if max_compile else "default",
	# mode="default",
	dynamic=False,
)

compiled_update_weights = torch.compile(
	update_weights,
	# max-autotune-no-cudagraphs for long uses. default mode for testing
	mode="max-autotune" if max_compile else "default",
	# mode="default",
	dynamic=False
)


#! Settings                                                 
embedding_size = 512
CONTEXT_LENGTH = 384
batch_size = 24
gradient_scale = 256
head_count = 4
ffn_hidden_multiplier = [4, 3, 3, 3, 3, 3, 3, 3]
momentum_rate = 0.9
pmax = 0.85        # relative layer correction
global_norm_power = 0.45      # whole-family magnitude correction
rms_scale_max = 2.5
calibration_steps = 100
calibration_burn_in_steps = 50
number_of_training_windows = len(token_ids) - CONTEXT_LENGTH
load_persistent_learning = True


example_start_positions = torch.arange(CONTEXT_LENGTH, CONTEXT_LENGTH * (batch_size + 1), step=CONTEXT_LENGTH)[:, None]
number_of_layers = len(ffn_hidden_multiplier)
learning_rate = 0.001
learning_user_scale = 2.0

#& Initialization
if (1):
	if (embedding_size % head_count != 0):
		raise ValueError(
			f"Head_count: {head_count}\n"
			f"Embedding_size / head_count (head_size): {embedding_size / head_count}"
		)
	#% Shape: vocabulary_size × embedding_size. 
	#* This is the map for all characters in the vocabulary to their respective vector, initialized random at first and changes later to learn relationships.
	ffn_hidden_size = [ffn_hidden_multiplier[i] * embedding_size for i in range(number_of_layers)]
	scale = 1 / math.sqrt(embedding_size)
	ffn_up_scale = 1 / math.sqrt(embedding_size)
	ffn_down_scale = [1 / math.sqrt(ffn_hidden_size[i]) for i in range(number_of_layers)]

	token_embedding_table = nn.Parameter(((torch.rand(number_of_ids, embedding_size) * (2 * scale) - scale) * 16).to(device))

	#% Shape: context_length x embedding_size
	#* This gives each possible position in the context window its own learnable vector, so the model can distinguish position 0 from position 1, position 2, etc.
	positional_embedding_table = nn.Parameter(((torch.rand(CONTEXT_LENGTH, embedding_size) * (2 * scale) - scale) * 16).to(device))

	#% Shape: 3 x embedding_size x embedding_size
	qkv_linear = nn.ModuleList([
		nn.Linear(embedding_size, 3 * embedding_size, bias=False, device=device)
		for _ in range(number_of_layers)
	])

	#% Shape: number_of_ids x embedding_size
	#* Determines/asks how much an attention_output vector supports a character being next
	output_weight_table = nn.Parameter((torch.rand(number_of_ids, embedding_size, device=device) * (2 * scale) - scale))

	ffn_up_linear = nn.ModuleList([
		nn.Linear(embedding_size, ffn_hidden_size[layer], bias=False, device=device)
		for layer in range(number_of_layers)
	])

	ffn_down_linear = nn.ModuleList([
		nn.Linear(ffn_hidden_size[layer], embedding_size, bias=False, device=device)
		for layer in range(number_of_layers)
	])

	Wo_linear = nn.ModuleList([
		nn.Linear(embedding_size, embedding_size, bias=False, device=device)
		for _ in range(number_of_layers)
	])

	attention_gamma_table = nn.Parameter(torch.ones(number_of_layers, embedding_size, device=device))
	ffn_gamma_table = nn.Parameter(torch.ones(number_of_layers, embedding_size, device=device))

	#% Shape: context_length x context_length
	#* Shows what positions a given position is allowed to look at
	future_mask = torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1).bool()

	#$ Momentum Gradients
	token_embedding_table_momentum_gradient = torch.zeros(number_of_ids, embedding_size, device=device)
	
	positional_embedding_table_momentum_gradient = torch.zeros(CONTEXT_LENGTH, embedding_size, device=device)

	qkv_linear_momentum_gradient = torch.zeros(number_of_layers, 3 * embedding_size, embedding_size, device=device)

	output_weight_table_momentum_gradient = torch.zeros(number_of_ids, embedding_size, device=device)

	ffn_up_linear_momentum_gradient = [torch.zeros(ffn_hidden_size[i], embedding_size, device=device) for i in range(number_of_layers)]
	ffn_down_linear_momentum_gradient = [torch.zeros(embedding_size, ffn_hidden_size[i], device=device) for i in range(number_of_layers)]	
	Wo_linear_momentum_gradient = [torch.zeros(embedding_size, embedding_size, device=device) for _ in range(number_of_layers)]	

	attention_gamma_table_momentum_gradient = torch.zeros(number_of_layers, embedding_size, device=device)
	ffn_gamma_table_momentum_gradient = torch.zeros(number_of_layers, embedding_size, device=device)

	future_mask = future_mask.to(device)

	batch_indices = torch.arange(batch_size, device=device, dtype=torch.long)[:, None]
	number_of_ids_indices = torch.arange(number_of_ids, device=device, dtype=torch.long)
	context_indices = torch.arange(CONTEXT_LENGTH, dtype=torch.long, device=device)[None, :]
	context_indices_plus1 = torch.arange(CONTEXT_LENGTH + 1, dtype=torch.long)[None, :]
	learning_rate_tensor = torch.tensor(learning_rate, device=device, dtype=torch.float32)
	# Per-layer debug multipliers and stats (updated each weight update)
	qkv_multiplier = torch.ones(number_of_layers, device=device)
	qkv_zscore_table = torch.zeros(number_of_layers, device=device)
	qkv_norm_power_table = torch.zeros(number_of_layers, device=device)
	qkv_ratio_table = torch.ones(number_of_layers, device=device)
    
	# Per-layer momentum-only multipliers for debugging
	qkv_momentum_multiplier = torch.ones(number_of_layers, device=device)

	ffn_up_multiplier = torch.ones(number_of_layers, device=device)
	ffn_up_zscore = torch.zeros(number_of_layers, device=device)
	ffn_up_norm_power = torch.zeros(number_of_layers, device=device)
	ffn_up_ratio = torch.ones(number_of_layers, device=device)

	ffn_up_momentum_multiplier = torch.ones(number_of_layers, device=device)

	ffn_down_multiplier = torch.ones(number_of_layers, device=device)
	ffn_down_zscore = torch.zeros(number_of_layers, device=device)
	ffn_down_norm_power = torch.zeros(number_of_layers, device=device)
	ffn_down_ratio = torch.ones(number_of_layers, device=device)

	ffn_down_momentum_multiplier = torch.ones(number_of_layers, device=device)

	Wo_multiplier = torch.ones(number_of_layers, device=device)
	Wo_zscore = torch.zeros(number_of_layers, device=device)
	Wo_norm_power = torch.zeros(number_of_layers, device=device)
	Wo_ratio = torch.ones(number_of_layers, device=device)

	Wo_momentum_multiplier = torch.ones(number_of_layers, device=device)

	# Fixed optimizer baselines. Fresh models fill these during calibration; checkpoints load them.
	qkv_initial_weight_rms = torch.zeros(number_of_layers, device=device)
	qkv_reference_momentum_rms = torch.zeros(number_of_layers, device=device)
	ffn_up_initial_weight_rms = torch.zeros(number_of_layers, device=device)
	ffn_up_reference_momentum_rms = torch.zeros(number_of_layers, device=device)
	ffn_down_initial_weight_rms = torch.zeros(number_of_layers, device=device)
	ffn_down_reference_momentum_rms = torch.zeros(number_of_layers, device=device)
	Wo_initial_weight_rms = torch.zeros(number_of_layers, device=device)
	Wo_reference_momentum_rms = torch.zeros(number_of_layers, device=device)


iteration = 0
loss_window_average = 0.0
window_loss_size = 1000
checkpoint_interval = 30000
max_training_hours = 3
learning_rate_patience = 20
last_improvement_window = 0
persistent_learning_rate = 1
number_of_validation_batches = 32


validation_generator = torch.Generator()
validation_generator.manual_seed(20260813)

fixed_validation_positions = torch.randint(
    high=len(validation_token_ids) - CONTEXT_LENGTH,
    size=(number_of_validation_batches, batch_size, 1),
    dtype=torch.long,
    generator=validation_generator,
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_path = CHECKPOINT_DIR / f"{CONTEXT_LENGTH}x{embedding_size}_{number_of_layers}lr_atNorm_{head_count}hd_mon_rms.pt"
checkpoint_loaded = False

if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
	iteration, current_least_loss, persistent_learning_rate, previous_elapsed_seconds = load_model(checkpoint_path)
	checkpoint_loaded = True
	if(not load_persistent_learning):
		persistent_learning_rate = 1
else:
	print("No valid checkpoint found. Starting training from scratch.")

	iteration = 0
	current_least_loss = float("inf")
	previous_elapsed_seconds = 0
	persistent_learning_rate = 1


# iteration = 0
# current_least_loss = float("inf")
# previous_elapsed_seconds = 0
# persistent_learning_rate = 1

previous_least_loss = current_least_loss


try:
	with open("learning_scale.txt", "r") as file:
		learning_user_scale = float(file.read().strip())
except (OSError, ValueError):
	print("Could not read learning scale; keeping previous value.")

window_start_time = time.perf_counter()
current_start_time = time.perf_counter()

forward_time = 0
backward_time = 0

set_learning_rate(current_least_loss)

fp8_config = Float8LinearConfig.from_recipe_name("tensorwise")
fp8_config = replace(
	fp8_config,
	force_recompute_fp8_weight_in_bwd=False,
	gemm_config_grad_input=Float8GemmConfig(use_fast_accum=True),
	gemm_config_grad_weight=Float8GemmConfig(use_fast_accum=True),
)

convert_to_float8_training(qkv_linear, config=fp8_config)
convert_to_float8_training(ffn_up_linear, config=fp8_config)
convert_to_float8_training(ffn_down_linear, config=fp8_config)
convert_to_float8_training(Wo_linear, config=fp8_config)

#& Fresh-model optimizer calibration
if not checkpoint_loaded:
	print(f"Calibrating optimizer baseline for {calibration_steps} steps...")
	calibration_generator = torch.Generator().manual_seed(20260815)

	with torch.no_grad():
		qkv_initial_weight_rms.copy_(torch.stack([m.weight.square().mean().sqrt() for m in qkv_linear]))
		ffn_up_initial_weight_rms.copy_(torch.stack([m.weight.square().mean().sqrt() for m in ffn_up_linear]))
		ffn_down_initial_weight_rms.copy_(torch.stack([m.weight.square().mean().sqrt() for m in ffn_down_linear]))
		Wo_initial_weight_rms.copy_(torch.stack([m.weight.square().mean().sqrt() for m in Wo_linear]))

	measurements = 0
	for calibration_step in range(calibration_steps):
		torch.compiler.cudagraph_mark_step_begin()
		start_positions = torch.randint(number_of_training_windows, (batch_size,), generator=calibration_generator, dtype=torch.long)[:, None]
		with torch.autocast("cuda", dtype=torch.bfloat16):
			compiled_training_step(start_positions, token_ids)

		with torch.no_grad():
			token_embedding_table_momentum_gradient.mul_(momentum_rate).add_(token_embedding_table.grad, alpha=1 - momentum_rate)
			positional_embedding_table_momentum_gradient.mul_(momentum_rate).add_(positional_embedding_table.grad, alpha=1 - momentum_rate)
			output_weight_table_momentum_gradient.mul_(momentum_rate).add_(output_weight_table.grad, alpha=1 - momentum_rate)
			attention_gamma_table_momentum_gradient.mul_(momentum_rate).add_(attention_gamma_table.grad, alpha=1 - momentum_rate)
			ffn_gamma_table_momentum_gradient.mul_(momentum_rate).add_(ffn_gamma_table.grad, alpha=1 - momentum_rate)

			for layer in range(number_of_layers):
				qkv_linear_momentum_gradient[layer].mul_(momentum_rate).add_(qkv_linear[layer].weight.grad, alpha=1 - momentum_rate)
				ffn_up_linear_momentum_gradient[layer].mul_(momentum_rate).add_(ffn_up_linear[layer].weight.grad, alpha=1 - momentum_rate)
				ffn_down_linear_momentum_gradient[layer].mul_(momentum_rate).add_(ffn_down_linear[layer].weight.grad, alpha=1 - momentum_rate)
				Wo_linear_momentum_gradient[layer].mul_(momentum_rate).add_(Wo_linear[layer].weight.grad, alpha=1 - momentum_rate)

			if calibration_step >= calibration_burn_in_steps:
				qkv_reference_momentum_rms.add_(qkv_linear_momentum_gradient.square().mean(dim=(1, 2)).sqrt())
				ffn_up_reference_momentum_rms.add_(torch.stack([m.square().mean().sqrt() for m in ffn_up_linear_momentum_gradient]))
				ffn_down_reference_momentum_rms.add_(torch.stack([m.square().mean().sqrt() for m in ffn_down_linear_momentum_gradient]))
				Wo_reference_momentum_rms.add_(torch.stack([m.square().mean().sqrt() for m in Wo_linear_momentum_gradient]))
				measurements += 1

			token_embedding_table.grad = positional_embedding_table.grad = output_weight_table.grad = None
			attention_gamma_table.grad = ffn_gamma_table.grad = None
			for layer in range(number_of_layers):
				qkv_linear[layer].weight.grad = ffn_up_linear[layer].weight.grad = None
				ffn_down_linear[layer].weight.grad = Wo_linear[layer].weight.grad = None

	qkv_reference_momentum_rms.div_(measurements)
	ffn_up_reference_momentum_rms.div_(measurements)
	ffn_down_reference_momentum_rms.div_(measurements)
	Wo_reference_momentum_rms.div_(measurements)

	# Save immediately so the fixed baseline survives restarts before the first validation checkpoint.
	save_model(checkpoint_path, iteration=0, best_loss=float("inf"), previous_elapsed_seconds=0)
	print("Optimizer baseline calibrated and saved.")

# Do not count calibration/compile time as training time.
window_start_time = time.perf_counter()
current_start_time = time.perf_counter()


#! Training loop                 
while True:
	torch.compiler.cudagraph_mark_step_begin()

	iteration += 1
	
	start_positions = torch.randint(high = number_of_training_windows, size = (batch_size,), dtype = torch.long)[:, None]

	with torch.autocast("cuda", dtype=torch.bfloat16):
		window_loss = compiled_training_step(start_positions, token_ids)

	compiled_update_weights()

	if iteration % (window_loss_size * 7) == 0:
		with torch.no_grad():
			print("\n--- Multipliers ---")
			for layer in range(number_of_layers):
				# QKV
				z = qkv_zscore_table[layer].item()
				p = qkv_norm_power_table[layer].item()
				ratio = qkv_ratio_table[layer].item()
				mult = qkv_multiplier[layer].item()
				mom = qkv_momentum_multiplier[layer].item()
				p_str = f"{p:.3f}".lstrip("0")
				print(f"QKV L{layer} | z={z:+.2f} | p={p_str} | ratio={ratio:.2f} | multiplier={mult:.3f} | mom={mom:.3f}")

			# FFN up/down
			for layer in range(number_of_layers):
				z = ffn_up_zscore[layer].item()
				p = ffn_up_norm_power[layer].item()
				ratio = ffn_up_ratio[layer].item()
				mult = ffn_up_multiplier[layer].item()
				mom = ffn_up_momentum_multiplier[layer].item()
				p_str = f"{p:.3f}".lstrip("0")
				print(f"UP  L{layer} | z={z:+.2f} | p={p_str} | ratio={ratio:.2f} | multiplier={mult:.3f} | mom={mom:.3f}")

			for layer in range(number_of_layers):
				z = ffn_down_zscore[layer].item()
				p = ffn_down_norm_power[layer].item()
				ratio = ffn_down_ratio[layer].item()
				mult = ffn_down_multiplier[layer].item()
				mom = ffn_down_momentum_multiplier[layer].item()
				p_str = f"{p:.3f}".lstrip("0")
				print(f"DOWN L{layer} | z={z:+.2f} | p={p_str} | ratio={ratio:.2f} | multiplier={mult:.3f} | mom={mom:.3f}")

			# Wo
			for layer in range(number_of_layers):
				z = Wo_zscore[layer].item()
				p = Wo_norm_power[layer].item()
				ratio = Wo_ratio[layer].item()
				mult = Wo_multiplier[layer].item()
				mom = Wo_momentum_multiplier[layer].item()
				p_str = f"{p:.3f}".lstrip("0")
				print(f"Wo  L{layer} | z={z:+.2f} | p={p_str} | ratio={ratio:.2f} | multiplier={mult:.3f} | mom={mom:.3f}")

	if max_training_hours is not None:
		current_session_elapsed_seconds = time.perf_counter() - current_start_time
		if current_session_elapsed_seconds >= max_training_hours * 3600:
			total_elapsed_seconds = (
				previous_elapsed_seconds + current_session_elapsed_seconds
			)
			print(
				f"Training time limit reached: "
				f"{current_session_elapsed_seconds / 3600:.2f} hours "
				"in this session."
			)
			save_model(
				checkpoint_path,
				iteration=iteration,
				best_loss=current_least_loss,
				previous_elapsed_seconds=total_elapsed_seconds,
			)
			print("Final checkpoint saved. Training stopped.")
			raise SystemExit(0)

	loss_window_average += window_loss.detach()

	if iteration % window_loss_size == 0:
		last_improvement_window += 1

		average_loss = (loss_window_average / window_loss_size).item()
			
		window_elapsed_time = (
			time.perf_counter() - window_start_time
		)

		current_elapsed_seconds = (
			time.perf_counter() - current_start_time
		)

		iterations_per_second = (
			window_loss_size / window_elapsed_time
		)

		print(
			f"Iterations "
			f"{iteration - window_loss_size + 1}-{iteration} "
			f"took {window_elapsed_time:.2f} seconds"
		)

		print(
			f"Iteration: {iteration}, "
			f"average loss: {average_loss:.6f}, "
			f"speed: {iterations_per_second:.2f} "
			f"iterations/sec, "
			f"elapsed: {(current_elapsed_seconds + previous_elapsed_seconds)/ 3600:.2f} hours"
		)
		with torch.no_grad():
			with torch.autocast(device_type = "cuda", dtype = torch.bfloat16):
				validation_average_loss = 0
				for validation_start_positions in fixed_validation_positions:
					validation_forward_pass_results = run_forward_pass(validation_start_positions, validation_token_ids)

					validation_average_loss += validation_forward_pass_results

				validation_average_loss /= number_of_validation_batches

		validation_average_loss = validation_average_loss.item()
		print(f"Validation loss at iteration {iteration}: {(validation_average_loss)}")

		if validation_average_loss < current_least_loss:
			previous_least_loss = current_least_loss
			current_least_loss = validation_average_loss
			last_improvement_window = 0

			save_model(
				CHECKPOINT_DIR / f"{CONTEXT_LENGTH}x{embedding_size}_{number_of_layers}lr_atNorm_{head_count}hd_mon_rms.pt",
				iteration=iteration,
				best_loss=current_least_loss,
				previous_elapsed_seconds=previous_elapsed_seconds + current_elapsed_seconds,
			)

			print(
				f"New best model saved with loss: "
				f"{current_least_loss:.6f}"
			)
			print(
				f"Loss difference new-previous best model:"
				f"{current_least_loss - previous_least_loss}"
			)

		# Reduce the learning rate if validation has not improved.
		if (
			last_improvement_window >= learning_rate_patience):
			persistent_learning_rate *= 0.9
			print(
				f"No improvement for {learning_rate_patience} loss windows. Setting learning rate to {round(learning_rate_tensor.item() * 0.9, 6)} and persistence_learning_rate to {round(persistent_learning_rate, 6)}. ")

			# Start a new patience period after rollback.
			last_improvement_window = 0

		if iteration % checkpoint_interval == 0:
			save_model(
				CHECKPOINT_DIR / f"checkpoint_{iteration}_{CONTEXT_LENGTH}_{embedding_size}.pt",
				iteration=iteration,
				best_loss=current_least_loss,
				previous_elapsed_seconds=previous_elapsed_seconds + current_elapsed_seconds
			)

		try:
			with open("learning_scale.txt", "r") as file:
				learning_user_scale = float(file.read().strip())
		except (OSError, ValueError):
			print("Could not read learning scale; keeping previous value.")

		set_learning_rate(current_least_loss)

		loss_window_average = 0.0
		window_start_time = time.perf_counter()