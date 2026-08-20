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

	"°": 96, "£": 97, "€": 98, "©": 99, "®": 100, "™": 101, "^": 102,
	"½": 103, "²": 104, "μ": 105, "π": 106,

	"±": 107, "×": 108, "÷": 109, "≈": 110, "≠": 111,
	"≤": 112, "≥": 113, "√": 114, "∞": 115,
	"∑": 116, "∏": 117, "∫": 118, "∂": 119,

	"α": 120, "β": 121, "γ": 122, "δ": 123,
	"Δ": 124, "λ": 125, "θ": 126, "σ": 127, "Ω": 128,

	"→": 129, "←": 130, "↑": 131, "↓": 132, "↔": 133,

	"¹": 134, "³": 135,
	"⁰": 136, "⁴": 137, "⁵": 138, "⁶": 139, "⁷": 140, "⁸": 141, "⁹": 142,

	"₀": 143, "₁": 144, "₂": 145, "₃": 146, "₄": 147,
	"₅": 148, "₆": 149, "₇": 150, "₈": 151, "₉": 152,

	"′": 153, "″": 154,
	"¼": 155, "¾": 156, "⅓": 157, "⅔": 158,
	"⅛": 159, "⅜": 160, "⅝": 161, "⅞": 162,

	"∈": 163, "∉": 164, "⊂": 165, "⊆": 166, "⊃": 167, "⊇": 168,
	"∪": 169, "∩": 170, "∅": 171, "∝": 172, "∴": 173, "∵": 174,
	"∇": 175, "∥": 176, "⊥": 177, "≡": 178, "≅": 179,

	"ε": 180, "η": 181, "κ": 182, "ρ": 183, "τ": 184, "φ": 185,
	"χ": 186, "ψ": 187, "ω": 188,

	"℃": 189, "℉": 190,
	"¢": 191, "¥": 192,

	"§": 193, "¶": 194, "†": 195, "‡": 196,
	"…": 197, "•": 198,

	"é": 199, "á": 200, "í": 201, "ó": 202, "ú": 203,
	"ñ": 204, "ü": 205, "ö": 206, "ä": 207, "ç": 208,

	"É": 209, "Á": 210, "Í": 211, "Ó": 212, "Ú": 213,
	"Ñ": 214, "Ü": 215, "Ö": 216, "Ä": 217, "Ç": 218,

	"æ": 219, "·": 220,
}

id_to_character = {
	token_id: character
	for character, token_id in character_to_id.items()
}

number_of_ids = len(character_to_id)

validation_size = 1_000_000

data = np.memmap(
	"finewebedu_uint8.bin",
	dtype=np.uint8,
	mode="r"
)
training_data = data[:-validation_size]
validation_data = data[-validation_size:]

training_token_ids = torch.from_numpy(training_data)
validation_token_ids = torch.tensor(validation_data, dtype=torch.uint8)

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
		"frequency_base": frequency_base,
		"pmax_layer": pmax_layer,
		"qkv_initial_weight_rms": qkv_initial_weight_rms.detach().cpu(),
		"query_reference_momentum_rms": query_reference_momentum_rms.detach().cpu(),
		"key_reference_momentum_rms": key_reference_momentum_rms.detach().cpu(),
		"value_reference_momentum_rms": value_reference_momentum_rms.detach().cpu(),
		"query_reference_momentum_column_rms": query_reference_momentum_column_rms.detach().cpu(),
		"key_reference_momentum_column_rms": key_reference_momentum_column_rms.detach().cpu(),
		"value_reference_momentum_column_rms": value_reference_momentum_column_rms.detach().cpu(),
		"ffn_up_initial_weight_rms": ffn_up_initial_weight_rms.detach().cpu(),
		"ffn_up_reference_momentum_rms": ffn_up_reference_momentum_rms.detach().cpu(),
		"ffn_up_reference_momentum_column_rms": ffn_up_reference_momentum_column_rms.detach().cpu(),
		"ffn_down_initial_weight_rms": ffn_down_initial_weight_rms.detach().cpu(),
		"ffn_down_reference_momentum_rms": ffn_down_reference_momentum_rms.detach().cpu(),
		"ffn_down_reference_momentum_column_rms": ffn_down_reference_momentum_column_rms.detach().cpu(),
		"Wo_initial_weight_rms": Wo_initial_weight_rms.detach().cpu(),
		"Wo_reference_momentum_rms": Wo_reference_momentum_rms.detach().cpu(),
		"Wo_reference_momentum_column_rms": Wo_reference_momentum_column_rms.detach().cpu(),

		"token_embedding_table": token_embedding_table.detach().cpu(),
		"output_weight_table": output_weight_table.detach().cpu(),
		"attention_gamma_table": attention_gamma_table.detach().cpu(),
		"ffn_gamma_table": ffn_gamma_table.detach().cpu(),

		"momentum_rate": momentum_rate,

		"token_embedding_table_momentum_gradient":
			token_embedding_table_momentum_gradient.detach().cpu(),

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
	global qkv_linear
	global output_weight_table
	global ffn_up_linear
	global ffn_down_linear
	global attention_gamma_table
	global ffn_gamma_table
	global Wo_linear
	global token_embedding_table_momentum_gradient
	global qkv_linear_momentum_gradient
	global output_weight_table_momentum_gradient
	global ffn_up_linear_momentum_gradient
	global ffn_down_linear_momentum_gradient
	global Wo_linear_momentum_gradient
	global attention_gamma_table_momentum_gradient
	global ffn_gamma_table_momentum_gradient
	global qkv_initial_weight_rms
	global query_reference_momentum_rms
	global key_reference_momentum_rms
	global value_reference_momentum_rms
	global query_reference_momentum_column_rms
	global key_reference_momentum_column_rms
	global value_reference_momentum_column_rms
	global ffn_up_initial_weight_rms, ffn_up_reference_momentum_rms, ffn_up_reference_momentum_column_rms
	global ffn_down_initial_weight_rms, ffn_down_reference_momentum_rms, ffn_down_reference_momentum_column_rms
	global Wo_initial_weight_rms, Wo_reference_momentum_rms, Wo_reference_momentum_column_rms

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

	if "frequency_base" not in checkpoint:
		raise ValueError("Cannot load checkpoint: frequency_base doesn't exist.\n")
	if checkpoint["frequency_base"] != frequency_base:
		raise ValueError(
			"Cannot load checkpoint: frequency_base differs.\n"
			f"Checkpoint: {checkpoint['frequency_base']}\n"
			f"Current: {frequency_base}"
		)

	if "pmax_layer" not in checkpoint:
		raise ValueError("Cannot load checkpoint: pmax_layer doesn't exist.\n")

	if "qkv_linear_momentum_gradient" not in checkpoint:
		raise ValueError("Checkpoint does not contain momentum buffers.")

	baseline_keys = (
		"qkv_initial_weight_rms",
		"query_reference_momentum_rms",
		"key_reference_momentum_rms",
		"value_reference_momentum_rms",
		"query_reference_momentum_column_rms",
		"key_reference_momentum_column_rms",
		"value_reference_momentum_column_rms",
		"ffn_up_initial_weight_rms",
		"ffn_up_reference_momentum_rms",
		"ffn_up_reference_momentum_column_rms",
		"ffn_down_initial_weight_rms",
		"ffn_down_reference_momentum_rms",
		"ffn_down_reference_momentum_column_rms",
		"Wo_initial_weight_rms",
		"Wo_reference_momentum_rms",
		"Wo_reference_momentum_column_rms",
	)
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
	output_weight_table = nn.Parameter(checkpoint["output_weight_table"].to(device))

	token_embedding_table_momentum_gradient = checkpoint[
		"token_embedding_table_momentum_gradient"
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
	query_reference_momentum_rms = checkpoint["query_reference_momentum_rms"].to(device)
	key_reference_momentum_rms = checkpoint["key_reference_momentum_rms"].to(device)
	value_reference_momentum_rms = checkpoint["value_reference_momentum_rms"].to(device)
	query_reference_momentum_column_rms = checkpoint["query_reference_momentum_column_rms"].to(device)
	key_reference_momentum_column_rms = checkpoint["key_reference_momentum_column_rms"].to(device)
	value_reference_momentum_column_rms = checkpoint["value_reference_momentum_column_rms"].to(device)
	ffn_up_initial_weight_rms = checkpoint["ffn_up_initial_weight_rms"].to(device)
	ffn_up_reference_momentum_rms = checkpoint["ffn_up_reference_momentum_rms"].to(device)
	ffn_up_reference_momentum_column_rms = checkpoint["ffn_up_reference_momentum_column_rms"].to(device)
	ffn_down_initial_weight_rms = checkpoint["ffn_down_initial_weight_rms"].to(device)
	ffn_down_reference_momentum_rms = checkpoint["ffn_down_reference_momentum_rms"].to(device)
	ffn_down_reference_momentum_column_rms = checkpoint["ffn_down_reference_momentum_column_rms"].to(device)
	Wo_initial_weight_rms = checkpoint["Wo_initial_weight_rms"].to(device)
	Wo_reference_momentum_rms = checkpoint["Wo_reference_momentum_rms"].to(device)
	Wo_reference_momentum_column_rms = checkpoint["Wo_reference_momentum_column_rms"].to(device)

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

	query_matrix = query_matrix.reshape(batch_size, CONTEXT_LENGTH, head_count, embedding_size // head_count // 2, 2).transpose(1, 2)
	key_matrix = key_matrix.reshape(batch_size, CONTEXT_LENGTH, head_count, embedding_size // head_count // 2, 2).transpose(1, 2)
	value_matrix = value_matrix.reshape(batch_size, CONTEXT_LENGTH, head_count, embedding_size // head_count).transpose(1, 2)

	query_first, query_second = torch.unbind(query_matrix, dim=-1)
	key_first, key_second = torch.unbind(key_matrix, dim=-1)

	rotated_query_first = query_first * cosine_table - query_second * sin_table
	rotated_query_second = query_first * sin_table + query_second * cosine_table
	rotated_key_first = key_first * cosine_table - key_second * sin_table
	rotated_key_second = key_first * sin_table + key_second * cosine_table

	query_matrix = torch.stack((rotated_query_first, rotated_query_second), dim=-1).reshape(batch_size, head_count, CONTEXT_LENGTH, embedding_size // head_count)
	key_matrix = torch.stack((rotated_key_first, rotated_key_second), dim=-1).reshape(batch_size, head_count, CONTEXT_LENGTH, embedding_size // head_count)
	
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

	layer_state = input_token_embeddings

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


def update_momentum():
	with torch.no_grad():
		#& Token embedding table
		token_embedding_table_momentum_gradient.mul_(momentum_rate)
		token_embedding_table_momentum_gradient.add_(token_embedding_table.grad, alpha=(1 - momentum_rate))

		#& QKV layers
		for layer in range(number_of_layers):
			qkv_linear_momentum_gradient[layer].mul_(momentum_rate)
			qkv_linear_momentum_gradient[layer].add_(qkv_linear[layer].weight.grad, alpha=(1 - momentum_rate))

		#& Output table
		output_weight_table_momentum_gradient.mul_(momentum_rate)
		output_weight_table_momentum_gradient.add_(output_weight_table.grad, alpha=(1 - momentum_rate))

		#& FFN layers
		for layer in range(number_of_layers):
			ffn_up_linear_momentum_gradient[layer].mul_(momentum_rate)
			ffn_up_linear_momentum_gradient[layer].add_(ffn_up_linear[layer].weight.grad, alpha=(1 - momentum_rate))
			ffn_down_linear_momentum_gradient[layer].mul_(momentum_rate)
			ffn_down_linear_momentum_gradient[layer].add_(ffn_down_linear[layer].weight.grad, alpha=(1 - momentum_rate))

		#& Wo layers
		for layer in range(number_of_layers):
			Wo_linear_momentum_gradient[layer].mul_(momentum_rate)
			Wo_linear_momentum_gradient[layer].add_(Wo_linear[layer].weight.grad, alpha=(1 - momentum_rate))

		#& Gamma tables
		attention_gamma_table_momentum_gradient.mul_(momentum_rate)
		attention_gamma_table_momentum_gradient.add_(attention_gamma_table.grad, alpha=(1 - momentum_rate))

		ffn_gamma_table_momentum_gradient.mul_(momentum_rate)
		ffn_gamma_table_momentum_gradient.add_(ffn_gamma_table.grad, alpha=(1 - momentum_rate))

def calculate_optimizer_stats():
	with torch.no_grad():

		query_layer_log_ratio = torch.zeros(number_of_layers, device=device)
		key_layer_log_ratio = torch.zeros(number_of_layers, device=device)
		value_layer_log_ratio = torch.zeros(number_of_layers, device=device)

		query_layer_momentum_rms = torch.zeros(number_of_layers, device=device)
		key_layer_momentum_rms = torch.zeros(number_of_layers, device=device)
		value_layer_momentum_rms = torch.zeros(number_of_layers, device=device)

		query_row_norm_power = torch.zeros(number_of_layers, embedding_size, device=device)
		key_row_norm_power = torch.zeros(number_of_layers, embedding_size, device=device)
		value_row_norm_power = torch.zeros(number_of_layers, embedding_size, device=device)

		query_row_ratio = torch.zeros(number_of_layers, embedding_size, device=device)
		key_row_ratio = torch.zeros(number_of_layers, embedding_size, device=device)
		value_row_ratio = torch.zeros(number_of_layers, embedding_size, device=device)

		query_column_norm_power = torch.zeros(number_of_layers, embedding_size, device=device)
		key_column_norm_power = torch.zeros(number_of_layers, embedding_size, device=device)
		value_column_norm_power = torch.zeros(number_of_layers, embedding_size, device=device)

		query_column_ratio = torch.zeros(number_of_layers, embedding_size, device=device)
		key_column_ratio = torch.zeros(number_of_layers, embedding_size, device=device)
		value_column_ratio = torch.zeros(number_of_layers, embedding_size, device=device)

		query_row_history_ratio = torch.zeros(number_of_layers, embedding_size, device=device)
		key_row_history_ratio = torch.zeros(number_of_layers, embedding_size, device=device)
		value_row_history_ratio = torch.zeros(number_of_layers, embedding_size, device=device)

		query_column_history_ratio = torch.zeros(number_of_layers, embedding_size, device=device)
		key_column_history_ratio = torch.zeros(number_of_layers, embedding_size, device=device)
		value_column_history_ratio = torch.zeros(number_of_layers, embedding_size, device=device)



		for layer in range(number_of_layers):
			query_momentum_gradient = qkv_linear_momentum_gradient[layer][:embedding_size]
			key_momentum_gradient = qkv_linear_momentum_gradient[layer][embedding_size : embedding_size * 2]
			value_momentum_gradient = qkv_linear_momentum_gradient[layer][embedding_size * 2:]
			query_matrix = qkv_linear[layer].weight[:embedding_size]
			key_matrix = qkv_linear[layer].weight[embedding_size : embedding_size * 2]
			value_matrix = qkv_linear[layer].weight[embedding_size * 2:]

			query_momentum_row_rms = query_momentum_gradient.square().mean(dim=1).sqrt()
			key_momentum_row_rms = key_momentum_gradient.square().mean(dim=1).sqrt()
			value_momentum_row_rms = value_momentum_gradient.square().mean(dim=1).sqrt()

			query_weight_row_rms = query_matrix.square().mean(dim=1).sqrt()
			key_weight_row_rms = key_matrix.square().mean(dim=1).sqrt()
			value_weight_row_rms = value_matrix.square().mean(dim=1).sqrt()

			query_row_log_ratio = torch.log(query_weight_row_rms / (query_momentum_row_rms + epsilon))
			key_row_log_ratio = torch.log(key_weight_row_rms / (key_momentum_row_rms + epsilon))
			value_row_log_ratio = torch.log(value_weight_row_rms / (value_momentum_row_rms + epsilon))

			query_row_log_average = query_row_log_ratio.mean()
			key_row_log_average = key_row_log_ratio.mean()
			value_row_log_average = value_row_log_ratio.mean()

			query_row_std = query_row_log_ratio.std()
			key_row_std = key_row_log_ratio.std()
			value_row_std = value_row_log_ratio.std()

			query_row_zscore = (query_row_log_ratio - query_row_log_average) / (query_row_std + epsilon)
			key_row_zscore = (key_row_log_ratio - key_row_log_average) / (key_row_std + epsilon)
			value_row_zscore = (value_row_log_ratio - value_row_log_average) / (value_row_std + epsilon)

			query_row_norm_power[layer] = pmax_row * (1 - 2 ** (-query_row_zscore.abs()))
			key_row_norm_power[layer] = pmax_row * (1 - 2 ** (-key_row_zscore.abs()))
			value_row_norm_power[layer] = pmax_row * (1 - 2 ** (-value_row_zscore.abs()))

			query_row_ratio[layer] = torch.exp(query_row_log_ratio - query_row_log_average)
			key_row_ratio[layer] = torch.exp(key_row_log_ratio - key_row_log_average)
			value_row_ratio[layer] = torch.exp(value_row_log_ratio - value_row_log_average)

			query_row_history_ratio[layer] = (query_reference_momentum_rms[layer] / (query_momentum_row_rms + epsilon)) ** global_norm_power
			key_row_history_ratio[layer] = (key_reference_momentum_rms[layer] / (key_momentum_row_rms + epsilon)) ** global_norm_power
			value_row_history_ratio[layer] = (value_reference_momentum_rms[layer] / (value_momentum_row_rms + epsilon)) ** global_norm_power

			#$ Column
			query_momentum_column_rms = query_momentum_gradient.square().mean(dim=0).sqrt()
			key_momentum_column_rms = key_momentum_gradient.square().mean(dim=0).sqrt()
			value_momentum_column_rms = value_momentum_gradient.square().mean(dim=0).sqrt()

			query_weight_column_rms = query_matrix.square().mean(dim=0).sqrt()
			key_weight_column_rms = key_matrix.square().mean(dim=0).sqrt()
			value_weight_column_rms = value_matrix.square().mean(dim=0).sqrt()

			query_column_log_ratio = torch.log(query_weight_column_rms / (query_momentum_column_rms + epsilon))
			key_column_log_ratio = torch.log(key_weight_column_rms / (key_momentum_column_rms + epsilon))
			value_column_log_ratio = torch.log(value_weight_column_rms / (value_momentum_column_rms + epsilon))

			query_column_log_average = query_column_log_ratio.mean()
			key_column_log_average = key_column_log_ratio.mean()
			value_column_log_average = value_column_log_ratio.mean()

			query_column_std = query_column_log_ratio.std()
			key_column_std = key_column_log_ratio.std()
			value_column_std = value_column_log_ratio.std()

			query_column_zscore = (query_column_log_ratio - query_column_log_average) / (query_column_std + epsilon)
			key_column_zscore = (key_column_log_ratio - key_column_log_average) / (key_column_std + epsilon)
			value_column_zscore = (value_column_log_ratio - value_column_log_average) / (value_column_std + epsilon)

			query_column_norm_power[layer] = pmax_column * (1 - 2 ** (-query_column_zscore.abs()))
			key_column_norm_power[layer] = pmax_column * (1 - 2 ** (-key_column_zscore.abs()))
			value_column_norm_power[layer] = pmax_column * (1 - 2 ** (-value_column_zscore.abs()))

			query_column_ratio[layer] = torch.exp(query_column_log_ratio - query_column_log_average)
			key_column_ratio[layer] = torch.exp(key_column_log_ratio - key_column_log_average)
			value_column_ratio[layer] = torch.exp(value_column_log_ratio - value_column_log_average)

			query_column_history_ratio[layer] = (query_reference_momentum_column_rms[layer] / (query_momentum_column_rms + epsilon)) ** global_norm_power
			key_column_history_ratio[layer] = (key_reference_momentum_column_rms[layer] / (key_momentum_column_rms + epsilon)) ** global_norm_power
			value_column_history_ratio[layer] = (value_reference_momentum_column_rms[layer] / (value_momentum_column_rms + epsilon)) ** global_norm_power

			#$ Layer

			query_layer_momentum_rms[layer] = query_momentum_row_rms.square().mean().sqrt()
			key_layer_momentum_rms[layer] = key_momentum_row_rms.square().mean().sqrt()
			value_layer_momentum_rms[layer] = value_momentum_row_rms.square().mean().sqrt()

			query_layer_weight_rms = query_weight_row_rms.square().mean().sqrt()
			key_layer_weight_rms = key_weight_row_rms.square().mean().sqrt()
			value_layer_weight_rms = value_weight_row_rms.square().mean().sqrt()

			query_layer_log_ratio[layer] = torch.log(query_layer_weight_rms / (query_layer_momentum_rms[layer] + epsilon))
			key_layer_log_ratio[layer] = torch.log(key_layer_weight_rms / (key_layer_momentum_rms[layer] + epsilon))
			value_layer_log_ratio[layer] = torch.log(value_layer_weight_rms / (value_layer_momentum_rms[layer] + epsilon))

		query_layer_log_average = query_layer_log_ratio.mean()
		query_layer_std = query_layer_log_ratio.std()
		query_layer_zscore = (query_layer_log_ratio - query_layer_log_average) / (query_layer_std + epsilon)
		query_layer_norm_power = pmax_layer * (1 - 2 ** (-query_layer_zscore.abs()))
		query_layer_ratio = torch.exp(query_layer_log_ratio - query_layer_log_average)

		key_layer_log_average = key_layer_log_ratio.mean()
		key_layer_std = key_layer_log_ratio.std()
		key_layer_zscore = (key_layer_log_ratio - key_layer_log_average) / (key_layer_std + epsilon)
		key_layer_norm_power = pmax_layer * (1 - 2 ** (-key_layer_zscore.abs()))
		key_layer_ratio = torch.exp(key_layer_log_ratio - key_layer_log_average)

		value_layer_log_average = value_layer_log_ratio.mean()
		value_layer_std = value_layer_log_ratio.std()
		value_layer_zscore = (value_layer_log_ratio - value_layer_log_average) / (value_layer_std + epsilon)
		value_layer_norm_power = pmax_layer * (1 - 2 ** (-value_layer_zscore.abs()))
		value_layer_ratio = torch.exp(value_layer_log_ratio - value_layer_log_average)

		



		#& FFN layers
		ffn_up_layer_log_ratio = torch.zeros(number_of_layers, device=device)
		ffn_down_layer_log_ratio = torch.zeros(number_of_layers, device=device)

		ffn_up_layer_momentum_rms = torch.zeros(number_of_layers, device=device)
		ffn_down_layer_momentum_rms = torch.zeros(number_of_layers, device=device)

		ffn_up_row_norm_power = torch.zeros(number_of_layers, max(ffn_hidden_size), device=device)
		ffn_down_row_norm_power = torch.zeros(number_of_layers, embedding_size, device=device)

		ffn_up_row_ratio = torch.zeros(number_of_layers, max(ffn_hidden_size), device=device)
		ffn_down_row_ratio = torch.zeros(number_of_layers, embedding_size, device=device)

		ffn_up_row_history_ratio = torch.zeros(number_of_layers, max(ffn_hidden_size), device=device)
		ffn_down_row_history_ratio = torch.zeros(number_of_layers, embedding_size, device=device)

		ffn_up_column_norm_power = torch.zeros(number_of_layers, embedding_size, device=device)
		ffn_down_column_norm_power = torch.zeros(number_of_layers, max(ffn_hidden_size), device=device)

		ffn_up_column_ratio = torch.zeros(number_of_layers, embedding_size, device=device)
		ffn_down_column_ratio = torch.zeros(number_of_layers, max(ffn_hidden_size), device=device)

		ffn_up_column_history_ratio = torch.zeros(number_of_layers, embedding_size, device=device)
		ffn_down_column_history_ratio = torch.zeros(number_of_layers, max(ffn_hidden_size), device=device)

		for layer in range(number_of_layers):
			ffn_up_momentum_row_rms = ffn_up_linear_momentum_gradient[layer].square().mean(dim=1).sqrt()
			ffn_down_momentum_row_rms = ffn_down_linear_momentum_gradient[layer].square().mean(dim=1).sqrt()

			ffn_up_weight_row_rms = ffn_up_linear[layer].weight.square().mean(dim=1).sqrt()
			ffn_down_weight_row_rms = ffn_down_linear[layer].weight.square().mean(dim=1).sqrt()

			ffn_up_row_log_ratio = torch.log(ffn_up_weight_row_rms / (ffn_up_momentum_row_rms + epsilon))
			ffn_down_row_log_ratio = torch.log(ffn_down_weight_row_rms / (ffn_down_momentum_row_rms + epsilon))

			ffn_up_row_log_average = ffn_up_row_log_ratio.mean()
			ffn_down_row_log_average = ffn_down_row_log_ratio.mean()

			ffn_up_row_std = ffn_up_row_log_ratio.std()
			ffn_down_row_std = ffn_down_row_log_ratio.std()

			ffn_up_row_zscore = (ffn_up_row_log_ratio - ffn_up_row_log_average) / (ffn_up_row_std + epsilon)
			ffn_down_row_zscore = (ffn_down_row_log_ratio - ffn_down_row_log_average) / (ffn_down_row_std + epsilon)

			ffn_up_row_norm_power[layer, :ffn_hidden_size[layer]] = pmax_row * (1 - 2 ** (-ffn_up_row_zscore.abs()))
			ffn_down_row_norm_power[layer] = pmax_row * (1 - 2 ** (-ffn_down_row_zscore.abs()))

			ffn_up_row_ratio[layer, :ffn_hidden_size[layer]] = torch.exp(ffn_up_row_log_ratio - ffn_up_row_log_average)
			ffn_down_row_ratio[layer] = torch.exp(ffn_down_row_log_ratio - ffn_down_row_log_average)

			ffn_up_row_history_ratio[layer, :ffn_hidden_size[layer]] = (ffn_up_reference_momentum_rms[layer, :ffn_hidden_size[layer]] / (ffn_up_momentum_row_rms + epsilon)) ** global_norm_power
			ffn_down_row_history_ratio[layer] = (ffn_down_reference_momentum_rms[layer] / (ffn_down_momentum_row_rms + epsilon)) ** global_norm_power

			#$ Column
			ffn_up_momentum_column_rms = ffn_up_linear_momentum_gradient[layer].square().mean(dim=0).sqrt()
			ffn_down_momentum_column_rms = ffn_down_linear_momentum_gradient[layer].square().mean(dim=0).sqrt()

			ffn_up_weight_column_rms = ffn_up_linear[layer].weight.square().mean(dim=0).sqrt()
			ffn_down_weight_column_rms = ffn_down_linear[layer].weight.square().mean(dim=0).sqrt()

			ffn_up_column_log_ratio = torch.log(ffn_up_weight_column_rms / (ffn_up_momentum_column_rms + epsilon))
			ffn_down_column_log_ratio = torch.log(ffn_down_weight_column_rms / (ffn_down_momentum_column_rms + epsilon))

			ffn_up_column_log_average = ffn_up_column_log_ratio.mean()
			ffn_down_column_log_average = ffn_down_column_log_ratio.mean()

			ffn_up_column_std = ffn_up_column_log_ratio.std()
			ffn_down_column_std = ffn_down_column_log_ratio.std()

			ffn_up_column_zscore = (ffn_up_column_log_ratio - ffn_up_column_log_average) / (ffn_up_column_std + epsilon)
			ffn_down_column_zscore = (ffn_down_column_log_ratio - ffn_down_column_log_average) / (ffn_down_column_std + epsilon)

			ffn_up_column_norm_power[layer] = pmax_column * (1 - 2 ** (-ffn_up_column_zscore.abs()))
			ffn_down_column_norm_power[layer, :ffn_hidden_size[layer]] = pmax_column * (1 - 2 ** (-ffn_down_column_zscore.abs()))

			ffn_up_column_ratio[layer] = torch.exp(ffn_up_column_log_ratio - ffn_up_column_log_average)
			ffn_down_column_ratio[layer, :ffn_hidden_size[layer]] = torch.exp(ffn_down_column_log_ratio - ffn_down_column_log_average)

			ffn_up_column_history_ratio[layer] = (ffn_up_reference_momentum_column_rms[layer] / (ffn_up_momentum_column_rms + epsilon)) ** global_norm_power
			ffn_down_column_history_ratio[layer, :ffn_hidden_size[layer]] = (ffn_down_reference_momentum_column_rms[layer, :ffn_hidden_size[layer]] / (ffn_down_momentum_column_rms + epsilon)) ** global_norm_power

			#$ Layer
			ffn_up_layer_momentum_rms[layer] = ffn_up_momentum_row_rms.square().mean().sqrt()
			ffn_down_layer_momentum_rms[layer] = ffn_down_momentum_row_rms.square().mean().sqrt()

			ffn_up_layer_weight_rms = ffn_up_weight_row_rms.square().mean().sqrt()
			ffn_down_layer_weight_rms = ffn_down_weight_row_rms.square().mean().sqrt()

			ffn_up_layer_log_ratio[layer] = torch.log(ffn_up_layer_weight_rms / (ffn_up_layer_momentum_rms[layer] + epsilon))
			ffn_down_layer_log_ratio[layer] = torch.log(ffn_down_layer_weight_rms / (ffn_down_layer_momentum_rms[layer] + epsilon))

		ffn_up_layer_log_average = ffn_up_layer_log_ratio.mean()
		ffn_up_layer_std = ffn_up_layer_log_ratio.std()
		ffn_up_layer_zscore = (ffn_up_layer_log_ratio - ffn_up_layer_log_average) / (ffn_up_layer_std + epsilon)
		ffn_up_layer_norm_power = pmax_layer * (1 - 2 ** (-ffn_up_layer_zscore.abs()))
		ffn_up_layer_ratio = torch.exp(ffn_up_layer_log_ratio - ffn_up_layer_log_average)

		ffn_down_layer_log_average = ffn_down_layer_log_ratio.mean()
		ffn_down_layer_std = ffn_down_layer_log_ratio.std()
		ffn_down_layer_zscore = (ffn_down_layer_log_ratio - ffn_down_layer_log_average) / (ffn_down_layer_std + epsilon)
		ffn_down_layer_norm_power = pmax_layer * (1 - 2 ** (-ffn_down_layer_zscore.abs()))
		ffn_down_layer_ratio = torch.exp(ffn_down_layer_log_ratio - ffn_down_layer_log_average)


		#& Wo layers — apply row, column, and layer normalization like QKV
		Wo_layer_log_ratio = torch.zeros(number_of_layers, device=device)
		Wo_layer_momentum_rms = torch.zeros(number_of_layers, device=device)
		Wo_row_norm_power = torch.zeros(number_of_layers, embedding_size, device=device)
		Wo_row_ratio = torch.zeros(number_of_layers, embedding_size, device=device)
		Wo_row_history_ratio = torch.zeros(number_of_layers, embedding_size, device=device)
		Wo_column_norm_power = torch.zeros(number_of_layers, embedding_size, device=device)
		Wo_column_ratio = torch.zeros(number_of_layers, embedding_size, device=device)
		Wo_column_history_ratio = torch.zeros(number_of_layers, embedding_size, device=device)

		for layer in range(number_of_layers):
			Wo_momentum_row_rms = Wo_linear_momentum_gradient[layer].square().mean(dim=1).sqrt()
			Wo_weight_row_rms = Wo_linear[layer].weight.square().mean(dim=1).sqrt()

			Wo_row_log_ratio = torch.log(Wo_weight_row_rms / (Wo_momentum_row_rms + epsilon))
			Wo_row_log_average = Wo_row_log_ratio.mean()
			Wo_row_std = Wo_row_log_ratio.std()
			Wo_row_zscore = (Wo_row_log_ratio - Wo_row_log_average) / (Wo_row_std + epsilon)
			Wo_row_norm_power[layer] = pmax_row * (1 - 2 ** (-Wo_row_zscore.abs()))
			Wo_row_ratio[layer] = torch.exp(Wo_row_log_ratio - Wo_row_log_average)
			Wo_row_history_ratio[layer] = (Wo_reference_momentum_rms[layer] / (Wo_momentum_row_rms + epsilon)) ** global_norm_power

			#$ Column
			Wo_momentum_column_rms = Wo_linear_momentum_gradient[layer].square().mean(dim=0).sqrt()
			Wo_weight_column_rms = Wo_linear[layer].weight.square().mean(dim=0).sqrt()

			Wo_column_log_ratio = torch.log(Wo_weight_column_rms / (Wo_momentum_column_rms + epsilon))
			Wo_column_log_average = Wo_column_log_ratio.mean()
			Wo_column_std = Wo_column_log_ratio.std()
			Wo_column_zscore = (Wo_column_log_ratio - Wo_column_log_average) / (Wo_column_std + epsilon)
			Wo_column_norm_power[layer] = pmax_column * (1 - 2 ** (-Wo_column_zscore.abs()))
			Wo_column_ratio[layer] = torch.exp(Wo_column_log_ratio - Wo_column_log_average)
			Wo_column_history_ratio[layer] = (Wo_reference_momentum_column_rms[layer] / (Wo_momentum_column_rms + epsilon)) ** global_norm_power

			#$ Layer
			Wo_layer_momentum_rms[layer] = Wo_momentum_row_rms.square().mean().sqrt()
			Wo_layer_weight_rms = Wo_weight_row_rms.square().mean().sqrt()
			Wo_layer_log_ratio[layer] = torch.log(Wo_layer_weight_rms / (Wo_layer_momentum_rms[layer] + epsilon))

		Wo_layer_log_average = Wo_layer_log_ratio.mean()
		Wo_layer_std = Wo_layer_log_ratio.std()
		Wo_layer_zscore = (Wo_layer_log_ratio - Wo_layer_log_average) / (Wo_layer_std + epsilon)
		Wo_layer_norm_power = pmax_layer * (1 - 2 ** (-Wo_layer_zscore.abs()))
		Wo_layer_ratio = torch.exp(Wo_layer_log_ratio - Wo_layer_log_average)


		return (
			query_row_norm_power, key_row_norm_power, value_row_norm_power,
			query_row_ratio, key_row_ratio, value_row_ratio,

			query_column_norm_power, key_column_norm_power, value_column_norm_power,
			query_column_ratio, key_column_ratio, value_column_ratio,

			query_row_history_ratio, key_row_history_ratio, value_row_history_ratio,
			query_column_history_ratio, key_column_history_ratio, value_column_history_ratio,

			query_layer_zscore, key_layer_zscore, value_layer_zscore,
			query_layer_norm_power, key_layer_norm_power, value_layer_norm_power,
			query_layer_ratio, key_layer_ratio, value_layer_ratio,

			ffn_up_row_norm_power, ffn_down_row_norm_power,
			ffn_up_row_ratio, ffn_down_row_ratio,
			ffn_up_row_history_ratio, ffn_down_row_history_ratio,

			ffn_up_column_norm_power, ffn_down_column_norm_power,
			ffn_up_column_ratio, ffn_down_column_ratio,
			ffn_up_column_history_ratio, ffn_down_column_history_ratio,

			ffn_up_layer_zscore, ffn_down_layer_zscore,
			ffn_up_layer_norm_power, ffn_down_layer_norm_power,
			ffn_up_layer_ratio, ffn_down_layer_ratio,

			Wo_row_norm_power, Wo_row_ratio, Wo_row_history_ratio,
			Wo_column_norm_power, Wo_column_ratio, Wo_column_history_ratio,

			Wo_layer_zscore, Wo_layer_norm_power, Wo_layer_ratio,
		)

def update_weights(debug_step, optimizer_stats):
	with torch.no_grad():
		normalized_learning_rate = learning_rate_tensor #** (1 - norm_power)

		(
			query_row_norm_power, key_row_norm_power, value_row_norm_power,
			query_row_ratio, key_row_ratio, value_row_ratio,

			query_column_norm_power, key_column_norm_power, value_column_norm_power,
			query_column_ratio, key_column_ratio, value_column_ratio,

			query_row_history_ratio, key_row_history_ratio, value_row_history_ratio,
			query_column_history_ratio, key_column_history_ratio, value_column_history_ratio,

			query_layer_zscore, key_layer_zscore, value_layer_zscore,
			query_layer_norm_power, key_layer_norm_power, value_layer_norm_power,
			query_layer_ratio, key_layer_ratio, value_layer_ratio,

			ffn_up_row_norm_power, ffn_down_row_norm_power,
			ffn_up_row_ratio, ffn_down_row_ratio,
			ffn_up_row_history_ratio, ffn_down_row_history_ratio,

			ffn_up_column_norm_power, ffn_down_column_norm_power,
			ffn_up_column_ratio, ffn_down_column_ratio,
			ffn_up_column_history_ratio, ffn_down_column_history_ratio,

			ffn_up_layer_zscore, ffn_down_layer_zscore,
			ffn_up_layer_norm_power, ffn_down_layer_norm_power,
			ffn_up_layer_ratio, ffn_down_layer_ratio,

			Wo_row_norm_power, Wo_row_ratio, Wo_row_history_ratio,
			Wo_column_norm_power, Wo_column_ratio, Wo_column_history_ratio,

			Wo_layer_zscore, Wo_layer_norm_power, Wo_layer_ratio,
		) = optimizer_stats

		#& Token embedding table
		token_embedding_table.sub_(token_embedding_table_momentum_gradient * normalized_learning_rate)

		#& QKV layers
		for layer in range(number_of_layers):
			qkv_linear[layer].weight[0:embedding_size].sub_(
				(query_row_ratio[layer] ** query_row_norm_power[layer]).unsqueeze(1)    #Row peer to peer
				* (query_column_ratio[layer] ** query_column_norm_power[layer]).unsqueeze(0)    #Column peer to peer
				* (query_row_history_ratio[layer].unsqueeze(1) * query_column_history_ratio[layer].unsqueeze(0)).sqrt()    #Row and column momentum history
				* query_layer_ratio[layer] ** query_layer_norm_power[layer]    #Layer peer to peer
				* qkv_linear_momentum_gradient[layer][0:embedding_size]    #Momentum gradient
				* normalized_learning_rate    #Learning rate
			)
			qkv_linear[layer].weight[embedding_size:embedding_size * 2].sub_(
				(key_row_ratio[layer] ** key_row_norm_power[layer]).unsqueeze(1)    #Row peer to peer
				* (key_column_ratio[layer] ** key_column_norm_power[layer]).unsqueeze(0)    #Column peer to peer
				* (key_row_history_ratio[layer].unsqueeze(1) * key_column_history_ratio[layer].unsqueeze(0)).sqrt()    #Row and column momentum history
				* key_layer_ratio[layer] ** key_layer_norm_power[layer]    #Layer peer to peer
				* qkv_linear_momentum_gradient[layer][embedding_size:embedding_size * 2]    #Momentum gradient
				* normalized_learning_rate    #Learning rate
			)
			qkv_linear[layer].weight[embedding_size * 2:embedding_size * 3].sub_(
				(value_row_ratio[layer] ** value_row_norm_power[layer]).unsqueeze(1)    #Row peer to peer
				* (value_column_ratio[layer] ** value_column_norm_power[layer]).unsqueeze(0)    #Column peer to peer
				* (value_row_history_ratio[layer].unsqueeze(1) * value_column_history_ratio[layer].unsqueeze(0)).sqrt()    #Row and column momentum history
				* value_layer_ratio[layer] ** value_layer_norm_power[layer]    #Layer peer to peer
				* qkv_linear_momentum_gradient[layer][embedding_size * 2:embedding_size * 3]    #Momentum gradient
				* normalized_learning_rate    #Learning rate
			)

			if debug_step:
				query_history_ratio = (query_row_history_ratio[layer].unsqueeze(1) * query_column_history_ratio[layer].unsqueeze(0)).sqrt()
				key_history_ratio = (key_row_history_ratio[layer].unsqueeze(1) * key_column_history_ratio[layer].unsqueeze(0)).sqrt()
				value_history_ratio = (value_row_history_ratio[layer].unsqueeze(1) * value_column_history_ratio[layer].unsqueeze(0)).sqrt()

				query_peer_multiplier = (query_row_ratio[layer] ** query_row_norm_power[layer]).unsqueeze(1) * (query_column_ratio[layer] ** query_column_norm_power[layer]).unsqueeze(0)
				key_peer_multiplier = (key_row_ratio[layer] ** key_row_norm_power[layer]).unsqueeze(1) * (key_column_ratio[layer] ** key_column_norm_power[layer]).unsqueeze(0)
				value_peer_multiplier = (value_row_ratio[layer] ** value_row_norm_power[layer]).unsqueeze(1) * (value_column_ratio[layer] ** value_column_norm_power[layer]).unsqueeze(0)

				query_zscore_table[layer] = query_layer_zscore[layer]
				key_zscore_table[layer] = key_layer_zscore[layer]
				value_zscore_table[layer] = value_layer_zscore[layer]

				query_norm_power_table[layer] = query_layer_norm_power[layer]
				key_norm_power_table[layer] = key_layer_norm_power[layer]
				value_norm_power_table[layer] = value_layer_norm_power[layer]

				query_ratio_table[layer] = query_layer_ratio[layer]
				key_ratio_table[layer] = key_layer_ratio[layer]
				value_ratio_table[layer] = value_layer_ratio[layer]

				query_momentum_multiplier[layer] = query_history_ratio.mean()
				key_momentum_multiplier[layer] = key_history_ratio.mean()
				value_momentum_multiplier[layer] = value_history_ratio.mean()

				query_multiplier[layer] = (query_peer_multiplier * query_history_ratio).mean() * query_layer_ratio[layer] ** query_layer_norm_power[layer]
				key_multiplier[layer] = (key_peer_multiplier * key_history_ratio).mean() * key_layer_ratio[layer] ** key_layer_norm_power[layer]
				value_multiplier[layer] = (value_peer_multiplier * value_history_ratio).mean() * value_layer_ratio[layer] ** value_layer_norm_power[layer]

		#& Output table
		output_weight_table.sub_(output_weight_table_momentum_gradient * normalized_learning_rate)

		for layer in range(number_of_layers):
			ffn_up_history_ratio = (ffn_up_row_history_ratio[layer, :ffn_hidden_size[layer]].unsqueeze(1) * ffn_up_column_history_ratio[layer].unsqueeze(0)).sqrt()
			ffn_down_history_ratio = (ffn_down_row_history_ratio[layer].unsqueeze(1) * ffn_down_column_history_ratio[layer, :ffn_hidden_size[layer]].unsqueeze(0)).sqrt()

			ffn_up_linear[layer].weight.sub_(
				(ffn_up_row_ratio[layer, :ffn_hidden_size[layer]] ** ffn_up_row_norm_power[layer, :ffn_hidden_size[layer]]).unsqueeze(1)    #Row peer to peer
				* (ffn_up_column_ratio[layer] ** ffn_up_column_norm_power[layer]).unsqueeze(0)    #Column peer to peer
				* ffn_up_history_ratio    #Row and column momentum history
				* ffn_up_layer_ratio[layer] ** ffn_up_layer_norm_power[layer]    #Layer peer to peer
				* ffn_up_linear_momentum_gradient[layer]    #Momentum gradient
				* normalized_learning_rate    #Learning rate
			)

			ffn_down_linear[layer].weight.sub_(
				(ffn_down_row_ratio[layer] ** ffn_down_row_norm_power[layer]).unsqueeze(1)    #Row peer to peer
				* (ffn_down_column_ratio[layer, :ffn_hidden_size[layer]] ** ffn_down_column_norm_power[layer, :ffn_hidden_size[layer]]).unsqueeze(0)    #Column peer to peer
				* ffn_down_history_ratio    #Row and column momentum history
				* ffn_down_layer_ratio[layer] ** ffn_down_layer_norm_power[layer]    #Layer peer to peer
				* ffn_down_linear_momentum_gradient[layer]    #Momentum gradient
				* normalized_learning_rate    #Learning rate
			)

			if debug_step:
				ffn_up_peer_multiplier = (ffn_up_row_ratio[layer, :ffn_hidden_size[layer]] ** ffn_up_row_norm_power[layer, :ffn_hidden_size[layer]]).unsqueeze(1) * (ffn_up_column_ratio[layer] ** ffn_up_column_norm_power[layer]).unsqueeze(0)
				ffn_down_peer_multiplier = (ffn_down_row_ratio[layer] ** ffn_down_row_norm_power[layer]).unsqueeze(1) * (ffn_down_column_ratio[layer, :ffn_hidden_size[layer]] ** ffn_down_column_norm_power[layer, :ffn_hidden_size[layer]]).unsqueeze(0)

				ffn_up_zscore[layer] = ffn_up_layer_zscore[layer]
				ffn_up_norm_power[layer] = ffn_up_layer_norm_power[layer]
				ffn_up_ratio[layer] = ffn_up_layer_ratio[layer]
				ffn_up_momentum_multiplier[layer] = ffn_up_history_ratio.mean()
				ffn_up_multiplier[layer] = (ffn_up_peer_multiplier * ffn_up_history_ratio).mean() * ffn_up_layer_ratio[layer] ** ffn_up_layer_norm_power[layer]

				ffn_down_zscore[layer] = ffn_down_layer_zscore[layer]
				ffn_down_norm_power[layer] = ffn_down_layer_norm_power[layer]
				ffn_down_ratio[layer] = ffn_down_layer_ratio[layer]
				ffn_down_momentum_multiplier[layer] = ffn_down_history_ratio.mean()
				ffn_down_multiplier[layer] = (ffn_down_peer_multiplier * ffn_down_history_ratio).mean() * ffn_down_layer_ratio[layer] ** ffn_down_layer_norm_power[layer]

		for layer in range(number_of_layers):
			Wo_history_ratio = (Wo_row_history_ratio[layer].unsqueeze(1) * Wo_column_history_ratio[layer].unsqueeze(0)).sqrt()

			Wo_linear[layer].weight.sub_(
				(Wo_row_ratio[layer] ** Wo_row_norm_power[layer]).unsqueeze(1)    #Row peer to peer
				* (Wo_column_ratio[layer] ** Wo_column_norm_power[layer]).unsqueeze(0)    #Column peer to peer
				* Wo_history_ratio    #Row and column momentum history
				* Wo_layer_ratio[layer] ** Wo_layer_norm_power[layer]    #Layer peer to peer
				* Wo_linear_momentum_gradient[layer]    #Momentum gradient
				* normalized_learning_rate    #Learning rate
			)

			if debug_step:
				Wo_peer_multiplier = (Wo_row_ratio[layer] ** Wo_row_norm_power[layer]).unsqueeze(1) * (Wo_column_ratio[layer] ** Wo_column_norm_power[layer]).unsqueeze(0)
				Wo_zscore[layer] = Wo_layer_zscore[layer]
				Wo_norm_power[layer] = Wo_layer_norm_power[layer]
				Wo_ratio[layer] = Wo_layer_ratio[layer]
				Wo_momentum_multiplier[layer] = Wo_history_ratio.mean()
				Wo_multiplier[layer] = (Wo_peer_multiplier * Wo_history_ratio).mean() * Wo_layer_ratio[layer] ** Wo_layer_norm_power[layer]

		#& Gamma tables
		attention_gamma_table.sub_(attention_gamma_table_momentum_gradient * normalized_learning_rate)

		ffn_gamma_table.sub_(ffn_gamma_table_momentum_gradient * normalized_learning_rate)

		token_embedding_table.grad = None
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
max_compile = True
compiled_training_step = torch.compile(
	training_step,
	# max-autotune for long uses. default mode for testing
	mode="max-autotune" if max_compile else "default",
	# mode="default",
	dynamic=False,
)

compiled_update_momentum = torch.compile(
	update_momentum,
	# This function intentionally mutates persistent momentum buffers.
	mode="max-autotune-no-cudagraphs" if max_compile else "default",
	dynamic=False,
)

compiled_optimizer_stats = torch.compile(
	calculate_optimizer_stats,
	# max-autotune for long uses. default mode for testing
	mode="max-autotune" if max_compile else "default",
	# mode="default",
	dynamic=False,
)

compiled_update_weights = torch.compile(
	update_weights,
	# max-autotune-no-cudagraphs for long uses. default mode for testing
	mode="max-autotune-no-cudagraphs" if max_compile else "default",
	# mode="default",
	dynamic=False
)


#! Settings                                                 
embedding_size = 768
CONTEXT_LENGTH = 384
batch_size = 8
gradient_scale = 256
head_count = 6
ffn_hidden_multiplier = [4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2]
momentum_rate = 0.9
pmax_layer = 0.85        # relative layer correction
pmax_row = 0.55       # relative row correction
pmax_column = 0.55        # relative column correction
global_norm_power = 0.45      # whole-family magnitude correction
frequency_base = 10000
calibration_steps = 600
calibration_burn_in_steps = 100
number_of_training_windows = len(training_token_ids) - CONTEXT_LENGTH
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

	head_size = embedding_size // head_count
	head_frequency = torch.tensor([1 / (frequency_base ** (i * 2 / head_size)) for i in range(head_size // 2)], device=device, dtype=torch.float32)
	rotation_angle = torch.arange(CONTEXT_LENGTH, device=device, dtype=torch.float32).unsqueeze(-1) * head_frequency.unsqueeze(0)

	cosine_table = torch.cos(rotation_angle).unsqueeze(0).unsqueeze(0)
	sin_table = torch.sin(rotation_angle).unsqueeze(0).unsqueeze(0)

	#$ Momentum Gradients
	token_embedding_table_momentum_gradient = torch.zeros(number_of_ids, embedding_size, device=device)

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
	query_multiplier = torch.ones(number_of_layers, device=device)
	key_multiplier = torch.ones(number_of_layers, device=device)
	value_multiplier = torch.ones(number_of_layers, device=device)
	query_zscore_table = torch.zeros(number_of_layers, device=device)
	key_zscore_table = torch.zeros(number_of_layers, device=device)
	value_zscore_table = torch.zeros(number_of_layers, device=device)
	query_norm_power_table = torch.zeros(number_of_layers, device=device)
	key_norm_power_table = torch.zeros(number_of_layers, device=device)
	value_norm_power_table = torch.zeros(number_of_layers, device=device)
	query_ratio_table = torch.ones(number_of_layers, device=device)
	key_ratio_table = torch.ones(number_of_layers, device=device)
	value_ratio_table = torch.ones(number_of_layers, device=device)
	query_momentum_multiplier = torch.ones(number_of_layers, device=device)
	key_momentum_multiplier = torch.ones(number_of_layers, device=device)
	value_momentum_multiplier = torch.ones(number_of_layers, device=device)
    
	# Per-layer momentum-only multipliers for debugging
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
	query_reference_momentum_rms = torch.zeros(number_of_layers, embedding_size, device=device)
	key_reference_momentum_rms = torch.zeros(number_of_layers, embedding_size, device=device)
	value_reference_momentum_rms = torch.zeros(number_of_layers, embedding_size, device=device)
	query_reference_momentum_column_rms = torch.zeros(number_of_layers, embedding_size, device=device)
	key_reference_momentum_column_rms = torch.zeros(number_of_layers, embedding_size, device=device)
	value_reference_momentum_column_rms = torch.zeros(number_of_layers, embedding_size, device=device)
	ffn_up_initial_weight_rms = torch.zeros(number_of_layers, device=device)
	ffn_up_reference_momentum_rms = torch.zeros(number_of_layers, max(ffn_hidden_size), device=device)
	ffn_up_reference_momentum_column_rms = torch.zeros(number_of_layers, embedding_size, device=device)
	ffn_down_initial_weight_rms = torch.zeros(number_of_layers, device=device)
	ffn_down_reference_momentum_rms = torch.zeros(number_of_layers, embedding_size, device=device)
	ffn_down_reference_momentum_column_rms = torch.zeros(number_of_layers, max(ffn_hidden_size), device=device)
	Wo_initial_weight_rms = torch.zeros(number_of_layers, device=device)
	Wo_reference_momentum_rms = torch.zeros(number_of_layers, embedding_size, device=device)
	Wo_reference_momentum_column_rms = torch.zeros(number_of_layers, embedding_size, device=device)


iteration = 0
loss_window_average = 0.0
window_loss_size = 1000
checkpoint_interval = 60000
max_training_hours = 6
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
			compiled_training_step(start_positions, training_token_ids)

		with torch.no_grad():
			token_embedding_table_momentum_gradient.mul_(momentum_rate).add_(token_embedding_table.grad, alpha=1 - momentum_rate)
			output_weight_table_momentum_gradient.mul_(momentum_rate).add_(output_weight_table.grad, alpha=1 - momentum_rate)
			attention_gamma_table_momentum_gradient.mul_(momentum_rate).add_(attention_gamma_table.grad, alpha=1 - momentum_rate)
			ffn_gamma_table_momentum_gradient.mul_(momentum_rate).add_(ffn_gamma_table.grad, alpha=1 - momentum_rate)

			for layer in range(number_of_layers):
				qkv_linear_momentum_gradient[layer].mul_(momentum_rate).add_(qkv_linear[layer].weight.grad, alpha=1 - momentum_rate)
				ffn_up_linear_momentum_gradient[layer].mul_(momentum_rate).add_(ffn_up_linear[layer].weight.grad, alpha=1 - momentum_rate)
				ffn_down_linear_momentum_gradient[layer].mul_(momentum_rate).add_(ffn_down_linear[layer].weight.grad, alpha=1 - momentum_rate)
				Wo_linear_momentum_gradient[layer].mul_(momentum_rate).add_(Wo_linear[layer].weight.grad, alpha=1 - momentum_rate)

			if calibration_step >= calibration_burn_in_steps:
				query_reference_momentum_rms.add_(qkv_linear_momentum_gradient[:, :embedding_size].square().mean(dim=2).sqrt())
				key_reference_momentum_rms.add_(qkv_linear_momentum_gradient[:, embedding_size:embedding_size * 2].square().mean(dim=2).sqrt())
				value_reference_momentum_rms.add_(qkv_linear_momentum_gradient[:, embedding_size * 2:].square().mean(dim=2).sqrt())
				query_reference_momentum_column_rms.add_(qkv_linear_momentum_gradient[:, :embedding_size].square().mean(dim=1).sqrt())
				key_reference_momentum_column_rms.add_(qkv_linear_momentum_gradient[:, embedding_size:embedding_size * 2].square().mean(dim=1).sqrt())
				value_reference_momentum_column_rms.add_(qkv_linear_momentum_gradient[:, embedding_size * 2:].square().mean(dim=1).sqrt())

				for layer in range(number_of_layers):
					ffn_up_reference_momentum_rms[layer, :ffn_hidden_size[layer]].add_(ffn_up_linear_momentum_gradient[layer].square().mean(dim=1).sqrt())
					ffn_up_reference_momentum_column_rms[layer].add_(ffn_up_linear_momentum_gradient[layer].square().mean(dim=0).sqrt())
					ffn_down_reference_momentum_rms[layer].add_(ffn_down_linear_momentum_gradient[layer].square().mean(dim=1).sqrt())
					ffn_down_reference_momentum_column_rms[layer, :ffn_hidden_size[layer]].add_(ffn_down_linear_momentum_gradient[layer].square().mean(dim=0).sqrt())
					Wo_reference_momentum_rms[layer].add_(Wo_linear_momentum_gradient[layer].square().mean(dim=1).sqrt())
					Wo_reference_momentum_column_rms[layer].add_(Wo_linear_momentum_gradient[layer].square().mean(dim=0).sqrt())
				measurements += 1

			token_embedding_table.grad = output_weight_table.grad = None
			attention_gamma_table.grad = ffn_gamma_table.grad = None
			for layer in range(number_of_layers):
				qkv_linear[layer].weight.grad = ffn_up_linear[layer].weight.grad = None
				ffn_down_linear[layer].weight.grad = Wo_linear[layer].weight.grad = None

	query_reference_momentum_rms.div_(measurements)
	key_reference_momentum_rms.div_(measurements)
	value_reference_momentum_rms.div_(measurements)
	query_reference_momentum_column_rms.div_(measurements)
	key_reference_momentum_column_rms.div_(measurements)
	value_reference_momentum_column_rms.div_(measurements)
	ffn_up_reference_momentum_rms.div_(measurements)
	ffn_up_reference_momentum_column_rms.div_(measurements)
	ffn_down_reference_momentum_rms.div_(measurements)
	ffn_down_reference_momentum_column_rms.div_(measurements)
	Wo_reference_momentum_rms.div_(measurements)
	Wo_reference_momentum_column_rms.div_(measurements)

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
		window_loss = compiled_training_step(start_positions, training_token_ids)

	debug_step = iteration % (window_loss_size * 7) == 0
	compiled_update_momentum()
	optimizer_stats = compiled_optimizer_stats()
	compiled_update_weights(debug_step, optimizer_stats)

	if debug_step:
		with torch.no_grad():
			print("\n--- Multipliers ---")
			for layer in range(number_of_layers):
				for name, z_table, p_table, ratio_table, multiplier_table, momentum_table in (
					("Q", query_zscore_table, query_norm_power_table, query_ratio_table, query_multiplier, query_momentum_multiplier),
					("K", key_zscore_table, key_norm_power_table, key_ratio_table, key_multiplier, key_momentum_multiplier),
					("V", value_zscore_table, value_norm_power_table, value_ratio_table, value_multiplier, value_momentum_multiplier),
				):
					z = z_table[layer].item()
					p = p_table[layer].item()
					ratio = ratio_table[layer].item()
					mult = multiplier_table[layer].item()
					mom = momentum_table[layer].item()
					p_str = f"{p:.3f}".lstrip("0")
					print(f"{name}   L{layer} | z={z:+.2f} | p={p_str} | ratio={ratio:.2f} | multiplier={mult:.3f} | mom={mom:.3f}")

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
			f"charspeed: {iterations_per_second * batch_size * CONTEXT_LENGTH:,.0f} "
			f"chars/sec, "
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