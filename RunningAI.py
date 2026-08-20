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
import random
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

def encode_text(text):
	token_ids = [character_to_id[char] if char in character_to_id else character_to_id[UNKNOWN_TOKEN] for char in text]
	return token_ids

def decode_token_ids(token_ids):
	return "".join([id_to_character[token_id] if token_id != character_to_id[UNKNOWN_TOKEN] else UNKNOWN_TOKEN for token_id in token_ids])

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"


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

	iteration = checkpoint.get("iteration", 0)
	best_loss = checkpoint.get("best_loss", float("inf"))
	persistent_learning_rate = checkpoint.get("persistent_learning_rate", 1)
	previous_elapsed_seconds = checkpoint.get("previous_elapsed_seconds", 0)

	print(f"Model loaded from: {path}")
	print(f"Resuming from iteration: {iteration}")
	print(f"Previous best loss: {best_loss:.6f}")

	return iteration, best_loss, persistent_learning_rate, previous_elapsed_seconds

def get_input_token_ids(start_positions, source_token_ids, L):
	offsets = torch.arange(L, device=device, dtype=torch.long)[None, :]
	return source_token_ids[start_positions + offsets]

def run_forward_attention(layer, input_embeddings, normalized_input_embeddings, L):
	#% Shape: 3 x context_length x embedding_size
	qkv_matrix = qkv_linear[layer](normalized_input_embeddings)

	query_matrix = qkv_matrix[...,0:embedding_size,]
	key_matrix = qkv_matrix[...,embedding_size:2 * embedding_size,]
	value_matrix = qkv_matrix[...,2 * embedding_size:3 * embedding_size,]

	query_matrix = query_matrix.reshape(batch_size, L, head_count, embedding_size // head_count // 2, 2).transpose(1, 2)
	key_matrix = key_matrix.reshape(batch_size, L, head_count, embedding_size // head_count // 2, 2).transpose(1, 2)
	value_matrix = value_matrix.reshape(batch_size, L, head_count, embedding_size // head_count).transpose(1, 2)

	query_first, query_second = torch.unbind(query_matrix, dim=-1)
	key_first, key_second = torch.unbind(key_matrix, dim=-1)

	spliced_cosine_table = cosine_table[:, :, :L, :]
	spliced_sin_table = sin_table[:, :, :L, :]

	rotated_query_first = query_first * spliced_cosine_table - query_second * spliced_sin_table
	rotated_query_second = query_first * spliced_sin_table + query_second * spliced_cosine_table
	rotated_key_first = key_first * spliced_cosine_table - key_second * spliced_sin_table
	rotated_key_second = key_first * spliced_sin_table + key_second * spliced_cosine_table

	query_matrix = torch.stack((rotated_query_first, rotated_query_second), dim=-1).reshape(batch_size, head_count, L, embedding_size // head_count)
	key_matrix = torch.stack((rotated_key_first, rotated_key_second), dim=-1).reshape(batch_size, head_count, L, embedding_size // head_count)
	
	attention_output_matrix = F.scaled_dot_product_attention(
		query_matrix, key_matrix, value_matrix,
		dropout_p=0, is_causal=True
	).transpose(1, 2)
	attention_output_matrix = attention_output_matrix.reshape(batch_size, L, embedding_size)
	attention_output_matrix = Wo_linear[layer](attention_output_matrix) + input_embeddings

	return attention_output_matrix



def run_forward_pass(start_positions, source_token_ids, L):
	# Look up one embedding vector for each input token ID.
	input_token_ids = get_input_token_ids(start_positions, source_token_ids, L)
	
	# print(input_token_ids.shape)

	#% Shape: context_length × embedding_size
	#* This is converting each token (character) in the input of size CONTEXT_LENGTH (64) to their vector representation from the token_embedding_table
	input_token_embeddings = token_embedding_table[input_token_ids]

	#% Shape: context_length x embedding_size
	#* Adds the token vector and position vector at each position, producing the final input representation for the transformer.
	combined_input_embeddings = input_token_embeddings

	layer_state = combined_input_embeddings

	#@ Layers 1 - number_of_layers

	for layer in range(0, number_of_layers):
		layer_state_input = layer_state
		root_mean_square = torch.sqrt(layer_state_input.square().mean(dim = -1, keepdim = True) + epsilon)
		attention_inverse_root_mean_square = 1 / root_mean_square
		normalized_layer_state = layer_state_input * attention_inverse_root_mean_square
		scaled_layer_state = normalized_layer_state * attention_gamma_table[layer]
		attention_output_matrix = run_forward_attention(layer, layer_state, scaled_layer_state, L)

		ffn_input = attention_output_matrix
		root_mean_square = torch.sqrt(ffn_input.square().mean(dim = -1, keepdim = True) + epsilon)
		ffn_inverse_root_mean_square = 1 / root_mean_square
		normalized_ffn_input = ffn_input * ffn_inverse_root_mean_square
		scaled_ffn_input = normalized_ffn_input * ffn_gamma_table[layer]

		ffn_upscale_matrix = ffn_up_linear[layer](scaled_ffn_input)
		
		ffn_upscale_matrix = F.gelu(ffn_upscale_matrix, approximate="tanh")

		layer_state = ffn_down_linear[layer](ffn_upscale_matrix) + attention_output_matrix

	

	logits_matrix = layer_state @ output_weight_table.transpose(-2, -1)

	log_probability_matrix = F.log_softmax(logits_matrix.float() / temperature, dim=-1,)

	probability_matrix = log_probability_matrix.exp()

	return probability_matrix

def get_predicted_id_for_position(position, probability_matrix, top_k):
	probability_matrix = probability_matrix.reshape(-1, number_of_ids)

	top_vals, top_indices = torch.topk(probability_matrix[position], top_k, dim=-1)
	top_vals /= top_vals.sum()

	if (always_best):
		return top_indices[0].item()

	random_probability = random.random()
	cumulative_probability = 0

	for i in range(top_k):
		cumulative_probability += top_vals[i]
		if (random_probability < cumulative_probability):
			return top_indices[i].item()
	return top_indices[top_k - 1].item()
		
def print_top_probabilities(position, probability_matrix, top_k=10):
	probability_matrix = probability_matrix.reshape(-1, number_of_ids)
	sorted_probabilities, character_ids = torch.sort(probability_matrix[position], descending=True)
	print(f"\n--- Top {top_k} Predictions at Position {position} ---")
	print(f"{'Rank':<6} | {'Char':<8} | {'Probability':<12}")
	print("-" * 32)

	for i in range(top_k):
		prob = sorted_probabilities[i].item()
		char_id = character_ids[i].item()
		char_str = id_to_character.get(char_id, f"ID:{char_id}")

		# Clean up special characters for clear printing
		if char_str == " ":
			char_display = "' '"
		elif char_str == "\n":
			char_display = "'\\n'"
		else:
			char_display = f"'{char_str}'"

		print(f"{i+1:<6} | {char_display:<8} | {prob:.4f} ({prob * 100:.2f}%)")

#! Settings                                                 
embedding_size = 768
CONTEXT_LENGTH = 384
batch_size = 1
gradient_scale = 256
head_count = 6
ffn_hidden_multiplier = [4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2]
frequency_base = 10000
top_k = 5
temperature = 0.7
always_best = False
load_persistent_learning = True

number_of_layers = len(ffn_hidden_multiplier)

#& Embedding and weight table initialization + mask + arange indices
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

	#$ Moving to GPU
	future_mask = future_mask.to(device)

	batch_indices = torch.arange(batch_size, device=device, dtype=torch.long)[:, None]
	number_of_ids_indices = torch.arange(number_of_ids, device=device, dtype=torch.long)
	context_indices = torch.arange(CONTEXT_LENGTH, dtype=torch.long, device=device)[None, :]
	context_indices_plus1 = torch.arange(CONTEXT_LENGTH + 1, dtype=torch.long)[None, :]

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SAVED_BESTS_DIR = BASE_DIR / "Saved_bests"
checkpoint_path = SAVED_BESTS_DIR / "384x768_12lr_atNorm_6hd_web.pt"

if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
	iteration, current_least_loss, persistent_learning_rate, previous_elapsed_seconds = load_model(checkpoint_path)
	if(not load_persistent_learning):
		persistent_learning_rate = 1
else:
	print("No valid checkpoint found. Starting training from scratch.")

	iteration = 0
	current_least_loss = float("inf")
	previous_elapsed_seconds = 0
	persistent_learning_rate = 1


base_prompt = "France's capital is"
prompt_addition = " Paris"
generation_context = 384
for i in range(1):

	prompt = base_prompt + prompt_addition[:i]
	token_ids = encode_text(prompt)
	
	#! Generation loop          
	for iteration in range(1000):

		start_position = torch.tensor([max(len(token_ids) - generation_context, 0)], device=device)[None, :]
		L = min(len(token_ids), generation_context)
		with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
			probability_matrix = run_forward_pass(start_position, torch.tensor(token_ids, device=device), L)

		predicted_id = get_predicted_id_for_position(L - 1, probability_matrix, top_k)
		# print_top_probabilities(L - 1, probability_matrix, top_k)
		token_ids.append(predicted_id)
		prompt += id_to_character[predicted_id]

		
		window_start_time = time.perf_counter()
	print(decode_token_ids(token_ids))



