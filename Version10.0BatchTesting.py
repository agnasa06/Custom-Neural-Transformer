#V10: Added multi head attention without output projection matrix

import torch
import torch.nn.functional as F
import pandas as pd
import math
import os
import time
from pathlib import Path
import numpy as np


UNKNOWN_TOKEN = "<unk>"
mask_value = -2e4
device = torch.device("cuda")
torch.manual_seed(12354)
epsilon = 1e-5

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

		"token_embedding_table": token_embedding_table.detach().cpu(),
		"positional_embedding_table": (
			positional_embedding_table.detach().cpu()
		),
		"qkv_weight_table": qkv_weight_table.detach().cpu(),
		"output_weight_table": output_weight_table.detach().cpu(),
		"attention_gamma_table": attention_gamma_table.detach().cpu(),
		"ffn_gamma_table": ffn_gamma_table.detach().cpu(),
	}
	for layer in range(number_of_layers):
		checkpoint[f"ffn_up_weight_table_layer{layer+1}"] = ffn_up_weight_table[layer].detach().cpu()
		checkpoint[f"ffn_down_weight_table_layer{layer+1}"] = ffn_down_weight_table[layer].detach().cpu()

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
	global qkv_weight_table
	global output_weight_table
	global ffn_up_weight_table
	global ffn_down_weight_table
	global attention_gamma_table
	global ffn_gamma_table

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

	expected_qkv_shape = (number_of_layers, 3 * embedding_size,embedding_size,)

	checkpoint_qkv_shape = (
		checkpoint["qkv_weight_table"].shape if "qkv_weight_table" in checkpoint
		else (
			checkpoint["query_weight_table"].shape[0],
			3 * checkpoint["query_weight_table"].shape[1],
			checkpoint["query_weight_table"].shape[2],
		)
	)

	if checkpoint_qkv_shape != expected_qkv_shape:
		raise ValueError(
			"Cannot load checkpoint: QKV weight shape differs.\n"
			f"Checkpoint: {checkpoint_qkv_shape}\n"
			f"Current: {expected_qkv_shape}"
		)

	if checkpoint["ffn_hidden_size"] != ffn_hidden_size:
		raise ValueError(
			"Cannot load checkpoint: ffn_hidden_size_ size differs.\n"
			f"Checkpoint: {checkpoint['ffn_hidden_size']}\n"
			f"Current: {ffn_hidden_size}"
		)

	if "attention_gamma_table" in checkpoint:
		attention_gamma_table = checkpoint["attention_gamma_table"].to(device)

	if "ffn_gamma_table" in checkpoint:
		ffn_gamma_table = checkpoint["ffn_gamma_table"].to(device)

	if "qkv_weight_table" in checkpoint:
		qkv_weight_table = checkpoint[
			"qkv_weight_table"
		].to(device)

	else:
		qkv_weight_table = torch.cat(
			[
				checkpoint["query_weight_table"],
				checkpoint["key_weight_table"],
				checkpoint["value_weight_table"],
			],
			dim=1,
		).to(device)
	token_embedding_table = checkpoint[
		"token_embedding_table"
	].to(device)

	positional_embedding_table = checkpoint[
		"positional_embedding_table"
	].to(device)

	output_weight_table = checkpoint[
		"output_weight_table"
	].to(device)

	for layer in range(number_of_layers):
		ffn_up_weight_table[layer] = checkpoint[f"ffn_up_weight_table_layer{layer+1}"].to(device)
		ffn_down_weight_table[layer] = checkpoint[f"ffn_down_weight_table_layer{layer+1}"].to(device)

	refresh_fp16_matrices()

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

def refresh_fp16_matrices():
	global fp16_token_embedding_table
	global fp16_positional_embedding_table
	global fp16_qkv_weight_table
	global fp16_output_weight_table
	global fp16_ffn_up_weight_table
	global fp16_ffn_down_weight_table
	global fp16_attention_gamma_table
	global fp16_ffn_gamma_table
	fp16_token_embedding_table.copy_(token_embedding_table)
	fp16_positional_embedding_table.copy_(positional_embedding_table)
	fp16_qkv_weight_table.copy_(qkv_weight_table)
	fp16_output_weight_table.copy_(output_weight_table)
	fp16_attention_gamma_table.copy_(attention_gamma_table)
	fp16_ffn_gamma_table.copy_(ffn_gamma_table)
	for layer in range(number_of_layers):
		fp16_ffn_up_weight_table[layer].copy_(ffn_up_weight_table[layer])
		fp16_ffn_down_weight_table[layer].copy_(ffn_down_weight_table[layer])

def run_forward_attention(layer, input_embeddings, normalized_input_embeddings):
	#% Shape: 3 x context_length x embedding_size
	qkv_matrix = (normalized_input_embeddings @ fp16_qkv_weight_table[layer].transpose(-2, -1))

	query_matrix = qkv_matrix[...,0:embedding_size,]
	key_matrix = qkv_matrix[...,embedding_size:2 * embedding_size,]
	value_matrix = qkv_matrix[...,2 * embedding_size:3 * embedding_size,]

	query_matrix = query_matrix.reshape(batch_size, CONTEXT_LENGTH, head_count, embedding_size // head_count).transpose(1, 2)
	key_matrix = key_matrix.reshape(batch_size, CONTEXT_LENGTH, head_count, embedding_size // head_count).transpose(1, 2)
	value_matrix = value_matrix.reshape(batch_size, CONTEXT_LENGTH, head_count, embedding_size // head_count).transpose(1, 2)
	
	#% Shape: context_length x context_length
	attention_score_matrix = query_matrix @ key_matrix.transpose(-2, -1) / math.sqrt(embedding_size // head_count)
	attention_score_matrix = attention_score_matrix.masked_fill(future_mask, mask_value)
	attention_weight_matrix = torch.softmax(attention_score_matrix.float(), dim=-1).to(torch.float16)

	#% Shape: context_length x embedding_size
	attention_output_matrix = (attention_weight_matrix @ value_matrix).transpose(1, 2)
	attention_output_matrix = attention_output_matrix.reshape(batch_size, CONTEXT_LENGTH, embedding_size)
	attention_output_matrix += input_embeddings

	# 0: query_matrix, 1: key_matrix, 2: value_matrix
	# 3: attention_weight_matrix, 4: attention_output_matrix

	return [query_matrix, key_matrix, value_matrix, attention_weight_matrix, attention_output_matrix]


def run_forward_pass(start_positions, source_token_ids):
	# Look up one embedding vector for each input token ID.
	window_token_ids = get_window_token_ids(start_positions, source_token_ids).to(device)
	input_token_ids = window_token_ids[:, :-1]
	target_token_ids = window_token_ids[:, 1:]

	# print(input_token_ids.shape)

	#% Shape: context_length × embedding_size
	#* This is converting each token (character) in the input of size CONTEXT_LENGTH (64) to their vector representation from the token_embedding_table
	input_token_embeddings = fp16_token_embedding_table[input_token_ids]

	#% Shape: context_length x embedding_size
	#* Retrieves the position vector for each position in this input window. For now, this matches the full positional_embedding_table and is mainly useful for debugging.
	input_position_embeddings = fp16_positional_embedding_table

	#% Shape: context_length x embedding_size
	#* Adds the token vector and position vector at each position, producing the final input representation for the transformer.
	combined_input_embeddings = input_token_embeddings + input_position_embeddings

	query_matrix = [None] * number_of_layers
	key_matrix = [None] * number_of_layers
	value_matrix = [None] * number_of_layers
	attention_weight_matrix = [None] * number_of_layers
	attention_output_matrix = [None] * number_of_layers
	layer_state = [None] * (number_of_layers + 1)
	ffn_upscale_matrix = [None] * number_of_layers
	normalized_layer_state = [None] * number_of_layers
	normalized_ffn_input = [None] * number_of_layers
	scaled_layer_state = [None] * number_of_layers
	scaled_ffn_input = [None] * number_of_layers
	attention_inverse_root_mean_square = [None] * number_of_layers
	ffn_inverse_root_mean_square = [None] * number_of_layers

	layer_state[0] = combined_input_embeddings

	#@ Layers 1 - number_of_layers

	for layer in range(0, number_of_layers):
		layer_state_input = layer_state[layer].float()
		root_mean_square = torch.sqrt(layer_state_input.square().mean(dim = -1, keepdim = True) + epsilon)
		attention_inverse_root_mean_square[layer] = 1 / root_mean_square
		normalized_layer_state[layer] = (layer_state_input * attention_inverse_root_mean_square[layer]).to(torch.float16)
		scaled_layer_state[layer] = normalized_layer_state[layer] * fp16_attention_gamma_table[layer]
		(query_matrix[layer], key_matrix[layer], value_matrix[layer], 
		attention_weight_matrix[layer], attention_output_matrix[layer]) = run_forward_attention(layer, layer_state[layer], scaled_layer_state[layer])

		ffn_input = attention_output_matrix[layer].float()
		root_mean_square = torch.sqrt(ffn_input.square().mean(dim = -1, keepdim = True) + epsilon)
		ffn_inverse_root_mean_square[layer] = 1 / root_mean_square
		normalized_ffn_input[layer] = (ffn_input * ffn_inverse_root_mean_square[layer]).to(torch.float16)
		scaled_ffn_input[layer] = normalized_ffn_input[layer] * fp16_ffn_gamma_table[layer]

		ffn_upscale_matrix[layer] = scaled_ffn_input[layer] @ fp16_ffn_up_weight_table[layer]
		
		ffn_upscale_matrix[layer].relu_()

		layer_state[layer + 1] = ffn_upscale_matrix[layer] @ fp16_ffn_down_weight_table[layer] + attention_output_matrix[layer]

	#% Shape: context_length x number_of_ids
	#* Dot products attention context vectors with each vocab character's output weights,
	#* so if logits_matrix[10][20] is high, it means the attenuated output vector at position 10 supports the usage of character_id 20, so 'u' in this case
	logits_matrix = layer_state[number_of_layers] @ fp16_output_weight_table.transpose(-2, -1)

	#% Shape: context_length x number_of_ids
	#* Softmaxes logits_matrix row by row, so each position gets softmaxed where the character ids are turned into probabilities
	log_probability_matrix = F.log_softmax(logits_matrix.float(),dim=-1,)

	probability_matrix = (log_probability_matrix.exp().to(torch.float16))

	average_loss = -log_probability_matrix[batch_indices,context_indices,target_token_ids,].mean()

	# 0: input_token_ids, 1: target_token_ids, 2: input_token_embeddings
	# 3: input_position_embeddings, 4: combined_input_embeddings, 5: query_matrix
	# 6: key_matrix, 7: value_matrix, 8: attention_score_matrix
	# 9: attention_weight_matrix, 10: attention_output_matrix, 11: logits_matrix
	# 12: probability_matrix, 13: average loss, 14: ffn_upscale_matrix
	# 15: layer_state, 16: normalized_layer_state, 17: normalized_ffn_input
	# 18: attention_inverse_root_mean_square, 19: ffn_inverse_root_mean_square
	# 20: scaled_layer_state, 21: scaled_ffn_input

	return [input_token_ids, target_token_ids, 0,
			0, combined_input_embeddings, query_matrix,
			key_matrix, value_matrix, 0,
			attention_weight_matrix, attention_output_matrix, 0,
			probability_matrix, average_loss, ffn_upscale_matrix,
			layer_state, normalized_layer_state, normalized_ffn_input, 
			attention_inverse_root_mean_square, ffn_inverse_root_mean_square,
			scaled_layer_state, scaled_ffn_input]



def run_backpropogation_attention(layer, scaled_layer_state, normalized_layer_state, query_matrix, key_matrix, value_matrix, attention_weight_matrix, attention_output_gradient, inverse_root_mean_square):
	#% Shape: context_length x embedding_size
	#$ dLoss / dValue_matrix_value
	attention_output_gradient = attention_output_gradient.reshape(batch_size, CONTEXT_LENGTH, head_count, embedding_size // head_count).transpose(1, 2)

	value_matrix_gradient = attention_weight_matrix.transpose(-2, -1) @ attention_output_gradient

	#% Shape: context_length x context_length
	#$ dLoss / dAttention_weight_value
	attention_weight_gradient = attention_output_gradient @ value_matrix.transpose(-2, -1)

	#$ dLoss / dAttention_score_value
	weighted_blame_per_row = (attention_weight_matrix * attention_weight_gradient).sum(dim=-1, keepdim=True,)

	attention_weight_gradient.sub_(weighted_blame_per_row)

	attention_weight_gradient.mul_(attention_weight_matrix)

	attention_score_gradient = attention_weight_gradient

	#$ dLoss / dQuery_matrix_value
	attention_score_gradient /= math.sqrt(embedding_size // head_count)
	query_matrix_gradient = (attention_score_gradient @ key_matrix)
	
	#$ dLoss / dKey_matrix_value
	key_matrix_gradient = (attention_score_gradient.transpose(-2, -1) @ query_matrix)

	query_matrix_gradient = query_matrix_gradient.transpose(1, 2).reshape(batch_size, CONTEXT_LENGTH, embedding_size)
	key_matrix_gradient = key_matrix_gradient.transpose(1, 2).reshape(batch_size, CONTEXT_LENGTH, embedding_size)
	value_matrix_gradient = value_matrix_gradient.transpose(1, 2).reshape(batch_size, CONTEXT_LENGTH, embedding_size)

	#$ dLoss / dCombined_input_value
	qkv_matrix_gradient = torch.cat([query_matrix_gradient,key_matrix_gradient,value_matrix_gradient,],dim=-1,)

	scaled_layer_state_gradient = (qkv_matrix_gradient @ fp16_qkv_weight_table[layer])
	# Do not combine dims into sum(dim=(0,1)):
	# TorchInductor/ROCm produced incorrect reductions on gfx1031.
	attention_gamma_gradient = (scaled_layer_state_gradient.float() * normalized_layer_state.float()).sum(dim=0).sum(dim=0)

	normalized_layer_state_gradient = (scaled_layer_state_gradient * fp16_attention_gamma_table[layer])
	shared_scale_gradient = (normalized_layer_state_gradient.float() * normalized_layer_state.float()).mean(dim=-1, keepdim=True)

	input_embeddings_gradient = (inverse_root_mean_square * (normalized_layer_state_gradient - normalized_layer_state * shared_scale_gradient)).to(torch.float16)
	attention_output_gradient = attention_output_gradient.transpose(1, 2).reshape(batch_size, CONTEXT_LENGTH, embedding_size)
	input_embeddings_gradient += attention_output_gradient
	

	scaled_layer_state_flat = scaled_layer_state.reshape(-1, embedding_size)
	qkv_matrix_gradient_flat = qkv_matrix_gradient.reshape(-1, 3 * embedding_size,)

	qkv_weight_table_gradient = qkv_matrix_gradient_flat.transpose(-2, -1) @ scaled_layer_state_flat

	return [input_embeddings_gradient, qkv_weight_table_gradient, attention_gamma_gradient]


def run_backpropogation(forward_pass_results):
	#~ Globals
	if(1):
		global token_embedding_table
		global positional_embedding_table
		global qkv_weight_table
		global output_weight_table
		global ffn_up_weight_table
		global ffn_down_weight_table
		global fp16_token_embedding_table
		global fp16_positional_embedding_table
		global fp16_qkv_weight_table
		global fp16_output_weight_table
		global fp16_ffn_up_weight_table
		global fp16_ffn_down_weight_table
		global batch_indices
		global number_of_ids_indices
		global context_indices
		global attention_gamma_table
		global ffn_gamma_table

	#~ Inputs
	if(1):
		#@ Turning characters into ids
		input_token_ids = forward_pass_results[0]
		target_token_ids = forward_pass_results[1]

		#@ Embedding character ids into vectors of embedding_size and combining input + positional data
		# input_token_embeddings = forward_pass_results[2]
		# input_position_embeddings = forward_pass_results[3]
		combined_input_embeddings = forward_pass_results[4]

		#@ Using combined_input_embeddings + respective matrix values to figure out what it should ask, advertise, and give
		query_matrix = forward_pass_results[5]
		key_matrix = forward_pass_results[6]
		value_matrix = forward_pass_results[7]

		#@ Attention results using QKV
		# attention_score_matrix = forward_pass_results[8]
		attention_weight_matrix = forward_pass_results[9]
		attention_output_matrix = forward_pass_results[10]

		#@
		# logits_matrix = forward_pass_results[11]
		probability_matrix = forward_pass_results[12]

		#@ FFN layers
		ffn_upscale_matrix = forward_pass_results[14]
		layer_state = forward_pass_results[15]

		#@ Normalization
		normalized_layer_state = forward_pass_results[16]
		normalized_ffn_input = forward_pass_results[17]
		attention_inverse_root_mean_square = forward_pass_results[18]
		ffn_inverse_root_mean_square = forward_pass_results[19]
		scaled_layer_state = forward_pass_results[20]
		scaled_ffn_input = forward_pass_results[21]

	# for layer in range(number_of_layers + 1):
	# 	state = layer_state[layer]
	# 	print(
	# 		f"Layer state {layer}: "
	# 		f"max={state.abs().max().item():.4f}, "
	# 		f"rms={state.float().square().mean().sqrt().item():.4f}"
	# 	)


	qkv_weight_table_gradient = [None] * number_of_layers
	ffn_up_weight_table_gradient = [None] * number_of_layers
	ffn_down_weight_table_gradient = [None] * number_of_layers
	attention_gamma_gradient = [None] * number_of_layers
	ffn_gamma_gradient = [None] * number_of_layers

	logits_gradient = probability_matrix

	#* For each row in row_indices, select the column in that row index from target_token_ids, and only those are subtracted by 1
	logits_gradient[batch_indices, context_indices, target_token_ids] -= 1
	logits_gradient /= batch_size * CONTEXT_LENGTH / gradient_scale

	output_gradient = logits_gradient @ fp16_output_weight_table

	for layer in range(number_of_layers-1, -1, -1):
		#$ dLoss / dFFN_activation_matrix_layer_value
		#* Upscale_matrix_gradient is the exact same as activation_matrix_gradient, so saving operations here
		ffn_upscale_matrix_gradient = output_gradient @ fp16_ffn_down_weight_table[layer].transpose(-2, -1)

		#$ dLoss / dFFN_down_weight_table_layer_value
		#* ffn_upscale_matrix_layer replaces wherever ffn_activation_matrix should be
		ffn_activation_matrix_flat = ffn_upscale_matrix[layer].reshape(-1, ffn_hidden_size[layer])
		output_gradient_flat = output_gradient.reshape(-1, embedding_size)
		ffn_down_weight_table_gradient[layer] = ffn_activation_matrix_flat.transpose(-2, -1) @ output_gradient_flat

		#$ dLoss / dFFN_upscale_matrix_layer_value
		ffn_upscale_matrix_gradient.masked_fill_(ffn_upscale_matrix[layer] <= 0, 0.0)

		#$ dLoss / dFFN_up_weight_table_layer3_value
		ffn_upscale_matrix_gradient_flat = ffn_upscale_matrix_gradient.reshape(-1, ffn_hidden_size[layer])
		scaled_ffn_input_flat = scaled_ffn_input[layer].reshape(-1, embedding_size)
		ffn_up_weight_table_gradient[layer] = scaled_ffn_input_flat.transpose(-2, -1) @ ffn_upscale_matrix_gradient_flat

		#$ dLoss / dLayer3_attention_output_value
		scaled_attention_output_gradient = ffn_upscale_matrix_gradient @ fp16_ffn_up_weight_table[layer].transpose(-2, -1)
		# Do not combine dims into sum(dim=(0,1)):
		# TorchInductor/ROCm produced incorrect reductions on gfx1031.
		ffn_gamma_gradient[layer] = (scaled_attention_output_gradient.float() * normalized_ffn_input[layer].float()).sum(dim=0).sum(dim=0)

		normalized_attention_output_gradient = (scaled_attention_output_gradient * fp16_ffn_gamma_table[layer])
		shared_scale_gradient = (normalized_attention_output_gradient.float() * normalized_ffn_input[layer].float()).mean(dim=-1, keepdim=True)

		attention_output_gradient = (ffn_inverse_root_mean_square[layer] * (normalized_attention_output_gradient - normalized_ffn_input[layer] * shared_scale_gradient)).to(torch.float16)
		attention_output_gradient.add_(output_gradient)

		(output_gradient, qkv_weight_table_gradient[layer], attention_gamma_gradient[layer]) = run_backpropogation_attention(layer, scaled_layer_state[layer],normalized_layer_state[layer], query_matrix[layer], key_matrix[layer], 
																							   value_matrix[layer], attention_weight_matrix[layer], attention_output_gradient, 
																							   attention_inverse_root_mean_square[layer])

	combined_input_embeddings_gradient = output_gradient

	#$ dLoss / dOutput_weight_values
	logits_gradient_flat = logits_gradient.reshape(-1, number_of_ids)
	layer_state_flat = layer_state[number_of_layers].reshape(-1, embedding_size)
	output_weight_table_gradient = logits_gradient_flat.transpose(-2, -1) @ layer_state_flat

	#$ dLoss / dPositional_embedding_table_value
	input_position_embeddings_gradient = combined_input_embeddings_gradient

	#$ dLoss / dToken_embedding_table_value
	flattened_input_token_ids = input_token_ids.reshape(-1)

	flattened_embedding_gradient = (combined_input_embeddings_gradient.reshape(-1, embedding_size))

	input_token_embeddings_gradient = torch.zeros_like(fp16_token_embedding_table)

	input_token_embeddings_gradient.index_add_(0,flattened_input_token_ids,flattened_embedding_gradient,)

	# + Learning
	#& Output
	scaled_learning_rate = learning_rate_tensor / gradient_scale

	output_weight_table -= output_weight_table_gradient.float() * scaled_learning_rate

	#& FFN tables
	for layer in range(number_of_layers):
		ffn_up_weight_table[layer] -= ffn_up_weight_table_gradient[layer].float()  * scaled_learning_rate
		ffn_down_weight_table[layer] -= ffn_down_weight_table_gradient[layer].float()  * scaled_learning_rate

	#& Gamma tables
	attention_gamma_gradient = torch.stack(attention_gamma_gradient, dim=0)
	ffn_gamma_gradient = torch.stack(ffn_gamma_gradient, dim=0)

	attention_gamma_table -= attention_gamma_gradient * scaled_learning_rate
	ffn_gamma_table -= ffn_gamma_gradient * scaled_learning_rate

	#& QKV tables
	qkv_weight_table -= (torch.stack(qkv_weight_table_gradient,dim=0,).float()* scaled_learning_rate)

	#& Token embedding tables
	positional_embedding_table -= input_position_embeddings_gradient.sum(dim=0).float()  * scaled_learning_rate
	token_embedding_table -= input_token_embeddings_gradient.float()  * scaled_learning_rate

	refresh_fp16_matrices()

def training_step(start_positions, source_token_ids):
	forward_results = run_forward_pass(
		start_positions,
		source_token_ids,
	)

	loss = forward_results[13]
	run_backpropogation(forward_results)

	return loss

def set_learning_rate(loss):
	learning_rate = 0.01
	# if (loss < 1.6):
	# 	learning_rate = 0.025
	# if (loss < 1.5):
	# 	learning_rate = 0.020
	if (loss < 1.47):
		learning_rate = 0.0045
	if (loss < 1.435):
		learning_rate = 0.0015
	learning_rate *= learning_user_scale * persistent_learning_rate
	learning_rate_tensor.fill_(learning_rate)	

compiled_training_step = torch.compile(
	training_step,
	mode="default",
	dynamic=False,
)


#! Settings                                                 
embedding_size = 512
CONTEXT_LENGTH = 128
batch_size = 128
gradient_scale = 256
head_count = 4
number_of_training_windows = len(token_ids) - CONTEXT_LENGTH
load_persistent_learning = True
ffn_hidden_multiplier = [4, 3, 3, 3, 3]


example_start_positions = torch.arange(CONTEXT_LENGTH, CONTEXT_LENGTH * (batch_size + 1), step=CONTEXT_LENGTH)[:, None]
number_of_layers = len(ffn_hidden_multiplier)
learning_rate = 0.001
learning_user_scale = 2.0

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

	token_embedding_table = torch.rand(number_of_ids, embedding_size) * (2 * scale) - scale

	#% Shape: context_length x embedding_size
	#* This gives each possible position in the context window its own learnable vector, so the model can distinguish position 0 from position 1, position 2, etc.
	positional_embedding_table = torch.rand(CONTEXT_LENGTH, embedding_size) * (2 * scale) - scale

	#% Shape: 3 x embedding_size x embedding_size
	qkv_weight_table = torch.rand(number_of_layers, 3 * embedding_size, embedding_size ) * (2 * scale) - scale

	#% Shape: number_of_ids x embedding_size
	#* Determines/asks how much an attention_output vector supports a character being next
	output_weight_table = torch.rand(number_of_ids, embedding_size) * (2 * scale) - scale

	ffn_up_weight_table = [torch.rand(embedding_size, ffn_hidden_size[i], device=device) * (2 * ffn_up_scale) - ffn_up_scale for i in range(number_of_layers)]
	ffn_down_weight_table = [torch.rand(ffn_hidden_size[i], embedding_size, device=device) * (2 * ffn_down_scale[i]) - ffn_down_scale[i] for i in range(number_of_layers)]

	attention_gamma_table = torch.ones(number_of_layers, embedding_size, device=device)
	ffn_gamma_table = torch.ones(number_of_layers, embedding_size, device=device)

	#% Shape: context_length x context_length
	#* Shows what positions a given position is allowed to look at
	future_mask = torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1).bool()

	#$ Moving to GPU
	token_embedding_table = token_embedding_table.to(device)
	positional_embedding_table = positional_embedding_table.to(device)

	qkv_weight_table = qkv_weight_table.to(device)
	output_weight_table = output_weight_table.to(device)
	
	future_mask = future_mask.to(device)

	batch_indices = torch.arange(batch_size, device=device, dtype=torch.long)[:, None]
	number_of_ids_indices = torch.arange(number_of_ids, device=device, dtype=torch.long)
	context_indices = torch.arange(CONTEXT_LENGTH, dtype=torch.long, device=device)[None, :]
	context_indices_plus1 = torch.arange(CONTEXT_LENGTH + 1, dtype=torch.long)[None, :]
	learning_rate_tensor = torch.tensor(learning_rate, device=device, dtype=torch.float32)

	fp16_token_embedding_table = token_embedding_table.to(torch.float16)
	fp16_positional_embedding_table = positional_embedding_table.to(torch.float16)
	fp16_qkv_weight_table = qkv_weight_table.to(torch.float16)
	fp16_output_weight_table = output_weight_table.to(torch.float16)
	fp16_ffn_up_weight_table = [ffn_up_weight_table[i].to(torch.float16) for i in range(number_of_layers)]
	fp16_ffn_down_weight_table = [ffn_down_weight_table[i].to(torch.float16) for i in range(number_of_layers)]
	fp16_attention_gamma_table = attention_gamma_table.to(torch.float16)
	fp16_ffn_gamma_table = ffn_gamma_table.to(torch.float16)

iteration = 0
loss_window_average = 0.0
window_loss_size = 400
checkpoint_interval = 6000
learning_rate_patience = 22
last_improvement_window = 0
persistent_learning_rate = 1
number_of_validation_batches = 32
fixed_validation_positions = torch.randint(high = len(validation_token_ids) - CONTEXT_LENGTH, size = (number_of_validation_batches, batch_size, 1), dtype = torch.long)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_path = CHECKPOINT_DIR / "best_model.pt"

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


#! ============================================================
#! Batch size + learning-rate benchmark
#! ============================================================

batch_sizes_to_test = [4, 8, 16, 32, 64, 128, 256]

learning_rates_to_test = [
    0.00125,  # 0.005 / 4
    0.0025,  # 0.005 / 2
    0.0050,  # baseline
    0.0100,  # 0.005 * 2
]

benchmark_seconds = 60.0 * 5
warmup_steps = 20

validation_examples = 2048
validation_batch_size = 256

# Stop obviously unstable trials early.
unstable_loss_threshold = 50.0


# ============================================================
# Load benchmark starting checkpoint
# ============================================================

BENCHMARK_CHECKPOINT_DIR = (
	Path(__file__).resolve().parent
	/ "checkpoints"
)

benchmark_checkpoint_path = (
	BENCHMARK_CHECKPOINT_DIR
	/ "best_model.pt"
)

if not benchmark_checkpoint_path.exists():
	raise FileNotFoundError(
		f"Benchmark checkpoint not found:\n"
		f"{benchmark_checkpoint_path}"
	)

print(
	f"\nLoading benchmark starting model from:\n"
	f"{benchmark_checkpoint_path}"
)

(
	loaded_iteration,
	loaded_best_loss,
	loaded_persistent_learning_rate,
	loaded_elapsed_seconds,
) = load_model(
	benchmark_checkpoint_path
)

print(
	f"Benchmark starting checkpoint:\n"
	f"iteration = {loaded_iteration}\n"
	f"best loss = {loaded_best_loss:.6f}"
)


# ============================================================
# Parameter snapshot / restore
# ============================================================

def snapshot_parameters():
	return {
		"token_embedding_table":
			token_embedding_table.clone(),

		"positional_embedding_table":
			positional_embedding_table.clone(),

		"qkv_weight_table":
			qkv_weight_table.clone(),

		"output_weight_table":
			output_weight_table.clone(),

		"attention_gamma_table":
			attention_gamma_table.clone(),

		"ffn_gamma_table":
			ffn_gamma_table.clone(),

		"ffn_up_weight_table": [
			table.clone()
			for table in ffn_up_weight_table
		],

		"ffn_down_weight_table": [
			table.clone()
			for table in ffn_down_weight_table
		],
	}


def restore_parameters(snapshot):
	token_embedding_table.copy_(
		snapshot["token_embedding_table"]
	)

	positional_embedding_table.copy_(
		snapshot["positional_embedding_table"]
	)

	qkv_weight_table.copy_(
		snapshot["qkv_weight_table"]
	)

	output_weight_table.copy_(
		snapshot["output_weight_table"]
	)

	attention_gamma_table.copy_(
		snapshot["attention_gamma_table"]
	)

	ffn_gamma_table.copy_(
		snapshot["ffn_gamma_table"]
	)

	for layer in range(number_of_layers):
		ffn_up_weight_table[layer].copy_(
			snapshot["ffn_up_weight_table"][layer]
		)

		ffn_down_weight_table[layer].copy_(
			snapshot["ffn_down_weight_table"][layer]
		)

	refresh_fp16_matrices()


# IMPORTANT:
# This snapshot now represents the LOADED TRAINED MODEL,
# not the random initialization.
initial_parameters = snapshot_parameters()


# ============================================================
# Batch-dependent globals
# ============================================================

def configure_batch_size(new_batch_size):
	global batch_size
	global batch_indices
	global context_indices
	global fixed_validation_positions

	batch_size = new_batch_size

	batch_indices = torch.arange(
		batch_size,
		device=device,
		dtype=torch.long,
	)[:, None]

	context_indices = torch.arange(
		CONTEXT_LENGTH,
		device=device,
		dtype=torch.long,
	)[None, :]


# ============================================================
# Validation
# ============================================================

def calculate_validation_loss(validation_positions):
	total_weighted_loss = 0.0
	total_examples = 0

	original_batch_size = batch_size

	for start_index in range(
		0,
		validation_positions.shape[0],
		validation_batch_size,
	):
		current_positions = validation_positions[
			start_index:
			start_index + validation_batch_size
		]

		current_validation_batch_size = (
			current_positions.shape[0]
		)

		configure_batch_size(
			current_validation_batch_size
		)

		forward_results = run_forward_pass(
			current_positions[:, None],
			validation_token_ids,
		)

		current_loss = forward_results[13].item()

		total_weighted_loss += (
			current_loss
			* current_validation_batch_size
		)

		total_examples += (
			current_validation_batch_size
		)

	configure_batch_size(original_batch_size)

	return total_weighted_loss / total_examples


# ============================================================
# Fixed benchmark data
# ============================================================

number_of_validation_windows = (
	len(validation_token_ids) - CONTEXT_LENGTH
)

validation_positions = torch.randint(
	low=0,
	high=number_of_validation_windows,
	size=(validation_examples,),
	dtype=torch.long,
)

# Every benchmark reads from the SAME training stream.
# Batch sizes only group those examples differently.
maximum_training_examples = 50_000_000

fixed_training_positions = torch.randint(
	low=0,
	high=number_of_training_windows,
	size=(maximum_training_examples,),
	dtype=torch.long,
)


# ============================================================
# Benchmark
# ============================================================

results = []

for test_batch_size in batch_sizes_to_test:

	# --------------------------------------------------------
	# Configure + compile separately for this batch size.
	# dynamic=False means each batch should get its own graph.
	# --------------------------------------------------------

	configure_batch_size(test_batch_size)

	print()
	print("=" * 80)
	print(
		f"Compiling batch size {test_batch_size}"
	)
	print("=" * 80)

	compiled_test_step = torch.compile(
		training_step,
		mode="default",
		dynamic=False,
		backend="inductor",
	)

	for test_learning_rate in learning_rates_to_test:

		print()
		print("=" * 80)
		print(
			f"Testing batch={test_batch_size}, "
			f"learning_rate={test_learning_rate}"
		)
		print("=" * 80)

		# ----------------------------------------------------
		# Restore exact same initial model
		# ----------------------------------------------------

		restore_parameters(initial_parameters)
		configure_batch_size(test_batch_size)

		learning_rate_tensor.fill_(
			test_learning_rate
		)

		# ----------------------------------------------------
		# Warmup
		# ----------------------------------------------------

		warmup_offset = 0
		last_warmup_loss = None

		for warmup_iteration in range(warmup_steps):

			positions = fixed_training_positions[
				warmup_offset:
				warmup_offset + test_batch_size
			]

			warmup_offset += test_batch_size

			last_warmup_loss = compiled_test_step(
				positions[:, None],
				token_ids,
			)

		# Synchronize GPU.
		last_warmup_loss.item()

		# ----------------------------------------------------
		# Throw away all warmup learning
		# ----------------------------------------------------

		restore_parameters(initial_parameters)
		configure_batch_size(test_batch_size)

		learning_rate_tensor.fill_(
			test_learning_rate
		)

		# ----------------------------------------------------
		# Initial validation
		# ----------------------------------------------------

		initial_validation_loss = (
			calculate_validation_loss(
				validation_positions
			)
		)

		configure_batch_size(test_batch_size)

		print(
			f"Initial validation loss: "
			f"{initial_validation_loss:.6f}"
		)

		# ----------------------------------------------------
		# Timed training
		# ----------------------------------------------------

		training_offset = 0
		measured_steps = 0
		last_training_loss = None
		unstable = False

		start_time = time.perf_counter()

		while True:

			positions = fixed_training_positions[
				training_offset:
				training_offset + test_batch_size
			]

			if (
				positions.shape[0]
				!= test_batch_size
			):
				raise RuntimeError(
					"fixed_training_positions is "
					"too short for this benchmark."
				)

			training_offset += test_batch_size

			last_training_loss = compiled_test_step(
				positions[:, None],
				token_ids,
			)

			measured_steps += 1

			# Check every 10 steps so we do not constantly
			# synchronize the GPU.
			if measured_steps % 10 == 0:

				current_training_loss = (
					last_training_loss.item()
				)

				if (
					not math.isfinite(
						current_training_loss
					)
					or current_training_loss
					> unstable_loss_threshold
				):
					print(
						"Trial became unstable. "
						f"Loss={current_training_loss}"
					)

					unstable = True
					break

				elapsed_time = (
					time.perf_counter()
					- start_time
				)

				if elapsed_time >= benchmark_seconds:
					break

		# Final synchronization.
		if last_training_loss is not None:
			final_training_loss = (
				last_training_loss.item()
			)
		else:
			final_training_loss = float("nan")

		benchmark_time = (
			time.perf_counter()
			- start_time
		)

		# ----------------------------------------------------
		# Validation
		# ----------------------------------------------------

		if unstable:
			final_validation_loss = float("inf")

		else:
			final_validation_loss = (
				calculate_validation_loss(
					validation_positions
				)
			)

		configure_batch_size(test_batch_size)

		# ----------------------------------------------------
		# Metrics
		# ----------------------------------------------------

		measured_examples = (
			measured_steps
			* test_batch_size
		)

		measured_characters = (
			measured_examples
			* CONTEXT_LENGTH
		)

		examples_per_second = (
			measured_examples
			/ benchmark_time
		)

		characters_per_second = (
			measured_characters
			/ benchmark_time
		)

		steps_per_second = (
			measured_steps
			/ benchmark_time
		)

		time_per_step_ms = (
			benchmark_time
			/ measured_steps
			* 1000
		)

		if unstable:
			loss_drop = float("-inf")
			loss_drop_per_second = float("-inf")

		else:
			loss_drop = (
				initial_validation_loss
				- final_validation_loss
			)

			loss_drop_per_second = (
				loss_drop
				/ benchmark_time
			)

		result = {
			"batch_size":
				test_batch_size,

			"learning_rate":
				test_learning_rate,

			"initial_validation_loss":
				initial_validation_loss,

			"final_validation_loss":
				final_validation_loss,

			"loss_drop":
				loss_drop,

			"loss_drop_per_second":
				loss_drop_per_second,

			"characters_per_second":
				characters_per_second,

			"examples_per_second":
				examples_per_second,

			"steps_per_second":
				steps_per_second,

			"time_per_step_ms":
				time_per_step_ms,

			"measured_steps":
				measured_steps,

			"measured_examples":
				measured_examples,

			"benchmark_time":
				benchmark_time,

			"final_training_loss":
				final_training_loss,

			"unstable":
				unstable,
		}

		results.append(result)

		print()
		print(
			f"Final validation loss:   "
			f"{final_validation_loss:.6f}"
		)

		print(
			f"Final training loss:     "
			f"{final_training_loss:.6f}"
		)

		print(
			f"Validation loss drop:    "
			f"{loss_drop:.6f}"
		)

		print(
			f"Training steps:          "
			f"{measured_steps}"
		)

		print(
			f"Examples processed:      "
			f"{measured_examples:,}"
		)

		print(
			f"Characters/sec:          "
			f"{characters_per_second:,.2f}"
		)

		print(
			f"Examples/sec:            "
			f"{examples_per_second:,.2f}"
		)

		print(
			f"Steps/sec:               "
			f"{steps_per_second:.3f}"
		)

		print(
			f"Step time:               "
			f"{time_per_step_ms:.2f} ms"
		)

		print(
			f"Benchmark time:          "
			f"{benchmark_time:.2f} sec"
		)

		print(
			f"Unstable:                "
			f"{unstable}"
		)


# ============================================================
# Restore original model after benchmarking
# ============================================================

restore_parameters(initial_parameters)


# ============================================================
# Sort by FINAL validation loss
# ============================================================

results.sort(
	key=lambda result:
		result["final_validation_loss"]
)


# ============================================================
# Final table
# ============================================================

print()
print("=" * 185)

print(
	f"{'Rank':<7}"
	f"{'Batch':<9}"
	f"{'LR':<12}"
	f"{'Steps':<10}"
	f"{'Initial Val':<15}"
	f"{'Final Val':<15}"
	f"{'Loss Drop':<14}"
	f"{'Chars/Sec':<16}"
	f"{'Examples/Sec':<16}"
	f"{'Steps/Sec':<13}"
	f"{'Step ms':<12}"
	f"{'Time':<12}"
	f"{'Status':<12}"
)

print("=" * 185)

for rank, result in enumerate(
	results,
	start=1,
):

	status = (
		"UNSTABLE"
		if result["unstable"]
		else "OK"
	)

	print(
		f"{rank:<7}"
		f"{result['batch_size']:<9}"
		f"{result['learning_rate']:<12.5g}"
		f"{result['measured_steps']:<10}"
		f"{result['initial_validation_loss']:<15.6f}"
		f"{result['final_validation_loss']:<15.6f}"
		f"{result['loss_drop']:<14.6f}"
		f"{result['characters_per_second']:<16,.2f}"
		f"{result['examples_per_second']:<16,.2f}"
		f"{result['steps_per_second']:<13.3f}"
		f"{result['time_per_step_ms']:<12.2f}"
		f"{result['benchmark_time']:<12.2f}"
		f"{status:<12}"
	)

print("=" * 185)


# ============================================================
# Best configuration per batch size
# ============================================================

print()
print("BEST LEARNING RATE FOR EACH BATCH SIZE")
print("=" * 90)

for test_batch_size in batch_sizes_to_test:

	batch_results = [
		result
		for result in results
		if (
			result["batch_size"]
			== test_batch_size
			and not result["unstable"]
		)
	]

	if len(batch_results) == 0:
		print(
			f"Batch {test_batch_size}: "
			"No stable trials"
		)
		continue

	best_result = min(
		batch_results,
		key=lambda result:
			result["final_validation_loss"],
	)

	print(
		f"Batch {test_batch_size:<4} | "
		f"best LR={best_result['learning_rate']:<8.5g} | "
		f"val={best_result['final_validation_loss']:.6f} | "
		f"steps/sec={best_result['steps_per_second']:.3f} | "
		f"chars/sec={best_result['characters_per_second']:,.0f}"
	)


# ============================================================
# Overall winner
# ============================================================

stable_results = [
	result
	for result in results
	if not result["unstable"]
]

if len(stable_results) > 0:

	best_result = min(
		stable_results,
		key=lambda result:
			result["final_validation_loss"],
	)

	print()
	print("=" * 90)
	print("OVERALL WINNER")
	print("=" * 90)

	print(
		f"Batch size:           "
		f"{best_result['batch_size']}"
	)

	print(
		f"Learning rate:        "
		f"{best_result['learning_rate']}"
	)

	print(
		f"Final validation:     "
		f"{best_result['final_validation_loss']:.6f}"
	)

	print(
		f"Validation loss drop: "
		f"{best_result['loss_drop']:.6f}"
	)

	print(
		f"Steps/sec:            "
		f"{best_result['steps_per_second']:.3f}"
	)

	print(
		f"Characters/sec:       "
		f"{best_result['characters_per_second']:,.2f}"
	)

	print(
		f"Examples/sec:         "
		f"{best_result['examples_per_second']:,.2f}"
	)

	print("=" * 90)