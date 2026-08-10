#V6: 3 layers each with attention

import torch
import torch.nn.functional as F
import pandas as pd
import random
import math
import copy
import os
import time
from pathlib import Path
import csv

UNKNOWN_TOKEN = "<unk>"
mask_value = -2e4
device = torch.device("cuda")
torch.manual_seed(12354)

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

# with open("wiki.train.tokens", "r", encoding="utf-8") as f:
# 	dataText = f.read()

# training_text = dataText[:len(dataText) - 100000]
# validation_text = dataText[len(dataText) - 100000:]

# token_ids = torch.tensor(encode_text(training_text))
# validation_token_ids = torch.tensor(encode_text(validation_text))

#^ 1. Load the compressed parquet table
df = pd.read_parquet("train-00000-of-00001.parquet")

#^ 2. Extract the 'text' column rows and combine them
dataText = "\n".join(df["text"].astype(str))
training_text = dataText[:len(dataText) - 50000]
validation_text = dataText[len(dataText) - 50000:]

token_ids = torch.tensor(encode_text(training_text))
validation_token_ids = torch.tensor(encode_text(validation_text))

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

def synchronize(tensor):
    tensor.reshape(-1)[0].cpu().item()

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

		"embedding_size": embedding_size,
		"context_length": CONTEXT_LENGTH,
		"number_of_ids": number_of_ids,
		"ffn_hidden_size_layer1": ffn_hidden_size_layer1,
		"ffn_hidden_size_layer2": ffn_hidden_size_layer2,
		"ffn_hidden_size_layer3": ffn_hidden_size_layer3,
		"persistent_learning_rate": persistent_learning_rate,

		"token_embedding_table": token_embedding_table.detach().cpu(),
		"positional_embedding_table": (
			positional_embedding_table.detach().cpu()
		),
		"qkv_weight_table": qkv_weight_table.detach().cpu(),
		"output_weight_table": output_weight_table.detach().cpu(),
		"ffn_up_weight_table_layer1": ffn_up_weight_table_layer1.detach().cpu(),
		"ffn_down_weight_table_layer1": ffn_down_weight_table_layer1.detach().cpu(),
		"ffn_up_weight_table_layer2": ffn_up_weight_table_layer2.detach().cpu(),
		"ffn_down_weight_table_layer2": ffn_down_weight_table_layer2.detach().cpu(),
		"ffn_up_weight_table_layer3": ffn_up_weight_table_layer3.detach().cpu(),
		"ffn_down_weight_table_layer3": ffn_down_weight_table_layer3.detach().cpu(),
	}

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
	global ffn_up_weight_table_layer1
	global ffn_down_weight_table_layer1
	global ffn_up_weight_table_layer2
	global ffn_down_weight_table_layer2
	global ffn_up_weight_table_layer3
	global ffn_down_weight_table_layer3

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

	expected_qkv_shape = (3,3 * embedding_size,embedding_size,)

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

	if checkpoint["ffn_hidden_size_layer1"] != ffn_hidden_size_layer1:
		raise ValueError(
			"Cannot load checkpoint: ffn_hidden_size_layer1 size differs.\n"
			f"Checkpoint: {checkpoint['ffn_hidden_size_layer1']}\n"
			f"Current: {ffn_hidden_size_layer1}"
		)

	if checkpoint["ffn_hidden_size_layer2"] != ffn_hidden_size_layer2:
		raise ValueError(
			"Cannot load checkpoint: ffn_hidden_size_layer2 size differs.\n"
			f"Checkpoint: {checkpoint['ffn_hidden_size_layer2']}\n"
			f"Current: {ffn_hidden_size_layer2}"
		)	

	if checkpoint["ffn_hidden_size_layer3"] != ffn_hidden_size_layer3:
		raise ValueError(
			"Cannot load checkpoint: ffn_hidden_size_layer3 size differs.\n"
			f"Checkpoint: {checkpoint['ffn_hidden_size_layer3']}\n"
			f"Current: {ffn_hidden_size_layer3}"
		)
	
	# Your current tensors are CPU tensors.
	device = torch.device("cuda")
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

	ffn_up_weight_table_layer1 = checkpoint[
		"ffn_up_weight_table_layer1"
	].to(device)

	ffn_down_weight_table_layer1 = checkpoint[
		"ffn_down_weight_table_layer1"
	].to(device)

	ffn_up_weight_table_layer2 = checkpoint[
		"ffn_up_weight_table_layer2"
	].to(device)

	ffn_down_weight_table_layer2 = checkpoint[
		"ffn_down_weight_table_layer2"
	].to(device)

	ffn_up_weight_table_layer3 = checkpoint[
		"ffn_up_weight_table_layer3"
	].to(device)

	ffn_down_weight_table_layer3 = checkpoint[
		"ffn_down_weight_table_layer3"
	].to(device)

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
	return source_token_ids[start_positions + offsets]

def refresh_fp16_matrices():
	global fp16_token_embedding_table
	global fp16_positional_embedding_table
	global fp16_qkv_weight_table
	global fp16_output_weight_table
	global fp16_ffn_up_weight_table_layer1
	global fp16_ffn_down_weight_table_layer1
	global fp16_ffn_up_weight_table_layer2
	global fp16_ffn_down_weight_table_layer2
	global fp16_ffn_up_weight_table_layer3
	global fp16_ffn_down_weight_table_layer3
	fp16_token_embedding_table = token_embedding_table
	fp16_positional_embedding_table = positional_embedding_table
	fp16_qkv_weight_table = qkv_weight_table
	fp16_output_weight_table = output_weight_table
	fp16_ffn_up_weight_table_layer1 = ffn_up_weight_table_layer1
	fp16_ffn_down_weight_table_layer1 = ffn_down_weight_table_layer1
	fp16_ffn_up_weight_table_layer2 = ffn_up_weight_table_layer2
	fp16_ffn_down_weight_table_layer2 = ffn_down_weight_table_layer2
	fp16_ffn_up_weight_table_layer3 = ffn_up_weight_table_layer3
	fp16_ffn_down_weight_table_layer3 = ffn_down_weight_table_layer3

def run_forward_attention(layer, input_embeddings):
	#% Shape: 3 x context_length x embedding_size
	qkv_matrix = (input_embeddings @ fp16_qkv_weight_table[layer].transpose(-2, -1))

	query_matrix = qkv_matrix[...,0:embedding_size,]
	key_matrix = qkv_matrix[...,embedding_size:2 * embedding_size,]
	value_matrix = qkv_matrix[...,2 * embedding_size:3 * embedding_size,]
	
	#% Shape: context_length x context_length
	attention_score_matrix = query_matrix @ key_matrix.transpose(-2, -1) / math.sqrt(embedding_size)
	attention_score_matrix = attention_score_matrix.masked_fill(future_mask, mask_value)
	attention_weight_matrix = torch.softmax(attention_score_matrix.float(), dim=-1)

	#% Shape: context_length x embedding_size
	attention_output_matrix = torch.baddbmm(input_embeddings, attention_weight_matrix, value_matrix,)

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

	query_matrix = [None] * 3
	key_matrix = [None] * 3
	value_matrix = [None] * 3
	attention_weight_matrix = [None] * 3
	attention_output_matrix = [None] * 3
	

	#@ Layer 1
	(query_matrix[0], key_matrix[0], value_matrix[0], 
	attention_weight_matrix[0], attention_output_matrix[0]) = run_forward_attention(0, combined_input_embeddings)
	ffn_upscale_matrix_layer1 = attention_output_matrix[0] @ fp16_ffn_up_weight_table_layer1

	# ffn_upscale_matrix_layer1 = torch.relu(ffn_upscale_matrix_layer1)
	ffn_upscale_matrix_layer1.relu_()

	layer1_output = ffn_upscale_matrix_layer1 @ fp16_ffn_down_weight_table_layer1 + attention_output_matrix[0]

	#@ Layer 2
	(query_matrix[1], key_matrix[1], value_matrix[1], 
	attention_weight_matrix[1], attention_output_matrix[1]) = run_forward_attention(1, layer1_output)
	ffn_upscale_matrix_layer2 = attention_output_matrix[1] @ fp16_ffn_up_weight_table_layer2
	
	ffn_upscale_matrix_layer2.relu_()

	layer2_output = ffn_upscale_matrix_layer2 @ fp16_ffn_down_weight_table_layer2 + attention_output_matrix[1]

	#@ Layer 3
	(query_matrix[2], key_matrix[2], value_matrix[2], 
	attention_weight_matrix[2], attention_output_matrix[2]) = run_forward_attention(2, layer2_output)
	ffn_upscale_matrix_layer3 = attention_output_matrix[2] @ fp16_ffn_up_weight_table_layer3
		
	ffn_upscale_matrix_layer3.relu_()

	layer3_output = ffn_upscale_matrix_layer3 @ fp16_ffn_down_weight_table_layer3 + attention_output_matrix[2]

	#% Shape: context_length x number_of_ids
	#* Dot products attention context vectors with each vocab character's output weights,
	#* so if logits_matrix[10][20] is high, it means the attenuated output vector at position 10 supports the usage of character_id 20, so 'u' in this case
	logits_matrix = layer3_output @ fp16_output_weight_table.transpose(-2, -1)

	#% Shape: context_length x number_of_ids
	#* Softmaxes logits_matrix row by row, so each position gets softmaxed where the character ids are turned into probabilities
	log_probability_matrix = F.log_softmax(logits_matrix.float(),dim=-1,)

	probability_matrix = (log_probability_matrix.exp())

	average_loss = -log_probability_matrix[batch_indices,context_indices,target_token_ids,].mean()

	# 0: input_token_ids, 1: target_token_ids, 2: input_token_embeddings
	# 3: input_position_embeddings, 4: combined_input_embeddings, 5: query_matrix
	# 6: key_matrix, 7: value_matrix, 8: attention_score_matrix
	# 9: attention_weight_matrix, 10: attention_output_matrix, 11: logits_matrix
	# 12: probability_matrix, 13: average loss, 14: ffn_upscale_matrix_layer1
	# 15: layer1_output, 16: ffn_upscale_matrix_layer2, 17: layer2_output
	# 18: ffn_upscale_matrix_layer3, 19: layer3_output

	return [input_token_ids, target_token_ids, 0,
			0, combined_input_embeddings, query_matrix,
			key_matrix, value_matrix, 0,
			attention_weight_matrix, attention_output_matrix, 0,
			probability_matrix, average_loss, ffn_upscale_matrix_layer1, 
			layer1_output, ffn_upscale_matrix_layer2, layer2_output,
			ffn_upscale_matrix_layer3, layer3_output]

def run_backpropogation_attention(layer, input_embeddings, query_matrix, key_matrix, value_matrix, attention_weight_matrix, attention_output_gradient):
	#% Shape: context_length x embedding_size
	#$ dLoss / dValue_matrix_value
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
	attention_score_gradient /= math.sqrt(embedding_size)
	query_matrix_gradient = (attention_score_gradient @ key_matrix)
	
	#$ dLoss / dKey_matrix_value
	key_matrix_gradient = (attention_score_gradient.transpose(-2, -1) @ query_matrix)

	#$ dLoss / dCombined_input_value
	qkv_matrix_gradient = torch.cat([query_matrix_gradient,key_matrix_gradient,value_matrix_gradient,],dim=-1,)

	input_embeddings_gradient = (qkv_matrix_gradient @ fp16_qkv_weight_table[layer])
	input_embeddings_gradient += attention_output_gradient


	input_embeddings_flat = input_embeddings.reshape(-1, embedding_size)
	qkv_matrix_gradient_flat = qkv_matrix_gradient.reshape(-1, 3 * embedding_size,)

	qkv_weight_table_gradient = qkv_matrix_gradient_flat.transpose(-2, -1) @ input_embeddings_flat

	return [input_embeddings_gradient, qkv_weight_table_gradient]


def run_backpropogation(forward_pass_results):
	#~ Globals
	if(1):
		global token_embedding_table
		global positional_embedding_table
		global qkv_weight_table
		global output_weight_table
		global ffn_up_weight_table_layer1
		global ffn_down_weight_table_layer1
		global ffn_up_weight_table_layer2
		global ffn_down_weight_table_layer2
		global ffn_up_weight_table_layer3
		global ffn_down_weight_table_layer3
		global fp16_token_embedding_table
		global fp16_positional_embedding_table
		global fp16_qkv_weight_table
		global fp16_output_weight_table
		global fp16_ffn_up_weight_table_layer1
		global fp16_ffn_down_weight_table_layer1
		global fp16_ffn_up_weight_table_layer2
		global fp16_ffn_down_weight_table_layer2
		global fp16_ffn_up_weight_table_layer3
		global fp16_ffn_down_weight_table_layer3
		global batch_indices
		global number_of_ids_indices
		global context_indices

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

		#% context_length x embedding_size * 4
		ffn_upscale_matrix_layer1 = forward_pass_results[14]
		#% context_length x embedding_size
		layer1_output = forward_pass_results[15]

		#% context_length x embedding_size * 2
		ffn_upscale_matrix_layer2 = forward_pass_results[16]
		ffn_upscale_matrix_layer3 = forward_pass_results[18]
		#% context_length x embedding_size
		layer2_output = forward_pass_results[17]
		layer3_output = forward_pass_results[19]

	qkv_weight_table_gradient = [None] * 3

	logits_gradient = probability_matrix

	#* For each row in row_indices, select the column in that row index from target_token_ids, and only those are subtracted by 1
	logits_gradient[batch_indices, context_indices, target_token_ids] -= 1
	logits_gradient /= batch_size * CONTEXT_LENGTH / gradient_scale

	#@ Layer 3
	if(1):
		#$ dLoss / dFFN_downscale_matrix_layer3_value
		layer3_output_gradient = logits_gradient @ fp16_output_weight_table

		#$ dLoss / dFFN_activation_matrix_layer3_value
		#* Upscale_matrix_gradient is the exact same as activation_matrix_gradient, so saving operations here
		ffn_upscale_matrix_layer3_gradient = layer3_output_gradient @ fp16_ffn_down_weight_table_layer3.transpose(-2, -1)

		#$ dLoss / dFFN_down_weight_table_layer3_value
		#* ffn_upscale_matrix_layer3 replaces wherever ffn_activation_matrix should be
		ffn_activation_matrix_layer3_flat = ffn_upscale_matrix_layer3.reshape(-1, ffn_hidden_size_layer3)
		layer3_output_gradient_flat = layer3_output_gradient.reshape(-1, embedding_size)
		ffn_down_weight_table_layer3_gradient = ffn_activation_matrix_layer3_flat.transpose(-2, -1) @ layer3_output_gradient_flat

		#$ dLoss / dFFN_upscale_matrix_layer3_value
		ffn_upscale_matrix_layer3_gradient.masked_fill_(ffn_upscale_matrix_layer3 <= 0, 0.0)

		#$ dLoss / dFFN_up_weight_table_layer3_value
		ffn_upscale_matrix_layer3_gradient_flat = ffn_upscale_matrix_layer3_gradient.reshape(-1, ffn_hidden_size_layer3)
		layer3_attention_output_flat = attention_output_matrix[2].reshape(-1, embedding_size)
		ffn_up_weight_table_layer3_gradient = layer3_attention_output_flat.transpose(-2, -1) @ ffn_upscale_matrix_layer3_gradient_flat

		#$ dLoss / dLayer3_attention_output_value
		layer3_attention_output_gradient = ffn_upscale_matrix_layer3_gradient @ fp16_ffn_up_weight_table_layer3.transpose(-2, -1)
		layer3_attention_output_gradient.add_(layer3_output_gradient)

		(layer2_output_gradient, qkv_weight_table_gradient[2]) = run_backpropogation_attention(2, layer2_output, query_matrix[2], key_matrix[2], 
																							   value_matrix[2], attention_weight_matrix[2], layer3_attention_output_gradient)

	#@ Layer 2
	if(1):
		#$ dLoss / dFFN_activation_matrix_layer2_value
		#* Upscale_matrix_gradient is the exact same as activation_matrix_gradient, so saving operations here
		ffn_upscale_matrix_layer2_gradient = layer2_output_gradient @ fp16_ffn_down_weight_table_layer2.transpose(-2, -1)

		#$ dLoss / dFFN_down_weight_table_layer2_value
		#* ffn_upscale_matrix_layer2 replaces wherever ffn_activation_matrix should be
		ffn_activation_matrix_layer2_flat = ffn_upscale_matrix_layer2.reshape(-1, ffn_hidden_size_layer2)
		layer2_output_gradient_flat = layer2_output_gradient.reshape(-1, embedding_size)
		ffn_down_weight_table_layer2_gradient = ffn_activation_matrix_layer2_flat.transpose(-2, -1) @ layer2_output_gradient_flat

		#$ dLoss / dFFN_upscale_matrix_layer2_value
		ffn_upscale_matrix_layer2_gradient.masked_fill_(ffn_upscale_matrix_layer2 <= 0, 0.0)

		#$ dLoss / dFFN_up_weight_table_layer2_value
		ffn_upscale_matrix_layer2_gradient_flat = ffn_upscale_matrix_layer2_gradient.reshape(-1, ffn_hidden_size_layer2)
		layer2_attention_output_flat = attention_output_matrix[1].reshape(-1, embedding_size)
		ffn_up_weight_table_layer2_gradient = layer2_attention_output_flat.transpose(-2, -1) @ ffn_upscale_matrix_layer2_gradient_flat

		#$ dLoss / dLayer2_attention_output_value
		layer2_attention_output_gradient = ffn_upscale_matrix_layer2_gradient @ fp16_ffn_up_weight_table_layer2.transpose(-2, -1)
		layer2_attention_output_gradient.add_(layer2_output_gradient)

		(layer1_output_gradient, qkv_weight_table_gradient[1]) = run_backpropogation_attention(1, layer1_output, query_matrix[1], key_matrix[1], 
																							   value_matrix[1], attention_weight_matrix[1], layer2_attention_output_gradient)
	#@ Layer 1
	if(1):
		#$ dLoss / dFFN_activation_matrix_layer1_value
		#* Upscale_matrix_gradient is the exact same as activation_matrix_gradient, so saving operations here
		ffn_upscale_matrix_layer1_gradient = layer1_output_gradient @ fp16_ffn_down_weight_table_layer1.transpose(-2, -1)

		#$ dLoss / dFFN_down_weight_table_layer1_value
		#* ffn_upscale_matrix_layer1 replaces wherever ffn_activation_matrix should be
		ffn_activation_matrix_layer1_flat = ffn_upscale_matrix_layer1.reshape(-1, ffn_hidden_size_layer1)
		layer1_output_gradient_flat = layer1_output_gradient.reshape(-1, embedding_size)
		ffn_down_weight_table_layer1_gradient = ffn_activation_matrix_layer1_flat.transpose(-2, -1) @ layer1_output_gradient_flat

		#$ dLoss / dFFN_upscale_matrix_layer1_value
		ffn_upscale_matrix_layer1_gradient.masked_fill_(ffn_upscale_matrix_layer1 <= 0, 0.0)

		#$ dLoss / dFFN_up_weight_table_layer1_value
		ffn_upscale_matrix_layer1_gradient_flat = ffn_upscale_matrix_layer1_gradient.reshape(-1, ffn_hidden_size_layer1)
		layer1_attention_output_flat = attention_output_matrix[0].reshape(-1, embedding_size)
		ffn_up_weight_table_layer1_gradient = layer1_attention_output_flat.transpose(-2, -1) @ ffn_upscale_matrix_layer1_gradient_flat

		#$ dLoss / dAttention_output_value
		layer1_attention_output_gradient = ffn_upscale_matrix_layer1_gradient @ fp16_ffn_up_weight_table_layer1.transpose(-2, -1)
		layer1_attention_output_gradient.add_(layer1_output_gradient)

		(combined_input_embeddings_gradient, qkv_weight_table_gradient[0]) = run_backpropogation_attention(0, combined_input_embeddings, query_matrix[0], key_matrix[0], 
																									   value_matrix[0], attention_weight_matrix[0], layer1_attention_output_gradient)

	#$ dLoss / dOutput_weight_values
	logits_gradient_flat = logits_gradient.reshape(-1, number_of_ids)
	ffn_downscale_matrix_layer3_flat = layer3_output.reshape(-1, embedding_size)
	output_weight_table_gradient = logits_gradient_flat.transpose(-2, -1) @ ffn_downscale_matrix_layer3_flat

	#$ dLoss / dPositional_embedding_table_value
	input_position_embeddings_gradient = combined_input_embeddings_gradient

	#$ dLoss / dToken_embedding_table_value
	token_numbers = number_of_ids_indices
	flattened_input_token_ids = input_token_ids.reshape(-1)
	token_matches = (token_numbers[:, None] == flattened_input_token_ids[None, :]).float()

	input_token_embeddings_gradient = token_matches @ combined_input_embeddings_gradient.reshape(-1, embedding_size)

	# + Learning
	#& Output
	scaled_learning_rate = learning_rate / gradient_scale

	output_weight_table -= output_weight_table_gradient.float() * scaled_learning_rate

	#& FFN weight tables
	ffn_up_weight_table_layer3 -= ffn_up_weight_table_layer3_gradient.float()  * scaled_learning_rate
	ffn_down_weight_table_layer3 -= ffn_down_weight_table_layer3_gradient.float()  * scaled_learning_rate
	ffn_up_weight_table_layer2 -= ffn_up_weight_table_layer2_gradient.float()  * scaled_learning_rate
	ffn_down_weight_table_layer2 -= ffn_down_weight_table_layer2_gradient.float()  * scaled_learning_rate
	ffn_up_weight_table_layer1 -= ffn_up_weight_table_layer1_gradient.float()  * scaled_learning_rate
	ffn_down_weight_table_layer1 -= ffn_down_weight_table_layer1_gradient.float()  * scaled_learning_rate

	#& QKV weight tables
	qkv_weight_table -= (torch.stack(qkv_weight_table_gradient,dim=0,).float()* scaled_learning_rate)

	#& Token embedding tables
	positional_embedding_table -= input_position_embeddings_gradient.sum(dim=0).float()  * scaled_learning_rate
	token_embedding_table -= input_token_embeddings_gradient.float()  * scaled_learning_rate

	refresh_fp16_matrices()


#! Settings                                                 
embedding_size = 384
CONTEXT_LENGTH = 128
number_of_training_windows = len(token_ids) - CONTEXT_LENGTH
learning_rate = 0.001
learning_user_scale = 2.0
load_persistent_learning = True
ffn_hidden_size_layer1 = embedding_size * 4
ffn_hidden_size_layer2 = embedding_size * 2
ffn_hidden_size_layer3 = embedding_size * 2
batch_size = 128
gradient_scale = 256

example_start_positions = torch.arange(CONTEXT_LENGTH, CONTEXT_LENGTH * (batch_size + 1), step=CONTEXT_LENGTH)[:, None]

#& Embedding and weight table initialization + mask + arange indices
if (1):
	#% Shape: vocabulary_size × embedding_size. 
	#* This is the map for all characters in the vocabulary to their respective vector, initialized random at first and changes later to learn relationships.
	scale = 1 / math.sqrt(embedding_size)
	ffn_up_scale = 1 / math.sqrt(embedding_size)
	ffn_down_scale_layer1 = 1 / math.sqrt(ffn_hidden_size_layer1)
	ffn_down_scale_layer2 = 1 / math.sqrt(ffn_hidden_size_layer2)
	ffn_down_scale_layer3 = 1 / math.sqrt(ffn_hidden_size_layer3)
	token_embedding_table = torch.rand(number_of_ids, embedding_size) * (2 * scale) - scale

	#% Shape: context_length x embedding_size
	#* This gives each possible position in the context window its own learnable vector, so the model can distinguish position 0 from position 1, position 2, etc.
	positional_embedding_table = torch.rand(CONTEXT_LENGTH, embedding_size) * (2 * scale) - scale

	#% Shape: 3 x embedding_size x embedding_size
	qkv_weight_table = torch.rand(3, 3 * embedding_size, embedding_size,) * (2 * scale) - scale

	#% Shape: number_of_ids x embedding_size
	#* Determines/asks how much an attention_output vector supports a character being next
	output_weight_table = torch.rand(number_of_ids, embedding_size) * (2 * scale) - scale

	#@ layer 1
	#% Shape: embedding_size x embedding_size * 4 (or ffn_hidden_size)
	ffn_up_weight_table_layer1 = torch.rand(embedding_size, ffn_hidden_size_layer1) * (2 * ffn_up_scale) - ffn_up_scale
	#% Shape: embedding_size * 4 (or ffn_hidden_size) x embedding_size 
	ffn_down_weight_table_layer1 = torch.rand(ffn_hidden_size_layer1, embedding_size) * (2 * ffn_down_scale_layer1) - ffn_down_scale_layer1

	#@ layer 2
	#% Shape: embedding_size x embedding_size * 2 (or ffn_hidden_size)
	ffn_up_weight_table_layer2 = torch.rand(embedding_size, ffn_hidden_size_layer2) * (2 * ffn_up_scale) - ffn_up_scale
	#% Shape: embedding_size * 2 (or ffn_hidden_size) x embedding_size 
	ffn_down_weight_table_layer2 = torch.rand(ffn_hidden_size_layer2, embedding_size) * (2 * ffn_down_scale_layer2) - ffn_down_scale_layer2

	#@ layer 3
	#% Shape: embedding_size x embedding_size * 2 (or ffn_hidden_size)
	ffn_up_weight_table_layer3 = torch.rand(embedding_size, ffn_hidden_size_layer3) * (2 * ffn_up_scale) - ffn_up_scale
	#% Shape: embedding_size * 2 (or ffn_hidden_size) x embedding_size 
	ffn_down_weight_table_layer3 = torch.rand(ffn_hidden_size_layer3, embedding_size) * (2 * ffn_down_scale_layer3) - ffn_down_scale_layer3

	#% Shape: context_length x context_length
	#* Shows what positions a given position is allowed to look at
	future_mask = torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1).bool()

	batch_indices = torch.arange(batch_size, device=device, dtype=torch.long)[:, None]
	number_of_ids_indices = torch.arange(number_of_ids, device=device, dtype=torch.long)
	context_indices = torch.arange(CONTEXT_LENGTH, dtype=torch.long, device=device)[None, :]
	context_indices_plus1 = torch.arange(CONTEXT_LENGTH + 1, dtype=torch.long)[None, :]

	fp16_token_embedding_table = token_embedding_table.to(device)
	fp16_positional_embedding_table = positional_embedding_table.to(device)
	fp16_qkv_weight_table = qkv_weight_table.to(device)
	fp16_output_weight_table = output_weight_table.to(device)
	fp16_ffn_up_weight_table_layer1 = ffn_up_weight_table_layer1.to(device)
	fp16_ffn_down_weight_table_layer1 = ffn_down_weight_table_layer1.to(device)
	fp16_ffn_up_weight_table_layer2 = ffn_up_weight_table_layer2.to(device)
	fp16_ffn_down_weight_table_layer2 = ffn_down_weight_table_layer2.to(device)
	fp16_ffn_up_weight_table_layer3 = ffn_up_weight_table_layer3.to(device)
	fp16_ffn_down_weight_table_layer3 = ffn_down_weight_table_layer3.to(device)

	

	#$ Moving to GPU
	token_embedding_table = token_embedding_table.to(device)
	positional_embedding_table = positional_embedding_table.to(device)

	qkv_weight_table = qkv_weight_table.to(device)
	output_weight_table = output_weight_table.to(device)

	ffn_up_weight_table_layer1 = ffn_up_weight_table_layer1.to(device)
	ffn_down_weight_table_layer1 = ffn_down_weight_table_layer1.to(device)
	ffn_up_weight_table_layer2 = ffn_up_weight_table_layer2.to(device)
	ffn_down_weight_table_layer2 = ffn_down_weight_table_layer2.to(device)
	ffn_up_weight_table_layer3 = ffn_up_weight_table_layer3.to(device)
	ffn_down_weight_table_layer3 = ffn_down_weight_table_layer3.to(device)
	future_mask = future_mask.to(device)

iteration = 0
loss_window_average = 0.0
window_loss_size = 500
checkpoint_interval = 7500
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

if(current_least_loss < 2.465):
	learning_rate = 0.0012
elif (current_least_loss < 2.48):
	learning_rate = 0.0012
elif (current_least_loss < 2.5):
	learning_rate = 0.0025
elif (current_least_loss < 2.55):
	learning_rate = 0.0045
elif (current_least_loss < 2.6):
	learning_rate = 0.007
elif (current_least_loss < 2.7):
	learning_rate = 0.010
elif (current_least_loss < 2.8):
	learning_rate = 0.015
elif (current_least_loss < 3):
	learning_rate = 0.03
elif (current_least_loss < 4):
	learning_rate = 0.05
else:
	learning_rate = 0.1

try:
	with open("learning_scale.txt", "r") as file:
		learning_user_scale = float(file.read().strip())
except (OSError, ValueError):
	print("Could not read learning scale; keeping previous value.")
learning_rate *= learning_user_scale * persistent_learning_rate

window_start_time = time.perf_counter()
current_start_time = time.perf_counter()

forward_time = 0
backward_time = 0

#! Training loop                 
while True:
	iteration += 1

	start_positions = torch.randint(high = number_of_training_windows, size = (batch_size,), dtype = torch.long)[:, None]
	forward_pass_results = run_forward_pass(start_positions, token_ids)

	# This is the loss before the current parameter update.
	window_loss = forward_pass_results[13]

	run_backpropogation(forward_pass_results)

	loss_window_average += window_loss

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
	
		validation_average_loss = 0
		for validation_start_positions in fixed_validation_positions:
			validation_forward_pass_results = run_forward_pass(validation_start_positions, validation_token_ids)

			validation_average_loss += validation_forward_pass_results[13]

		validation_average_loss /= number_of_validation_batches
		validation_average_loss = validation_average_loss.item()
		print(f"Validation loss at iteration {iteration}: {(validation_average_loss)}")

		if validation_average_loss < current_least_loss:
			previous_least_loss = current_least_loss
			current_least_loss = validation_average_loss
			last_improvement_window = 0

			save_model(
				CHECKPOINT_DIR / "best_model.pt",
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
				f"No improvement for {learning_rate_patience} loss windows. Setting learning rate to {round(learning_rate * 0.9, 6)} and persistence_learning_rate to {round(persistent_learning_rate, 6)}. ")

			# Start a new patience period after rollback.
			last_improvement_window = 0

		if iteration % checkpoint_interval == 0:
			save_model(
				CHECKPOINT_DIR / f"checkpoint_{iteration}_{CONTEXT_LENGTH}_{embedding_size}.pt",
				iteration=iteration,
				best_loss=current_least_loss,
				previous_elapsed_seconds=previous_elapsed_seconds + current_elapsed_seconds
			)

		if(current_least_loss < 2.465):
			learning_rate = 0.0012
		elif (current_least_loss < 2.48):
			learning_rate = 0.0012
		elif (current_least_loss < 2.5):
			learning_rate = 0.0025
		elif (current_least_loss < 2.55):
			learning_rate = 0.0045
		elif (current_least_loss < 2.6):
			learning_rate = 0.007
		elif (current_least_loss < 2.7):
			learning_rate = 0.010
		elif (current_least_loss < 2.8):
			learning_rate = 0.015
		elif (current_least_loss < 3):
			learning_rate = 0.03
		elif (current_least_loss < 4):
			learning_rate = 0.05
		else:
			learning_rate = 0.1

		try:
			with open("learning_scale.txt", "r") as file:
				learning_user_scale = float(file.read().strip())
		except (OSError, ValueError):
			print("Could not read learning scale; keeping previous value.")
		learning_rate *= learning_user_scale * persistent_learning_rate
		
		loss_window_average = 0.0
		window_start_time = time.perf_counter()