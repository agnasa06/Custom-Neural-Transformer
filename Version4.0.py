#V4: Introduced batching and a few memory + FFN optimizations

import torch
import torch.nn.functional as F
import pandas as pd
import random
import math
import copy
import os
import copy
import time
from pathlib import Path
import csv

UNKNOWN_TOKEN = "<unk>"
mask_value = -1e9
device = torch.device("cuda")

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

#^ 1. Load the compressed parquet table
df = pd.read_parquet("train-00000-of-00001.parquet")

#^ 2. Extract the 'text' column rows and combine them
dataText = "\n".join(df["text"].astype(str))
training_text = dataText[:len(dataText) - 50000]
validation_text = dataText[len(dataText) - 50000:]

token_ids = torch.tensor(encode_text(training_text))
token_ids = token_ids.to(device)
validation_token_ids = torch.tensor(encode_text(validation_text))
validation_token_ids = validation_token_ids.to(device)

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

def save_model(path, iteration, best_loss):
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

		"embedding_size": embedding_size,
		"context_length": CONTEXT_LENGTH,
		"number_of_ids": number_of_ids,

		"token_embedding_table": token_embedding_table.detach().cpu(),
		"positional_embedding_table": (
			positional_embedding_table.detach().cpu()
		),
		"query_weight_table": query_weight_table.detach().cpu(),
		"key_weight_table": key_weight_table.detach().cpu(),
		"value_weight_table": value_weight_table.detach().cpu(),
		"output_weight_table": output_weight_table.detach().cpu(),
		"ffn_up_weight_table": ffn_up_weight_table.detach().cpu(),
		"ffn_down_weight_table": ffn_down_weight_table.detach().cpu(),
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
		iteration, best_loss
	"""

	global token_embedding_table
	global positional_embedding_table
	global query_weight_table
	global key_weight_table
	global value_weight_table
	global output_weight_table
	global ffn_up_weight_table
	global ffn_down_weight_table

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

	# Your current tensors are CPU tensors.
	# If you later use DirectML, change "cpu" to your DirectML device.
	device = torch.device("cuda")

	token_embedding_table = checkpoint[
		"token_embedding_table"
	].to(device)

	positional_embedding_table = checkpoint[
		"positional_embedding_table"
	].to(device)

	query_weight_table = checkpoint[
		"query_weight_table"
	].to(device)

	key_weight_table = checkpoint[
		"key_weight_table"
	].to(device)

	value_weight_table = checkpoint[
		"value_weight_table"
	].to(device)

	output_weight_table = checkpoint[
		"output_weight_table"
	].to(device)

	ffn_up_weight_table = checkpoint[
		"ffn_up_weight_table"
	].to(device)

	ffn_down_weight_table = checkpoint[
		"ffn_down_weight_table"
	].to(device)

	iteration = checkpoint.get("iteration", 0)
	best_loss = checkpoint.get("best_loss", float("inf"))

	print(f"Model loaded from: {path}")
	print(f"Resuming from iteration: {iteration}")
	print(f"Previous best loss: {best_loss:.6f}")

	return iteration, best_loss

def get_input_token_ids(start_positions, source_token_ids):
	offsets = context_indices
	return source_token_ids[start_positions + offsets]

def get_target_token_ids(start_positions, source_token_ids):
	offsets = context_indices
	return source_token_ids[start_positions + 1 + offsets]

def run_forward_pass(start_positions, source_token_ids):
	# Look up one embedding vector for each input token ID.
	input_token_ids = get_input_token_ids(start_positions, source_token_ids)
	target_token_ids = get_target_token_ids(start_positions, source_token_ids)

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

	#% Shape: context_length x embedding_size
	#* The asking matrix based on combined_input_embeddings
	query_matrix = combined_input_embeddings @ query_weight_table.transpose(-2, -1)

	#% Shape: context_length x embedding_size
	#* The advertising matrix based on combined_input_embeddings
	key_matrix = combined_input_embeddings @ key_weight_table.transpose(-2, -1)

	#% Shape: context_length x embedding_size
	#* The conveying matrix based on combined_input_embeddings
	value_matrix = combined_input_embeddings @ value_weight_table.transpose(-2, -1)
	
	#% Shape: context_length x context_length
	#* Stores the raw relevance scores between every current position and every position it is allowed to look at.
	attention_score_matrix = query_matrix @ key_matrix.transpose(-2, -1) / math.sqrt(embedding_size)
	attention_score_matrix = attention_score_matrix.masked_fill(future_mask, -1e9)

	#% Shape: context_length x context_length
	#* Stores how much each current position attends/cares to each previous/current position. Future positions receive weight 0.
	attention_weight_matrix = torch.softmax(attention_score_matrix, dim=-1)

	#% Shape: context_length x embedding_size
	#* Computes the necessary information from previous and current position for each position in context_length in a blended vector
	attention_output_matrix = attention_weight_matrix @ value_matrix

	ffn_upscale_matrix = attention_output_matrix @ ffn_up_weight_table

	# ffn_upscale_matrix = torch.relu(ffn_upscale_matrix)
	ffn_upscale_matrix.relu_()

	ffn_downscale_matrix = ffn_upscale_matrix @ ffn_down_weight_table

	#% Shape: context_length x number_of_ids
	#* Dot products attention context vectors with each vocab character's output weights,
	#* so if logits_matrix[10][20] is high, it means the attenuated output vector at position 10 supports the usage of character_id 20, so 'u' in this case
	logits_matrix = ffn_downscale_matrix @ output_weight_table.transpose(-2, -1)

	#% Shape: context_length x number_of_ids
	#* Softmaxes logits_matrix row by row, so each position gets softmaxed where the character ids are turned into probabilities
	probability_matrix = torch.softmax(logits_matrix, dim=-1)

	average_loss = F.cross_entropy(
		logits_matrix.view(-1, number_of_ids),
		target_token_ids.view(-1)
	)



	# 0: input_token_ids, 1: target_token_ids, 2: input_token_embeddings
	# 3: input_position_embeddings, 4: combined_input_embeddings, 5: query_matrix
	# 6: key_matrix, 7: value_matrix, 8: attention_score_matrix
	# 9: attention_weight_matrix, 10: attention_output_matrix, 11: logits_matrix
	# 12: probability_matrix, 13: average loss, 14: ffn_upscale_matrix
	# 15: ffn_activated_matrix, 16: ffn_downscale_matrix

	return [input_token_ids, target_token_ids, input_token_embeddings,
			input_position_embeddings, combined_input_embeddings, query_matrix,
			key_matrix, value_matrix, attention_score_matrix,
			attention_weight_matrix, attention_output_matrix, logits_matrix,
			probability_matrix, average_loss, ffn_upscale_matrix, 0, ffn_downscale_matrix]

def run_backpropogation(forward_pass_results):
	global token_embedding_table
	global positional_embedding_table
	global query_weight_table
	global key_weight_table
	global value_weight_table
	global output_weight_table
	global ffn_up_weight_table
	global ffn_down_weight_table
	global batch_indices
	global number_of_ids_indices
	global context_indices

	#@ Turning characters into ids
	input_token_ids = forward_pass_results[0]
	target_token_ids = forward_pass_results[1]

	#@ Embedding character ids into vectors of embedding_size and combining input + positional data
	input_token_embeddings = forward_pass_results[2]
	input_position_embeddings = forward_pass_results[3]
	combined_input_embeddings = forward_pass_results[4]

	#@ Using combined_input_embeddings + respective matrix values to figure out what it should ask, advertise, and give
	query_matrix = forward_pass_results[5]
	key_matrix = forward_pass_results[6]
	value_matrix = forward_pass_results[7]

	#@ Attention results using QKV
	attention_score_matrix = forward_pass_results[8]
	attention_weight_matrix = forward_pass_results[9]
	attention_output_matrix = forward_pass_results[10]

	#@
	logits_matrix = forward_pass_results[11]
	probability_matrix = forward_pass_results[12]

	#% context_length x embedding_size * 4
	ffn_upscale_matrix = forward_pass_results[14]
	#% context_length x embedding_size
	ffn_downscale_matrix = forward_pass_results[16]

	logits_gradient = probability_matrix.clone()

	#* For each row in row_indices, select the column in that row index from target_token_ids, and only those are subtracted by 1
	logits_gradient[batch_indices, context_indices, target_token_ids] -= 1
	logits_gradient /= batch_size * CONTEXT_LENGTH

	#$ dLoss / dFFN_downscale_matrix_value
	ffn_downscale_matrix_gradient = logits_gradient @ output_weight_table

	#$ dLoss / dFFN_activation_matrix_value
	#* Upscale_matrix_gradient is the exact same as activation_matrix_gradient, so saving operations here
	ffn_upscale_matrix_gradient =  ffn_downscale_matrix_gradient @ ffn_down_weight_table.transpose(-2, -1)

	#$ dLoss / dFFN_down_weight_table_value
	#* ffn_upscale_matrix replaces wherever ffn_activation_matrix should be
	ffn_activation_matrix_flat = ffn_upscale_matrix.reshape(-1, ffn_hidden_size)
	ffn_downscale_matrix_gradient_flat = ffn_downscale_matrix_gradient.reshape(-1, embedding_size)
	ffn_down_weight_table_gradient = ffn_activation_matrix_flat.transpose(-2, -1) @ ffn_downscale_matrix_gradient_flat

	#$ dLoss / dFFN_upscale_matrix_value
	ffn_upscale_matrix_gradient.masked_fill_(ffn_upscale_matrix <= 0, 0.0)

	#$ dLoss / dFFN_up_weight_table_value
	ffn_upscale_matrix_gradient_flat = ffn_upscale_matrix_gradient.reshape(-1, ffn_hidden_size)
	attention_output_matrix_flat = attention_output_matrix.reshape(-1, embedding_size)
	ffn_up_weight_table_gradient = attention_output_matrix_flat.transpose(-2, -1) @ ffn_upscale_matrix_gradient_flat

	#$ dLoss / dAttention_output_value
	attention_output_gradient = ffn_upscale_matrix_gradient @ ffn_up_weight_table.transpose(-2, -1)

	#$ dLoss / dOutput_weight_values
	logits_gradient_flat = logits_gradient.reshape(-1, number_of_ids)
	ffn_downscale_matrix_flat = ffn_downscale_matrix.reshape(-1, embedding_size)
	output_weight_table_gradient = logits_gradient_flat.transpose(-2, -1) @ ffn_downscale_matrix_flat

	#% Shape: context_length x embedding_size
	#$ dLoss / dValue_matrix_value
	value_matrix_gradient = attention_weight_matrix.transpose(-2, -1) @ attention_output_gradient

	#% Shape: context_length x context_length
	#$ dLoss / dAttention_weight_value
	attention_weight_gradient = attention_output_gradient @ value_matrix.transpose(-2, -1)

	# print(attention_output_gradient.shape)
	# print(attention_weight_gradient.shape)
	# print(value_matrix.shape)

	#$ dLoss / dAttention_score_value
	weighted_attention_blame = attention_weight_matrix * attention_weight_gradient
	weighted_blame_per_row = weighted_attention_blame.sum(dim=-1, keepdim=True)
	# print(weighted_blame_per_row.shape)
	blame_difference = attention_weight_gradient - weighted_blame_per_row
	attention_score_gradient = blame_difference * attention_weight_matrix

	#$ dLoss / dQuery_matrix_value
	query_matrix_gradient = (attention_score_gradient @ key_matrix) / math.sqrt(embedding_size)
	
	#$ dLoss / dKey_matrix_value
	key_matrix_gradient = (attention_score_gradient.transpose(-2, -1) @ query_matrix) / math.sqrt(embedding_size)

	#$ dLoss / dCombined_input_value
	combined_input_embeddings_gradient = (query_matrix_gradient @ query_weight_table) + (key_matrix_gradient @ key_weight_table) + (value_matrix_gradient @ value_weight_table)

	#$ dLoss / dValue_weight_table_value
	value_matrix_gradient_flat = value_matrix_gradient.reshape(-1, embedding_size)
	combined_input_embeddings_flat = combined_input_embeddings.reshape(-1, embedding_size)
	value_weight_table_gradient = value_matrix_gradient_flat.transpose(-2, -1) @ combined_input_embeddings_flat
	
	#$ dLoss / dKey_weight_table_value
	key_matrix_gradient_flat = key_matrix_gradient.reshape(-1, embedding_size)
	key_weight_table_gradient = key_matrix_gradient_flat.transpose(-2, -1) @ combined_input_embeddings_flat

	#$ dLoss / dQuery_weight_table_value
	query_matrix_gradient_flat = query_matrix_gradient.reshape(-1, embedding_size)
	query_weight_table_gradient = query_matrix_gradient_flat.transpose(-2, -1) @ combined_input_embeddings_flat

	#$ dLoss / dPositional_embedding_table_value
	input_position_embeddings_gradient = combined_input_embeddings_gradient

	#$ dLoss / dToken_embedding_table_value
	token_numbers = number_of_ids_indices
	flattened_input_token_ids = input_token_ids.view(-1)
	token_matches = (token_numbers[:, None] == flattened_input_token_ids[None, :]).float()

	input_token_embeddings_gradient = token_matches @ combined_input_embeddings_gradient.view(-1, embedding_size)

	#~ Gradient debugging
	if(0):
		print("\n========== GRADIENT DEBUG ==========")
		print(
			"ffn_downscale_matrix_gradient:",
			"shape =", tuple(ffn_downscale_matrix_gradient.shape),
			"min =", ffn_downscale_matrix_gradient.min().item(),
			"max =", ffn_downscale_matrix_gradient.max().item(),
			"mean =", ffn_downscale_matrix_gradient.mean().item(),
			"norm =", ffn_downscale_matrix_gradient.norm().item(),
		)

		print(
			"ffn_activation_matrix_gradient:",
			"shape =", tuple(ffn_activation_matrix_gradient.shape),
			"min =", ffn_activation_matrix_gradient.min().item(),
			"max =", ffn_activation_matrix_gradient.max().item(),
			"mean =", ffn_activation_matrix_gradient.mean().item(),
			"norm =", ffn_activation_matrix_gradient.norm().item(),
		)

		print(
			"ffn_down_weight_table_gradient:",
			"shape =", tuple(ffn_down_weight_table_gradient.shape),
			"min =", ffn_down_weight_table_gradient.min().item(),
			"max =", ffn_down_weight_table_gradient.max().item(),
			"mean =", ffn_down_weight_table_gradient.mean().item(),
			"norm =", ffn_down_weight_table_gradient.norm().item(),
		)

		print(
			"ffn_upscale_matrix_gradient:",
			"shape =", tuple(ffn_upscale_matrix_gradient.shape),
			"min =", ffn_upscale_matrix_gradient.min().item(),
			"max =", ffn_upscale_matrix_gradient.max().item(),
			"mean =", ffn_upscale_matrix_gradient.mean().item(),
			"norm =", ffn_upscale_matrix_gradient.norm().item(),
		)

		print(
			"ffn_up_weight_table_gradient:",
			"shape =", tuple(ffn_up_weight_table_gradient.shape),
			"min =", ffn_up_weight_table_gradient.min().item(),
			"max =", ffn_up_weight_table_gradient.max().item(),
			"mean =", ffn_up_weight_table_gradient.mean().item(),
			"norm =", ffn_up_weight_table_gradient.norm().item(),
		)

		print(
			"attention_output_gradient:",
			"shape =", tuple(attention_output_gradient.shape),
			"min =", attention_output_gradient.min().item(),
			"max =", attention_output_gradient.max().item(),
			"mean =", attention_output_gradient.mean().item(),
			"norm =", attention_output_gradient.norm().item(),
		)

		print(
			"output_weight_table_gradient:",
			"shape =", tuple(output_weight_table_gradient.shape),
			"min =", output_weight_table_gradient.min().item(),
			"max =", output_weight_table_gradient.max().item(),
			"mean =", output_weight_table_gradient.mean().item(),
			"norm =", output_weight_table_gradient.norm().item(),
		)

		print(
			"value_matrix_gradient:",
			"shape =", tuple(value_matrix_gradient.shape),
			"min =", value_matrix_gradient.min().item(),
			"max =", value_matrix_gradient.max().item(),
			"mean =", value_matrix_gradient.mean().item(),
			"norm =", value_matrix_gradient.norm().item(),
		)

		print(
			"attention_weight_gradient:",
			"shape =", tuple(attention_weight_gradient.shape),
			"min =", attention_weight_gradient.min().item(),
			"max =", attention_weight_gradient.max().item(),
			"mean =", attention_weight_gradient.mean().item(),
			"norm =", attention_weight_gradient.norm().item(),
		)

		print(
			"attention_score_gradient:",
			"shape =", tuple(attention_score_gradient.shape),
			"min =", attention_score_gradient.min().item(),
			"max =", attention_score_gradient.max().item(),
			"mean =", attention_score_gradient.mean().item(),
			"norm =", attention_score_gradient.norm().item(),
		)

		print(
			"query_matrix_gradient:",
			"shape =", tuple(query_matrix_gradient.shape),
			"min =", query_matrix_gradient.min().item(),
			"max =", query_matrix_gradient.max().item(),
			"mean =", query_matrix_gradient.mean().item(),
			"norm =", query_matrix_gradient.norm().item(),
		)

		print(
			"key_matrix_gradient:",
			"shape =", tuple(key_matrix_gradient.shape),
			"min =", key_matrix_gradient.min().item(),
			"max =", key_matrix_gradient.max().item(),
			"mean =", key_matrix_gradient.mean().item(),
			"norm =", key_matrix_gradient.norm().item(),
		)

		print(
			"value_weight_table_gradient:",
			"shape =", tuple(value_weight_table_gradient.shape),
			"min =", value_weight_table_gradient.min().item(),
			"max =", value_weight_table_gradient.max().item(),
			"mean =", value_weight_table_gradient.mean().item(),
			"norm =", value_weight_table_gradient.norm().item(),
		)

		print(
			"key_weight_table_gradient:",
			"shape =", tuple(key_weight_table_gradient.shape),
			"min =", key_weight_table_gradient.min().item(),
			"max =", key_weight_table_gradient.max().item(),
			"mean =", key_weight_table_gradient.mean().item(),
			"norm =", key_weight_table_gradient.norm().item(),
		)

		print(
			"query_weight_table_gradient:",
			"shape =", tuple(query_weight_table_gradient.shape),
			"min =", query_weight_table_gradient.min().item(),
			"max =", query_weight_table_gradient.max().item(),
			"mean =", query_weight_table_gradient.mean().item(),
			"norm =", query_weight_table_gradient.norm().item(),
		)

		print(
			"combined_input_embeddings_gradient:",
			"shape =", tuple(combined_input_embeddings_gradient.shape),
			"min =", combined_input_embeddings_gradient.min().item(),
			"max =", combined_input_embeddings_gradient.max().item(),
			"mean =", combined_input_embeddings_gradient.mean().item(),
			"norm =", combined_input_embeddings_gradient.norm().item(),
		)

		print(
			"input_position_embeddings_gradient:",
			"shape =", tuple(input_position_embeddings_gradient.shape),
			"min =", input_position_embeddings_gradient.min().item(),
			"max =", input_position_embeddings_gradient.max().item(),
			"mean =", input_position_embeddings_gradient.mean().item(),
			"norm =", input_position_embeddings_gradient.norm().item(),
		)

		print(
			"input_token_embeddings_gradient:",
			"shape =", tuple(input_token_embeddings_gradient.shape),
			"min =", input_token_embeddings_gradient.min().item(),
			"max =", input_token_embeddings_gradient.max().item(),
			"mean =", input_token_embeddings_gradient.mean().item(),
			"norm =", input_token_embeddings_gradient.norm().item(),
		)

		print(
			"logits_gradient:",
			"shape =", tuple(logits_gradient.shape),
			"min =", logits_gradient.min().item(),
			"max =", logits_gradient.max().item(),
			"mean =", logits_gradient.mean().item(),
			"norm =", logits_gradient.norm().item(),
		)

		print("====================================\n")


	# + Learning
	#& Output
	output_weight_table -= output_weight_table_gradient * learning_rate

	#& FFN weight tables
	ffn_up_weight_table -= ffn_up_weight_table_gradient * learning_rate
	ffn_down_weight_table -= ffn_down_weight_table_gradient * learning_rate

	#& QKV weight tables
	value_weight_table -= value_weight_table_gradient * learning_rate
	key_weight_table -= key_weight_table_gradient * learning_rate
	query_weight_table -= query_weight_table_gradient * learning_rate

	#& Token embedding tables
	positional_embedding_table -= input_position_embeddings_gradient.sum(dim=0) * learning_rate
	token_embedding_table -= input_token_embeddings_gradient * learning_rate

def get_predicted_id_for_position(position, probability_matrix):
	number_of_ids = len(probability_matrix[0])
	predicted_id = 0
	current_max_probability = 0
	for i in range(number_of_ids):
		if current_max_probability < probability_matrix[position][i]: 
			predicted_id = i
			current_max_probability = probability_matrix[position][i]

	return predicted_id

#! Settings                                                 
torch.manual_seed(12354)
embedding_size = 384
CONTEXT_LENGTH = 128
number_of_training_windows = len(token_ids) - CONTEXT_LENGTH
learning_rate = 0.001
learning_user_scale = 1
ffn_hidden_size = embedding_size * 4
batch_size = 128

example_start_positions = torch.arange(CONTEXT_LENGTH, CONTEXT_LENGTH * (batch_size + 1), step=CONTEXT_LENGTH, device=device)[:, None]

#& Embedding and weight table initialization + mask + arange indices
if (1):
	#% Shape: vocabulary_size × embedding_size. 
	#* This is the map for all characters in the vocabulary to their respective vector, initialized random at first and changes later to learn relationships.
	scale = 1 / math.sqrt(embedding_size)
	ffn_up_scale = 1 / math.sqrt(embedding_size)
	ffn_down_scale = 1 / math.sqrt(ffn_hidden_size)
	token_embedding_table = torch.rand(number_of_ids, embedding_size) * (2 * scale) - scale

	#% Shape: context_length x embedding_size
	#* This gives each possible position in the context window its own learnable vector, so the model can distinguish position 0 from position 1, position 2, etc.
	positional_embedding_table = torch.rand(CONTEXT_LENGTH, embedding_size) * (2 * scale) - scale

	#% Shape: embedding_size x embedding_size
	#* Turns raw input vectors into queries to ask what it needs
	query_weight_table = torch.rand(embedding_size, embedding_size) * (2 * scale) - scale

	#% Shape: embedding_size x embedding_size
	#* Turns raw input vectors into keys to advertise what a query might need
	key_weight_table = torch.rand(embedding_size, embedding_size) * (2 * scale) - scale

	#% Shape: embedding_size x embedding_size
	#* Turns raw input vectors into values to convey useful information about the next position
	value_weight_table = torch.rand(embedding_size, embedding_size) * (2 * scale) - scale

	#% Shape: number_of_ids x embedding_size
	#* Determines/asks how much an attention_output vector supports a character being next
	output_weight_table = torch.rand(number_of_ids, embedding_size) * (2 * scale) - scale

	#% Shape: embedding_size x embedding_size * 4 (or ffn_hidden_size)
	ffn_up_weight_table = torch.rand(embedding_size, ffn_hidden_size) * (2 * ffn_up_scale) - ffn_up_scale
	#% Shape: embedding_size * 4 (or ffn_hidden_size) x embedding_size 
	ffn_down_weight_table = torch.rand(ffn_hidden_size, embedding_size) * (2 * ffn_down_scale) - ffn_down_scale

	#% Shape: context_length x context_length
	#* Shows what positions a given position is allowed to look at
	future_mask = torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1).bool()

	batch_indices = torch.arange(batch_size, device=device, dtype=torch.long)[:, None]
	number_of_ids_indices = torch.arange(number_of_ids, device=device, dtype=torch.long)
	context_indices = torch.arange(CONTEXT_LENGTH, device=device, dtype=torch.long)[None, :]

	

	#$ Moving to GPU
	token_embedding_table = token_embedding_table.to(device)
	positional_embedding_table = positional_embedding_table.to(device)

	query_weight_table = query_weight_table.to(device)
	key_weight_table = key_weight_table.to(device)
	value_weight_table = value_weight_table.to(device)
	output_weight_table = output_weight_table.to(device)

	ffn_up_weight_table = ffn_up_weight_table.to(device)
	ffn_down_weight_table = ffn_down_weight_table.to(device)


	future_mask = future_mask.to(device)

#& Running forward pass
forward_pass_results = run_forward_pass(example_start_positions, token_ids)

#& Assigning each matrix and vector their result from forward pass
if (1):
	#@ Turning characters into ids
	input_token_ids = forward_pass_results[0]
	target_token_ids = forward_pass_results[1]

	#@ Embedding character ids into vectors of embedding_size and combining input + positional data
	input_token_embeddings = forward_pass_results[2]
	input_position_embeddings = forward_pass_results[3]
	combined_input_embeddings = forward_pass_results[4]

	#@ Using combined_input_embeddings + respective matrix values to figure out what it should ask, advertise, and give
	query_matrix = forward_pass_results[5]
	key_matrix = forward_pass_results[6]
	value_matrix = forward_pass_results[7]

	#@ Attention results using QKV
	attention_score_matrix = forward_pass_results[8]
	attention_weight_matrix = forward_pass_results[9]
	attention_output_matrix = forward_pass_results[10]

	#@
	logits_matrix = forward_pass_results[11]
	probability_matrix = forward_pass_results[12]


	average_loss = forward_pass_results[13]

# # Test the loaded model

print("Loss:", forward_pass_results[13])

iteration = 0
loss_window_average = 0.0
window_loss_size = 1000
checkpoint_interval = 15000
learning_rate_patience = 50
last_improvement_window = 0
number_of_validation_batches = 8
window_start_time = time.perf_counter()
total_start_time = time.perf_counter()
fixed_validation_positions = torch.randint(high = len(validation_token_ids) - CONTEXT_LENGTH - 1, size = (number_of_validation_batches, batch_size, 1), dtype = torch.long).to(device)


CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_path = CHECKPOINT_DIR / "best_model.pt"

if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
	iteration, current_least_loss = load_model(checkpoint_path)
else:
	print("No valid checkpoint found. Starting training from scratch.")

	iteration = 0
	current_least_loss = float("inf")

test_positions = torch.randint(
    low=0,
    high=len(validation_token_ids) - CONTEXT_LENGTH,
    size=(256,),
    device=device,
    dtype=torch.long,
)

test_results = run_forward_pass(
    test_positions[:, None],
    validation_token_ids,
)

print(
    "Checkpoint validation loss immediately after loading:",
    test_results[13].item(),
)

## Test batch size and learning-rate combinations.
if(0):
	# ============================================================
	# Benchmark configuration
	# ============================================================

	batch_sizes_to_test = [64, 128, 256, 512]

	base_learning_rate = learning_rate

	learning_rate_multipliers = {
		64:  [0.25, 0.5, 1.0, 2.0],
		128: [0.5, 1.0, 2.0],
		256: [1.0, 2.0, 3.0],
		512: [1.0, 2.0, 4.0],
	}

	benchmark_seconds = 60.0 * 4
	warmup_steps = 10

	validation_examples = 2048
	validation_batch_size = 256

	parameter_names = [
		"token_embedding_table",
		"positional_embedding_table",
		"query_weight_table",
		"key_weight_table",
		"value_weight_table",
		"output_weight_table",
		"ffn_up_weight_table",
		"ffn_down_weight_table",
	]


	# ============================================================
	# Parameter management
	# ============================================================

	def snapshot_parameters():
		return {
			name: globals()[name].clone()
			for name in parameter_names
		}


	def restore_parameters(snapshot):
		global token_embedding_table
		global positional_embedding_table
		global query_weight_table
		global key_weight_table
		global value_weight_table
		global output_weight_table
		global ffn_up_weight_table
		global ffn_down_weight_table

		token_embedding_table = snapshot[
			"token_embedding_table"
		].clone()

		positional_embedding_table = snapshot[
			"positional_embedding_table"
		].clone()

		query_weight_table = snapshot[
			"query_weight_table"
		].clone()

		key_weight_table = snapshot[
			"key_weight_table"
		].clone()

		value_weight_table = snapshot[
			"value_weight_table"
		].clone()

		output_weight_table = snapshot[
			"output_weight_table"
		].clone()

		ffn_up_weight_table = snapshot[
			"ffn_up_weight_table"
		].clone()

		ffn_down_weight_table = snapshot[
			"ffn_down_weight_table"
		].clone()


	# ============================================================
	# Batch-dependent globals
	# ============================================================

	def configure_batch_size(new_batch_size):
		global batch_size
		global batch_indices
		global context_indices
		global number_of_ids_indices

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

		number_of_ids_indices = torch.arange(
			number_of_ids,
			device=device,
			dtype=torch.long,
		)


	# ============================================================
	# Synchronization
	# ============================================================

	def synchronize_device():
		if device.type == "cuda":
			torch.cuda.synchronize()

		elif device.type == "privateuseone":
			# DirectML does not expose torch.cuda.synchronize().
			#
			# Reading one device scalar forces all preceding work needed
			# for that scalar to finish. This variable is set at the end
			# of each measured training section.
			pass


	# ============================================================
	# Validation
	# ============================================================

	def calculate_validation_loss(validation_positions):
		total_weighted_loss = 0.0
		total_examples = 0

		for start_index in range(
			0,
			validation_positions.shape[0],
			validation_batch_size,
		):
			current_positions = validation_positions[
				start_index:
				start_index + validation_batch_size
			]

			forward_results = run_forward_pass(
				current_positions[:, None],
				validation_token_ids,
			)

			current_batch_size = current_positions.shape[0]

			total_weighted_loss += (
				forward_results[13].item()
				* current_batch_size
			)

			total_examples += current_batch_size

		return total_weighted_loss / total_examples


	# ============================================================
	# Fixed benchmark data
	# ============================================================

	initial_parameters = snapshot_parameters()

	number_of_validation_windows = (
		len(validation_token_ids) - CONTEXT_LENGTH
	)

	validation_positions = torch.randint(
		low=0,
		high=number_of_validation_windows,
		size=(validation_examples,),
		device=device,
		dtype=torch.long,
	)

	# Create a long, fixed position stream.
	#
	# Every experiment reads from this same stream. Different batch sizes
	# group examples differently, but they do not receive unrelated random
	# samples.
	maximum_training_examples = 20_000_000

	fixed_training_positions = torch.randint(
		low=0,
		high=number_of_training_windows,
		size=(maximum_training_examples,),
		device=device,
		dtype=torch.long,
	)


	# ============================================================
	# Benchmark
	# ============================================================

	results = []

	for test_batch_size in batch_sizes_to_test:
		for multiplier in learning_rate_multipliers[
			test_batch_size
		]:
			test_learning_rate = (
				base_learning_rate * multiplier
			)

			print(
				"\n"
				f"Testing batch={test_batch_size}, "
				f"learning_rate={test_learning_rate:.8g}"
			)

			# ----------------------------------------------------
			# Warmup
			# ----------------------------------------------------

			restore_parameters(initial_parameters)
			configure_batch_size(test_batch_size)

			learning_rate = test_learning_rate

			warmup_offset = 0
			last_warmup_loss = None

			for _ in range(warmup_steps):
				positions = fixed_training_positions[
					warmup_offset:
					warmup_offset + test_batch_size
				]

				warmup_offset += test_batch_size

				forward_results = run_forward_pass(
					positions[:, None],
					token_ids,
				)

				run_backpropogation(forward_results)

				last_warmup_loss = forward_results[13]

			# Force warmup operations to finish.
			last_warmup_loss.item()

			# Discard all learning performed during warmup.
			restore_parameters(initial_parameters)
			configure_batch_size(test_batch_size)

			learning_rate = test_learning_rate

			# ----------------------------------------------------
			# Initial validation
			# ----------------------------------------------------

			initial_validation_loss = (
				calculate_validation_loss(
					validation_positions
				)
			)

			# ----------------------------------------------------
			# Timed training
			# ----------------------------------------------------

			training_offset = 0
			measured_steps = 0
			last_training_loss = None

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
						"fixed_training_positions is too short "
						"for this benchmark."
					)

				training_offset += test_batch_size

				forward_results = run_forward_pass(
					positions[:, None],
					token_ids,
				)

				run_backpropogation(forward_results)

				last_training_loss = forward_results[13]
				measured_steps += 1

				# Checking the Python clock does not synchronize the GPU.
				# Check less frequently to avoid unnecessary CPU overhead.
				if measured_steps % 10 == 0:
					elapsed_submission_time = (
						time.perf_counter() - start_time
					)

					if (
						elapsed_submission_time
						>= benchmark_seconds
					):
						break

			# Force all queued training operations to complete before
			# stopping the benchmark timer.
			final_training_loss = last_training_loss.item()

			benchmark_time = (
				time.perf_counter() - start_time
			)

			# ----------------------------------------------------
			# Final validation
			# ----------------------------------------------------

			final_validation_loss = (
				calculate_validation_loss(
					validation_positions
				)
			)

			# ----------------------------------------------------
			# Metrics
			# ----------------------------------------------------

			measured_examples = (
				measured_steps * test_batch_size
			)

			measured_characters = (
				measured_examples * CONTEXT_LENGTH
			)

			loss_drop = (
				initial_validation_loss
				- final_validation_loss
			)

			chars_per_second = (
				measured_characters
				/ benchmark_time
			)

			examples_per_second = (
				measured_examples
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

			loss_drop_per_second = (
				loss_drop
				/ benchmark_time
			)

			result = {
				"batch_size": test_batch_size,
				"learning_rate": test_learning_rate,
				"learning_rate_multiplier": multiplier,
				"initial_validation_loss": (
					initial_validation_loss
				),
				"final_validation_loss": (
					final_validation_loss
				),
				"loss_drop": loss_drop,
				"loss_drop_per_second": (
					loss_drop_per_second
				),
				"chars_per_second": chars_per_second,
				"examples_per_second": (
					examples_per_second
				),
				"steps_per_second": steps_per_second,
				"time_per_step_ms": time_per_step_ms,
				"measured_steps": measured_steps,
				"measured_examples": measured_examples,
				"benchmark_time": benchmark_time,
				"final_training_loss": (
					final_training_loss
				),
			}

			results.append(result)

			print(
				f"Initial validation loss: "
				f"{initial_validation_loss:.6f}"
			)

			print(
				f"Final validation loss:   "
				f"{final_validation_loss:.6f}"
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
				f"{chars_per_second:,.2f}"
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
				f"Loss drop/sec:           "
				f"{loss_drop_per_second:.8f}"
			)


	# ============================================================
	# Restore original model
	# ============================================================

	restore_parameters(initial_parameters)


	# ============================================================
	# Sort and display results
	# ============================================================

	results.sort(
		key=lambda result: result["loss_drop_per_second"],
		reverse=True,
	)

	print("\n" + "=" * 157)

	print(
		f"{'Rank':<7}"
		f"{'Batch':<9}"
		f"{'LR':<14}"
		f"{'Steps':<10}"
		f"{'Initial Val':<15}"
		f"{'Final Val':<15}"
		f"{'Loss Drop':<14}"
		f"{'Chars/Sec':<16}"
		f"{'Step ms':<12}"
		f"{'Loss Drop/Sec':<18}"
	)

	print("=" * 157)

	for rank, result in enumerate(results, start=1):
		print(
			f"{rank:<7}"
			f"{result['batch_size']:<9}"
			f"{result['learning_rate']:<14.6g}"
			f"{result['measured_steps']:<10}"
			f"{result['initial_validation_loss']:<15.6f}"
			f"{result['final_validation_loss']:<15.6f}"
			f"{result['loss_drop']:<14.6f}"
			f"{result['chars_per_second']:<16,.2f}"
			f"{result['time_per_step_ms']:<12.2f}"
			f"{result['loss_drop_per_second']:<18.8f}"
		)

	print("=" * 157)


#! Training loop                 
while True:
	iteration += 1

	start_positions = torch.randint(high = number_of_training_windows - 1, size = (batch_size,), device=device, dtype = torch.long)[:, None]
	forward_pass_results = run_forward_pass(start_positions, token_ids)

	# This is the loss before the current parameter update.
	window_loss = forward_pass_results[13]

	run_backpropogation(forward_pass_results)

	loss_window_average += window_loss

	if iteration % window_loss_size == 0:
		last_improvement_window += 1

		average_loss = (loss_window_average / window_loss_size).item()

		elapsed_hours = (
			time.perf_counter() - total_start_time
		) / 3600
			
		window_elapsed_time = (
			time.perf_counter() - window_start_time
		)

		total_elapsed_time = (
			time.perf_counter() - total_start_time
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
			f"elapsed: {total_elapsed_time / 3600:.2f} hours"
		)

		if iteration % checkpoint_interval == 0:
			save_model(
				CHECKPOINT_DIR / f"checkpoint_{iteration}_{CONTEXT_LENGTH}_{embedding_size}.pt",
				iteration=iteration,
				best_loss=current_least_loss,
			)

		# if(current_least_loss < 2.465):
		# 	pass
		# elif (current_least_loss < 2.48):
		# 	learning_rate = 0.0012
		# elif (current_least_loss < 2.5):
		# 	learning_rate = 0.0025
		# elif (current_least_loss < 2.55):
		# 	learning_rate = 0.0045
		# elif (current_least_loss < 2.6):
		# 	learning_rate = 0.007
		# elif (current_least_loss < 2.7):
		# 	learning_rate = 0.010
		# elif (current_least_loss < 2.8):
		# 	learning_rate = 0.015
		# elif (current_least_loss < 3):
		# 	learning_rate = 0.03
		# elif (current_least_loss < 4):
		# 	learning_rate = 0.05
		# else:
		# 	learning_rate = 0.1
		# learning_rate *= learning_user_scale
	
		validation_average_loss = 0
		for validation_start_positions in fixed_validation_positions:
			validation_forward_pass_results = run_forward_pass(validation_start_positions, validation_token_ids)

			validation_average_loss += validation_forward_pass_results[13]

		validation_average_loss /= number_of_validation_batches
		validation_average_loss = validation_average_loss.item()
		print(f"Validation loss at iteration {iteration}: {(validation_average_loss)}")

		if validation_average_loss < current_least_loss:
			current_least_loss = validation_average_loss
			last_improvement_window = 0

			save_model(
				CHECKPOINT_DIR / "best_model.pt",
				iteration=iteration,
				best_loss=current_least_loss,
			)

			print(
				f"New best model saved with loss: "
				f"{current_least_loss:.6f}"
			)
		



		# Reduce the learning rate if validation has not improved.
		if (
			last_improvement_window >= learning_rate_patience):
			print(
				f"No improvement for {learning_rate_patience} loss windows. Setting learning rate to {round(learning_rate * 0.9, 6)}. ")

			learning_rate *= 0.9

			# Start a new patience period after rollback.
			last_improvement_window = 0
		
		loss_window_average = 0.0
		window_start_time = time.perf_counter()