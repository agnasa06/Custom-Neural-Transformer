#V5: 3 layers 1 attention

import torch
import torch_directml
import torch.nn.functional as F
import pandas as pd
import random
import math
import os
import time
from pathlib import Path

UNKNOWN_TOKEN = "<unk>"
mask_value = -1e9
device = torch_directml.device()
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

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"


def load_model(path):
	"""
	Load a saved tensor checkpoint into the current global parameter tables.

	Returns:
		iteration, best_loss, persistent_learning_rate, previous_elapsed_seconds
	"""

	global token_embedding_table
	global positional_embedding_table
	global query_weight_table
	global key_weight_table
	global value_weight_table
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

	if checkpoint["query_weight_table"].shape != query_weight_table.shape:
		raise ValueError(
			"Cannot load checkpoint: query_weight_table shape differs.\n"
			f"Checkpoint: {checkpoint['query_weight_table'].shape}\n"
			f"Current: {query_weight_table.shape}"
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

def run_forward_attention(layer, input_embeddings, causal_mask):
	#% Shape: context_length x embedding_size
	query_matrix = input_embeddings @ query_weight_table[layer].transpose(-2, -1)
	key_matrix = input_embeddings @ key_weight_table[layer].transpose(-2, -1)
	value_matrix = input_embeddings @ value_weight_table[layer].transpose(-2, -1)
	
	#% Shape: context_length x context_length
	attention_score_matrix = query_matrix @ key_matrix.transpose(-2, -1) / math.sqrt(embedding_size)
	attention_score_matrix = attention_score_matrix.masked_fill(causal_mask, -1e9)
	attention_weight_matrix = torch.softmax(attention_score_matrix, dim=-1)

	#% Shape: context_length x embedding_size
	attention_output_matrix = attention_weight_matrix @ value_matrix
	attention_output_matrix += input_embeddings

	# 0: query_matrix, 1: key_matrix, 2: value_matrix
	# 3: attention_score_matrix, 4: attention_weight_matrix, 5: attention_output_matrix

	return [query_matrix, key_matrix, value_matrix,
			attention_score_matrix, attention_weight_matrix, attention_output_matrix]


def run_forward_pass(start_positions, source_token_ids, L):
	# Look up one embedding vector for each input token ID.
	input_token_ids = get_input_token_ids(start_positions, source_token_ids, L)

	causal_mask = future_mask[:L, :L]
	pos_embeds = positional_embedding_table[:L, :]

	#% Shape: context_length × embedding_size
	#* This is converting each token (character) in the input of size CONTEXT_LENGTH (64) to their vector representation from the token_embedding_table
	input_token_embeddings = token_embedding_table[input_token_ids]

	#% Shape: context_length x embedding_size
	#* Retrieves the position vector for each position in this input window. For now, this matches the full positional_embedding_table and is mainly useful for debugging.
	input_position_embeddings = pos_embeds

	#% Shape: context_length x embedding_size
	#* Adds the token vector and position vector at each position, producing the final input representation for the transformer.
	combined_input_embeddings = input_token_embeddings + input_position_embeddings

	query_matrix = torch.zeros((3, batch_size, L, embedding_size), device=device)
	key_matrix = torch.zeros((3, batch_size, L, embedding_size), device=device)
	value_matrix = torch.zeros((3, batch_size, L, embedding_size), device=device)
	attention_score_matrix = torch.zeros((3, batch_size, L, L), device=device)
	attention_weight_matrix = torch.zeros((3, batch_size, L, L), device=device)
	attention_output_matrix = torch.zeros((3, batch_size, L, embedding_size), device=device)
	

	#@ Layer 1
	query_matrix[0], key_matrix[0], value_matrix[0], attention_score_matrix[0], attention_weight_matrix[0], attention_output_matrix[0] = run_forward_attention(0, combined_input_embeddings, causal_mask)
	ffn_upscale_matrix_layer1 = attention_output_matrix[0] @ ffn_up_weight_table_layer1

	# ffn_upscale_matrix_layer1 = torch.relu(ffn_upscale_matrix_layer1)
	ffn_upscale_matrix_layer1.relu_()

	layer1_output = ffn_upscale_matrix_layer1 @ ffn_down_weight_table_layer1 + attention_output_matrix[0]

	#@ Layer 2
	query_matrix[1], key_matrix[1], value_matrix[1], attention_score_matrix[1], attention_weight_matrix[1], attention_output_matrix[1] = run_forward_attention(1, layer1_output, causal_mask)
	ffn_upscale_matrix_layer2 = attention_output_matrix[1] @ ffn_up_weight_table_layer2
	
	ffn_upscale_matrix_layer2.relu_()

	layer2_output = ffn_upscale_matrix_layer2 @ ffn_down_weight_table_layer2 + attention_output_matrix[1]

	#@ Layer 3
	query_matrix[2], key_matrix[2], value_matrix[2], attention_score_matrix[2], attention_weight_matrix[2], attention_output_matrix[2] = run_forward_attention(2, layer2_output, causal_mask)
	ffn_upscale_matrix_layer3 = attention_output_matrix[2] @ ffn_up_weight_table_layer3
		
	ffn_upscale_matrix_layer3.relu_()

	layer3_output = ffn_upscale_matrix_layer3 @ ffn_down_weight_table_layer3 + attention_output_matrix[2]

	#% Shape: context_length x number_of_ids
	#* Dot products attention context vectors with each vocab character's output weights,
	#* so if logits_matrix[10][20] is high, it means the attenuated output vector at position 10 supports the usage of character_id 20, so 'u' in this case
	logits_matrix = layer3_output @ output_weight_table.transpose(-2, -1)

	scaled_logits = logits_matrix / temperature

	#% Shape: context_length x number_of_ids
	#* Softmaxes logits_matrix row by row, so each position gets softmaxed where the character ids are turned into probabilities
	probability_matrix = torch.softmax(scaled_logits, dim=-1)

	average_loss = 0

	# 0: input_token_ids, 1: target_token_ids, 2: input_token_embeddings
	# 3: input_position_embeddings, 4: combined_input_embeddings, 5: query_matrix
	# 6: key_matrix, 7: value_matrix, 8: attention_score_matrix
	# 9: attention_weight_matrix, 10: attention_output_matrix, 11: logits_matrix
	# 12: probability_matrix, 13: average loss, 14: ffn_upscale_matrix_layer1
	# 15: layer1_output, 16: ffn_upscale_matrix_layer2, 17: layer2_output
	# 18: ffn_upscale_matrix_layer3, 19: layer3_output

	return [input_token_ids, 0, input_token_embeddings,
			input_position_embeddings, combined_input_embeddings, query_matrix,
			key_matrix, value_matrix, attention_score_matrix,
			attention_weight_matrix, attention_output_matrix, logits_matrix,
			probability_matrix, average_loss, ffn_upscale_matrix_layer1, 
			layer1_output, ffn_upscale_matrix_layer2, layer2_output,
			ffn_upscale_matrix_layer3, layer3_output]

def get_predicted_id_for_position(position, probability_matrix):
	probability_matrix = probability_matrix.reshape(-1, number_of_ids)
	# predicted_id = 0
	# current_max_probability = 0
	# for i in range(number_of_ids):
	# 	if current_max_probability < probability_matrix[position][i]: 
	# 		predicted_id = i
	# 		current_max_probability = probability_matrix[position][i]
	
	# return predicted_id

	random_probability = random.random()
	cumulative_probability = 0

	for i in range(number_of_ids):
		cumulative_probability += probability_matrix[position][i]
		if (random_probability < cumulative_probability):
			return i
	return number_of_ids - 1
		
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
embedding_size = 384
CONTEXT_LENGTH = 128
load_persistent_learning = True
ffn_hidden_size_layer1 = embedding_size * 4
ffn_hidden_size_layer2 = embedding_size * 2
ffn_hidden_size_layer3 = embedding_size * 2
batch_size = 1
temperature = 0.5

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

	#% Shape: embedding_size x embedding_size
	#* Turns raw input vectors into queries to ask what it needs
	query_weight_table = torch.rand(3, embedding_size, embedding_size) * (2 * scale) - scale

	#% Shape: embedding_size x embedding_size
	#* Turns raw input vectors into keys to advertise what a query might need
	key_weight_table = torch.rand(3, embedding_size, embedding_size) * (2 * scale) - scale

	#% Shape: embedding_size x embedding_size
	#* Turns raw input vectors into values to convey useful information about the next position
	value_weight_table = torch.rand(3, embedding_size, embedding_size) * (2 * scale) - scale

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
	context_indices = torch.arange(CONTEXT_LENGTH, device=device, dtype=torch.long)[None, :]

	

	#$ Moving to GPU
	token_embedding_table = token_embedding_table.to(device)
	positional_embedding_table = positional_embedding_table.to(device)

	query_weight_table = query_weight_table.to(device)
	key_weight_table = key_weight_table.to(device)
	value_weight_table = value_weight_table.to(device)
	output_weight_table = output_weight_table.to(device)

	ffn_up_weight_table_layer1 = ffn_up_weight_table_layer1.to(device)
	ffn_down_weight_table_layer1 = ffn_down_weight_table_layer1.to(device)
	ffn_up_weight_table_layer2 = ffn_up_weight_table_layer2.to(device)
	ffn_down_weight_table_layer2 = ffn_down_weight_table_layer2.to(device)
	ffn_up_weight_table_layer3 = ffn_up_weight_table_layer3.to(device)
	ffn_down_weight_table_layer3 = ffn_down_weight_table_layer3.to(device)


	future_mask = future_mask.to(device)


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


prompt = "France is a country whose capital is "
token_ids = encode_text(prompt)

window_start_time = time.perf_counter()
current_start_time = time.perf_counter()

#! Generation loop          
for iteration in range(10):

	start_position = torch.tensor([max(len(token_ids) - CONTEXT_LENGTH, 0)], device=device)[None, :]
	L = min(len(token_ids), CONTEXT_LENGTH)
	forward_pass_results = run_forward_pass(start_position, torch.tensor(token_ids, device=device), L)

	predicted_id = get_predicted_id_for_position(L - 1, forward_pass_results[12])
	token_ids.append(predicted_id)
	# print(decode_token_ids(token_ids))
	prompt += id_to_character[predicted_id]

	# This is the loss before the current parameter update.
	window_loss = forward_pass_results[13]

	window_start_time = time.perf_counter()
print(decode_token_ids(token_ids))


