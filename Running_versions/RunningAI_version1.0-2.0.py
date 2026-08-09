import torch
import torch_directml
import torch.nn.functional as F
import pandas as pd
import random
import math
import copy
import os
import copy
import time
from pathlib import Path
import matplotlib.pyplot as plt
import csv

UNKNOWN_TOKEN = "<unk>"
mask_value = -1e9

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
CHECKPOINT_DIR = BASE_DIR / "Saved_bests"

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
	device = torch.device("cpu")

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

	iteration = checkpoint.get("iteration", 0)
	best_loss = checkpoint.get("best_loss", float("inf"))

	print(f"Model loaded from: {path}")
	print(f"Resuming from iteration: {iteration}")
	print(f"Previous best loss: {best_loss:.6f}")

	return iteration, best_loss

def get_input_token_ids():
	return token_ids

def run_forward_pass():
	# Look up one embedding vector for each input token ID.
	input_token_ids = get_input_token_ids()

	L = len(input_token_ids)

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

	#% Shape: context_length x embedding_size
	#* The asking matrix based on combined_input_embeddings
	query_matrix = combined_input_embeddings @ query_weight_table.T

	#% Shape: context_length x embedding_size
	#* The advertising matrix based on combined_input_embeddings
	key_matrix = combined_input_embeddings @ key_weight_table.T

	#% Shape: context_length x embedding_size
	#* The conveying matrix based on combined_input_embeddings
	value_matrix = combined_input_embeddings @ value_weight_table.T
	
	#% Shape: context_length x context_length
	#* Stores the raw relevance scores between every current position and every position it is allowed to look at.
	attention_score_matrix = query_matrix @ key_matrix.T / math.sqrt(embedding_size)
	attention_score_matrix = attention_score_matrix.masked_fill(causal_mask, -1e9)

	#% Shape: context_length x context_length
	#* Stores how much each current position attends/cares to each previous/current position. Future positions receive weight 0.
	attention_weight_matrix = torch.softmax(attention_score_matrix, dim=-1)

	#% Shape: context_length x embedding_size
	#* Computes the necessary information from previous and current position for each position in context_length in a blended vector
	attention_output_matrix = attention_weight_matrix @ value_matrix

	#% Shape: context_length x number_of_ids
	#* Dot products attention context vectors with each vocab character's output weights,
	#* so if logits_matrix[10][20] is high, it means the attenuated output vector at position 10 supports the usage of character_id 20, so 'u' in this case
	logits_matrix = attention_output_matrix @ output_weight_table.T

	#% Shape: context_length x number_of_ids
	#* Softmaxes logits_matrix row by row, so each position gets softmaxed where the character ids are turned into probabilities
	scaled_logits = logits_matrix / temperature

	probability_matrix = torch.softmax(scaled_logits, dim=-1)

	# 0: input_token_ids, 1: 0 (placeholder), 2: input_token_embeddings
	# 3: input_position_embeddings, 4: combined_input_embeddings, 5: query_matrix
	# 6: key_matrix, 7: value_matrix, 8: attention_score_matrix
	# 9: attention_weight_matrix, 10: attention_output_matrix, 11: logits_matrix
	# 12: probability_matrix

	return [input_token_ids, 0, input_token_embeddings,
			input_position_embeddings, combined_input_embeddings, query_matrix,
			key_matrix, value_matrix, attention_score_matrix,
			attention_weight_matrix, attention_output_matrix, logits_matrix,
			probability_matrix]

def get_predicted_id_for_position(position, probability_matrix):
	number_of_ids = len(probability_matrix[0])
	# predicted_id = 0
	# current_max_probability = 0
	# for i in range(number_of_ids):
	# 	if current_max_probability < probability_matrix[position][i]: 
	# 		predicted_id = i
	# 		current_max_probability = probability_matrix[position][i]
	# 
	# return predicted_id

	random_probability = random.random()
	cumulative_probability = 0

	for i in range(number_of_ids):
		cumulative_probability += probability_matrix[position][i]
		if (random_probability < cumulative_probability):
					return i
	return number_of_ids - 1
		
def print_top_probabilities(position, probability_matrix, top_k=10):
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
embedding_size = 2048
CONTEXT_LENGTH = 128
learning_rate = 0.001
temperature = 0.5

#& Embedding and weight table initialization + mask
if (1):
	#% Shape: vocabulary_size × embedding_size. 
	#* This is the map for all characters in the vocabulary to their respective vector, initialized random at first and changes later to learn relationships.
	token_embedding_table = torch.rand(number_of_ids, embedding_size) * 2 - 1

	#% Shape: context_length x embedding_size
	#* This gives each possible position in the context window its own learnable vector, so the model can distinguish position 0 from position 1, position 2, etc.
	positional_embedding_table = torch.rand(CONTEXT_LENGTH, embedding_size) * 2 - 1

	#% Shape: embedding_size x embedding_size
	#* Turns raw input vectors into queries to ask what it needs
	query_weight_table = torch.rand(embedding_size, embedding_size) * 0.6 - 0.3

	#% Shape: embedding_size x embedding_size
	#* Turns raw input vectors into keys to advertise what a query might need
	key_weight_table = torch.rand(embedding_size, embedding_size) * 0.6 - 0.3

	#% Shape: embedding_size x embedding_size
	#* Turns raw input vectors into values to convey useful information about the next position
	value_weight_table = torch.rand(embedding_size, embedding_size) * 0.6 - 0.3

	#% Shape: number_of_ids x embedding_size
	#* Determines/asks how much an attention_output vector supports a character being next
	output_weight_table = torch.rand(number_of_ids, embedding_size) * 2 - 1

	#% Shape: context_length x context_length
	#* Shows what positions a given position is allowed to look at
	future_mask = torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1).bool()

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_path = CHECKPOINT_DIR / "128x2048.pt"

if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
	iteration, current_least_loss = load_model(checkpoint_path)

total_start_time = time.perf_counter()
window_start_time = time.perf_counter()

prompt = "The cat chased a mouse and chased after t"
token_ids = encode_text(prompt)

for iteration in range(20):

	forward_pass_results = run_forward_pass()

	# print_top_probabilities(len(prompt) - 1, forward_pass_results[12])

	prompt += id_to_character[get_predicted_id_for_position(len(prompt) - 1, forward_pass_results[12])]
	token_ids = encode_text(prompt)
	print(prompt)

	window_start_time = time.perf_counter()