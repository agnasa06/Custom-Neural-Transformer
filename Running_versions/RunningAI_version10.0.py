import torch
import torch.nn.functional as F
import pandas as pd
import random
import math
import os
import time
from pathlib import Path

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

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "Saved_bests"

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

def get_input_token_ids(start_positions, source_token_ids, L):
	offsets = torch.arange(L, device=device, dtype=torch.long)[None, :]
	return source_token_ids[start_positions + offsets]

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

def run_forward_attention(layer, input_embeddings, normalized_input_embeddings, causal_mask, L):
	#% Shape: 3 x context_length x embedding_size
	qkv_matrix = (normalized_input_embeddings @ fp16_qkv_weight_table[layer].transpose(-2, -1))

	query_matrix = qkv_matrix[...,0:embedding_size,]
	key_matrix = qkv_matrix[...,embedding_size:2 * embedding_size,]
	value_matrix = qkv_matrix[...,2 * embedding_size:3 * embedding_size,]

	query_matrix = query_matrix.reshape(batch_size, L, head_count, embedding_size // head_count).transpose(1, 2)
	key_matrix = key_matrix.reshape(batch_size, L, head_count, embedding_size // head_count).transpose(1, 2)
	value_matrix = value_matrix.reshape(batch_size, L, head_count, embedding_size // head_count).transpose(1, 2)
	
	#% Shape: context_length x context_length
	attention_score_matrix = query_matrix @ key_matrix.transpose(-2, -1) / math.sqrt(embedding_size // head_count)
	attention_score_matrix = attention_score_matrix.masked_fill(causal_mask, mask_value)
	attention_weight_matrix = torch.softmax(attention_score_matrix.float(), dim=-1).to(torch.float16)

	#% Shape: context_length x embedding_size
	attention_output_matrix = (attention_weight_matrix @ value_matrix).transpose(1, 2)
	attention_output_matrix = attention_output_matrix.reshape(batch_size, L, embedding_size)
	attention_output_matrix += input_embeddings

	# 0: query_matrix, 1: key_matrix, 2: value_matrix
	# 3: attention_weight_matrix, 4: attention_output_matrix

	return [query_matrix, key_matrix, value_matrix, attention_weight_matrix, attention_output_matrix]


def run_forward_pass(start_positions, source_token_ids, L):
	# Look up one embedding vector for each input token ID.
	input_token_ids = get_input_token_ids(start_positions, source_token_ids, L)
	
	causal_mask = future_mask[:L, :L]
	pos_embeds = positional_embedding_table[:L, :].to(torch.float16)

	# print(input_token_ids.shape)

	#% Shape: context_length × embedding_size
	#* This is converting each token (character) in the input of size CONTEXT_LENGTH (64) to their vector representation from the token_embedding_table
	input_token_embeddings = fp16_token_embedding_table[input_token_ids]

	#% Shape: context_length x embedding_size
	#* Retrieves the position vector for each position in this input window. For now, this matches the full positional_embedding_table and is mainly useful for debugging.
	input_position_embeddings = pos_embeds

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
		attention_weight_matrix[layer], attention_output_matrix[layer]) = run_forward_attention(layer, layer_state[layer], scaled_layer_state[layer], causal_mask, L)

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

	# average_loss = -log_probability_matrix[batch_indices,context_indices,target_token_ids,].mean()

	# 0: input_token_ids, 1: target_token_ids, 2: input_token_embeddings
	# 3: input_position_embeddings, 4: combined_input_embeddings, 5: query_matrix
	# 6: key_matrix, 7: value_matrix, 8: attention_score_matrix
	# 9: attention_weight_matrix, 10: attention_output_matrix, 11: logits_matrix
	# 12: probability_matrix, 13: average loss, 14: ffn_upscale_matrix
	# 15: layer_state, 16: normalized_layer_state, 17: normalized_ffn_input
	# 18: attention_inverse_root_mean_square, 19: ffn_inverse_root_mean_square
	# 20: scaled_layer_state, 21: scaled_ffn_input

	return [input_token_ids, 0, 0,
			0, combined_input_embeddings, query_matrix,
			key_matrix, value_matrix, 0,
			attention_weight_matrix, attention_output_matrix, 0,
			probability_matrix, None, ffn_upscale_matrix,
			layer_state, normalized_layer_state, normalized_ffn_input, 
			attention_inverse_root_mean_square, ffn_inverse_root_mean_square,
			scaled_layer_state, scaled_ffn_input]

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
embedding_size = 512
CONTEXT_LENGTH = 128
batch_size = 1
gradient_scale = 256
head_count = 4
top_k = 5
always_best = False
load_persistent_learning = True
ffn_hidden_multiplier = [4, 3, 3, 3, 3, 3, 3, 3]

number_of_layers = len(ffn_hidden_multiplier)

#& Embedding and weight table initialization + mask + arange indices
if (1):
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
	qkv_weight_table = torch.rand(number_of_layers, 3 * embedding_size, embedding_size,) * (2 * scale) - scale

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

	fp16_token_embedding_table = token_embedding_table.to(torch.float16)
	fp16_positional_embedding_table = positional_embedding_table.to(torch.float16)
	fp16_qkv_weight_table = qkv_weight_table.to(torch.float16)
	fp16_output_weight_table = output_weight_table.to(torch.float16)
	fp16_ffn_up_weight_table = [ffn_up_weight_table[i].to(torch.float16) for i in range(number_of_layers)]
	fp16_ffn_down_weight_table = [ffn_down_weight_table[i].to(torch.float16) for i in range(number_of_layers)]
	fp16_attention_gamma_table = attention_gamma_table.to(torch.float16)
	fp16_ffn_gamma_table = ffn_gamma_table.to(torch.float16)


CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_path = CHECKPOINT_DIR / "128x512_8lr_atNorm_4hd_mon_rms.pt"

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


prompt = "France's capital is "
token_ids = encode_text(prompt)

window_start_time = time.perf_counter()
current_start_time = time.perf_counter()

#! Generation loop          
for iteration in range(1000):

	start_position = torch.tensor([max(len(token_ids) - CONTEXT_LENGTH, 0)], device=device)[None, :]
	L = min(len(token_ids), CONTEXT_LENGTH)
	forward_pass_results = run_forward_pass(start_position, torch.tensor(token_ids, device=device), L)

	predicted_id = get_predicted_id_for_position(L - 1, forward_pass_results[12], top_k)
	token_ids.append(predicted_id)
	# print(decode_token_ids(token_ids))
	prompt += id_to_character[predicted_id]

	
	window_start_time = time.perf_counter()
print(decode_token_ids(token_ids))


