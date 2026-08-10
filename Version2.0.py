#V2: Optimized matrix multiplication using pytorch tensors and moved to GPU. Still attention only network, no layers.

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
device = torch_directml.device()

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
	device = torch_directml.device()

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

def get_input_token_ids(position):
	return token_ids[position : position + CONTEXT_LENGTH]

def get_target_token_ids(position):
	return token_ids[position + 1: position + CONTEXT_LENGTH + 1]

def run_forward_pass(start_position, source_token_ids):
	# Look up one embedding vector for each input token ID.
	input_token_ids = source_token_ids[start_position : start_position + CONTEXT_LENGTH]
	target_token_ids = source_token_ids[start_position + 1: start_position + CONTEXT_LENGTH + 1]

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
	attention_score_matrix = attention_score_matrix.masked_fill(future_mask, -1e9)

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
	probability_matrix = torch.softmax(logits_matrix, dim=-1)

	average_loss = F.cross_entropy(
		logits_matrix,
		target_token_ids
	)
	# 0: input_token_ids, 1: target_token_ids, 2: input_token_embeddings
	# 3: input_position_embeddings, 4: combined_input_embeddings, 5: query_matrix
	# 6: key_matrix, 7: value_matrix, 8: attention_score_matrix
	# 9: attention_weight_matrix, 10: attention_output_matrix, 11: logits_matrix
	# 12: probability_matrix, 13: average loss

	return [input_token_ids, target_token_ids, input_token_embeddings,
			input_position_embeddings, combined_input_embeddings, query_matrix,
			key_matrix, value_matrix, attention_score_matrix,
			attention_weight_matrix, attention_output_matrix, logits_matrix,
			probability_matrix, average_loss]

def run_backpropogation(forward_pass_results):
	global token_embedding_table
	global positional_embedding_table
	global query_weight_table
	global key_weight_table
	global value_weight_table
	global output_weight_table

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

	logits_gradient = probability_matrix.clone()

	row_indices = torch.arange(CONTEXT_LENGTH, device=device)

	#* For each row in row_indices, select the column in that row index from target_token_ids, and only those are subtracted by 1
	logits_gradient[row_indices, target_token_ids] -= 1
	logits_gradient /= CONTEXT_LENGTH

	#$ dLoss / dAttention_output_value
	attention_output_gradient = logits_gradient @ output_weight_table

	#$ dLoss / dOutput_weight_values
	output_weight_table_gradient = logits_gradient.T @ attention_output_matrix
	output_weight_table -= output_weight_table_gradient * learning_rate

	#% Shape: context_length x embedding_size
	#$ dLoss / dValue_matrix_value
	value_matrix_gradient = attention_weight_matrix.T @ attention_output_gradient

	#% Shape: context_length x context_length
	#$ dLoss / dAttention_weight_value
	attention_weight_gradient = attention_output_gradient @ value_matrix.T

	#$ dLoss / dAttention_score_value
	weighted_attention_blame = attention_weight_matrix * attention_weight_gradient
	weighted_blame_per_row = weighted_attention_blame.sum(dim=1, keepdim=True)
	blame_difference = attention_weight_gradient - weighted_blame_per_row
	attention_score_gradient = blame_difference * attention_weight_matrix

	#$ dLoss / dQuery_matrix_value
	query_matrix_gradient = (attention_score_gradient @ key_matrix) / math.sqrt(embedding_size)
	
	#$ dLoss / dKey_matrix_value
	key_matrix_gradient = (attention_score_gradient.T @ query_matrix) / math.sqrt(embedding_size)

	#$ dLoss / dCombined_input_value
	combined_input_embeddings_gradient = (query_matrix_gradient @ query_weight_table) + (key_matrix_gradient @ key_weight_table) + (value_matrix_gradient @ value_weight_table)

	#$ dLoss / dValue_weight_table_value
	value_weight_table_gradient = value_matrix_gradient.T @ combined_input_embeddings
	value_weight_table -= value_weight_table_gradient * learning_rate
	
	#$ dLoss / dKey_weight_table_value
	key_weight_table_gradient = key_matrix_gradient.T @ combined_input_embeddings
	key_weight_table -= key_weight_table_gradient * learning_rate

	#$ dLoss / dQuery_weight_table_value
	query_weight_table_gradient = query_matrix_gradient.T @ combined_input_embeddings
	query_weight_table -= query_weight_table_gradient * learning_rate

	#$ dLoss / dPositional_embedding_table_value
	input_position_embeddings_gradient = combined_input_embeddings_gradient

	positional_embedding_table -= input_position_embeddings_gradient * learning_rate

	#$ dLoss / dToken_embedding_table_value
	token_numbers = torch.arange(number_of_ids, device=device)
	token_matches = (token_numbers[:, None] == input_token_ids[None, :]).float()

	input_token_embeddings_gradient = token_matches @ combined_input_embeddings_gradient

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

example_start_position = 128

#& Settings
#! 2 temporarily, otherwise 16
embedding_size = 2048
CONTEXT_LENGTH = 128
number_of_training_windows = len(token_ids) - CONTEXT_LENGTH
learning_rate = 0.05

#& Embedding and weight table initialization + mask
if (1):
	#% Shape: vocabulary_size × embedding_size. 
	#* This is the map for all characters in the vocabulary to their respective vector, initialized random at first and changes later to learn relationships.
	scale = 1 / math.sqrt(embedding_size)
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

	#% Shape: context_length x context_length
	#* Shows what positions a given position is allowed to look at
	future_mask = torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1).bool()

	#$ Moving to GPU
	token_embedding_table = token_embedding_table.to(device)
	positional_embedding_table = positional_embedding_table.to(device)

	query_weight_table = query_weight_table.to(device)
	key_weight_table = key_weight_table.to(device)
	value_weight_table = value_weight_table.to(device)
	output_weight_table = output_weight_table.to(device)

	future_mask = future_mask.to(device)

#& Running forward pass
forward_pass_results = run_forward_pass(example_start_position, token_ids)

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

probability_matrix = forward_pass_results[12]
target_token_ids = forward_pass_results[1]

print("Loss:", forward_pass_results[13])

iteration = 0
loss_window_average = 0.0
window_loss_size = 5000
checkpoint_interval = 50000
rollback_patience = 20
last_improvement_window = 0
validation_interval = 20000
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_path = CHECKPOINT_DIR / "best_model.pt"

if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
	iteration, current_least_loss = load_model(checkpoint_path)
else:
	print("No valid checkpoint found. Starting training from scratch.")

	iteration = 0
	current_least_loss = float("inf")

total_start_time = time.perf_counter()
window_start_time = time.perf_counter()

#! Graphing
is_graphing = False
if (is_graphing):
	LOSS_LOG_PATH = BASE_DIR / "loss_history.csv"

	loss_history = []

	# Load existing loss history if it exists
	if LOSS_LOG_PATH.exists():
		with open(LOSS_LOG_PATH, "r", newline="", encoding="utf-8") as file:
			reader = csv.DictReader(file)

			for row in reader:
				loss_history.append({
					"iteration": int(row["iteration"]),
					"elapsed_hours": float(row["elapsed_hours"]),
					"loss": float(
						row["loss"].strip().removeprefix("tensor(").removesuffix(")")
					),
				})

	# Create the graph
	plt.ion()

	figure, axis = plt.subplots(figsize=(10, 6))

	if loss_history:
		axis.plot(
			[item["elapsed_hours"] for item in loss_history],
			[item["loss"] for item in loss_history],
			label="Training loss",
		)

	axis.set_title("Training Loss Over Time")
	axis.set_xlabel("Elapsed training time (hours)")
	axis.set_ylabel("Average cross-entropy loss")
	axis.grid(True, alpha=0.3)
	axis.legend()

	plt.show(block=False)


while True:
	iteration += 1

	start_position = random.randint(
		0,
		number_of_training_windows - 1,
	)
	forward_pass_results = run_forward_pass(start_position, token_ids)

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
		#! Graphing'
		if (is_graphing):
			loss_history.append({
				"iteration": iteration,
				"elapsed_hours": elapsed_hours,
				"loss": average_loss,
			})

			csv_exists = LOSS_LOG_PATH.exists()

			with open(
				LOSS_LOG_PATH,
				"a",
				newline="",
				encoding="utf-8",
			) as file:
				writer = csv.writer(file)

				if not csv_exists:
					writer.writerow([
						"iteration",
						"elapsed_hours",
						"loss",
					])

				writer.writerow([
					iteration,
					elapsed_hours,
					average_loss,
				])

			axis.clear()

			axis.plot(
				[item["elapsed_hours"] for item in loss_history],
				[item["loss"] for item in loss_history],
				color="blue",
				label="Training loss",
			)

			axis.set_title("Training Loss Over Time")
			axis.set_xlabel("Elapsed training time (hours)")
			axis.set_ylabel("Average cross-entropy loss")
			axis.grid(True, alpha=0.3)
			axis.legend()

			figure.tight_layout()
			figure.canvas.draw()
			figure.canvas.flush_events()
			plt.pause(0.001)
			
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

		if average_loss < current_least_loss:
			current_least_loss = average_loss
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
			if(current_least_loss < 2.465):
				pass
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
			
			


		if (iteration % validation_interval == 0):
			validation_total_loss = 0
			for i in range(500):
				validation_start_position = random.randint(0, len(validation_token_ids) - CONTEXT_LENGTH - 1)

				validation_forward_pass_results = run_forward_pass(validation_start_position, validation_token_ids)

				validation_total_loss += validation_forward_pass_results[13]
			print(f"Validation loss at iteration {iteration}: {(validation_total_loss / 500).item()}")



		# Roll back if no progress was made for 20 loss windows.
		if (
			last_improvement_window >= rollback_patience and (CHECKPOINT_DIR / "best_model.pt").exists()):
			print(
				f"No improvement for {rollback_patience} loss windows. Setting learning rate to {round(learning_rate * 0.9, 4)}. "
				"Loading the previous best model."
			)
			_, loaded_best_loss = load_model(
				CHECKPOINT_DIR / "best_model.pt"
			)

			current_least_loss = loaded_best_loss
			learning_rate *= 0.9

			# Start a new patience period after rollback.
			last_improvement_window = 0
		
		loss_window_average = 0.0
		window_start_time = time.perf_counter()