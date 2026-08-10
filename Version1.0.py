import torch
import torch_directml
import pandas as pd
import random
import math
import copy
import os
import copy
import torch
import time


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

#^ 1. Load the compressed parquet table
df = pd.read_parquet("train-00000-of-00001.parquet")

#^ 2. Extract the 'text' column rows and combine them
dataText = "\n".join(df["text"].astype(str))

token_ids = encode_text(dataText)

def get_input_token_ids(position):
	if position + CONTEXT_LENGTH - 1 >= len(token_ids):
		return [-1]
	return [token_ids[i] for i in range(position, position + CONTEXT_LENGTH)]

def get_target_token_ids(position):
	if position + CONTEXT_LENGTH >= len(token_ids):
		return [-1]
	return [token_ids[i + 1] for i in range(position, position + CONTEXT_LENGTH)]

def dot_product(vector_a, vector_b):
	total = 0
	for i in range(len(vector_a)):
		total += vector_a[i] * vector_b[i]
	return total

def transform_vector(input_vector, weight_table):
	return [dot_product(input_vector, weight_table[i]) for i in range(embedding_size)]

def get_attention_scores_for_position(current_position, query_matrix, key_matrix):
	return [dot_product(query_matrix[current_position], key_matrix[other_position]) / math.sqrt(embedding_size) if causal_attention_mask[current_position][other_position] == 1 else mask_value for other_position in range(CONTEXT_LENGTH)]

def softmax(input_vector):
	# convert all scores to their exponential form
	exp_scores = [math.e ** s for s in input_vector]

	total = sum(exp_scores)
	return [e / total for e in exp_scores]


def get_attention_output_for_position(position, attention_weight_matrix, value_matrix):
	#* stores the weighted vectors using weights from position
	weighted_matrix = [[attention_weight_matrix[position][i] * val for val in value_matrix[i]] for i in range(CONTEXT_LENGTH)]

	attention_output = [0] * embedding_size
	for i in range(len(attention_output)):
		for j in range(len(weighted_matrix)):
			attention_output[i] += weighted_matrix[j][i]

	return attention_output
	# attention_output_for_position_10
	# =
	# weight for pos 0  × vector at pos 0
	# +
	# weight for pos 1  × vector at pos 1
	# +
	# ...
	# +
	# weight for pos 10 × vector at pos 10
	# +
	# 0.0 for future positions

def get_loss_for_position(position, probability_matrix, target_token_ids):
	return -math.log(probability_matrix[position][target_token_ids[position]])

def get_average_loss(probability_matrix, target_token_ids):
	return sum([get_loss_for_position(j, probability_matrix, target_token_ids) for j in range(CONTEXT_LENGTH)]) / CONTEXT_LENGTH

def run_forward_pass(start_position):
	# Look up one embedding vector for each input token ID.
	input_token_ids = get_input_token_ids(start_position)
	target_token_ids = get_target_token_ids(start_position)

	#% Shape: context_length × embedding_size
	#* This is converting each token (character) in the input of size CONTEXT_LENGTH (64) to their vector representation from the token_embedding_table
	input_token_embeddings = [token_embedding_table[input_token_ids[i]] for i in range(CONTEXT_LENGTH)]

	#% Shape: context_length x embedding_size
	#* Retrieves the position vector for each position in this input window. For now, this matches the full positional_embedding_table and is mainly useful for debugging.
	input_position_embeddings = [positional_embedding_table[i] for i in range(CONTEXT_LENGTH)]

	#% Shape: context_length x embedding_size
	#* Adds the token vector and position vector at each position, producing the final input representation for the transformer.
	combined_input_embeddings = [[input_token_embeddings[i][j] + input_position_embeddings[i][j] for j in range(embedding_size)] for i in range(CONTEXT_LENGTH)]

	#% Shape: context_length x embedding_size
	#* The asking matrix based on combined_input_embeddings
	query_matrix = [transform_vector(vector, query_weight_table) for vector in combined_input_embeddings]

	#% Shape: context_length x embedding_size
	#* The advertising matrix based on combined_input_embeddings
	key_matrix = [transform_vector(vector, key_weight_table) for vector in combined_input_embeddings]

	#% Shape: context_length x embedding_size
	#* The conveying matrix based on combined_input_embeddings
	value_matrix = [transform_vector(vector, value_weight_table) for vector in combined_input_embeddings]

	#% Shape: context_length x context_length
	#* Stores the raw relevance scores between every current position and every position it is allowed to look at.
	attention_score_matrix = [get_attention_scores_for_position(i, query_matrix, key_matrix) for i in range(CONTEXT_LENGTH)]

	#% Shape: context_length x context_length
	#* Stores how much each current position attends/cares to each previous/current position. Future positions receive weight 0.
	attention_weight_matrix = [softmax(a) for a in attention_score_matrix]

	#% Shape: context_length x embedding_size
	#* Computes the necessary information from previous and current position for each position in context_length in a blended vector
	attention_output_matrix = [get_attention_output_for_position(i, attention_weight_matrix, value_matrix) for i in range(CONTEXT_LENGTH)]

	#% Shape: context_length x number_of_ids
	#* Dot products attention context vectors with each vocab character's output weights,
	#* so if logits_matrix[10][20] is high, it means the attenuated output vector at position 10 supports the usage of character_id 20, so 'u' in this case
	logits_matrix = [[dot_product(attention_output_matrix[position], output_weight_table[character_position]) for character_position in range(number_of_ids)] for position in range(CONTEXT_LENGTH)]

	#% Shape: context_length x number_of_ids
	#* Softmaxes logits_matrix row by row, so each position gets softmaxed where the character ids are turned into probabilities
	probability_matrix = [softmax(logit_vector) for logit_vector in logits_matrix]

	# 0: input_token_ids, 1: target_token_ids, 2: input_token_embeddings
	# 3: input_position_embeddings, 4: combined_input_embeddings, 5: query_matrix
	# 6: key_matrix, 7: value_matrix, 8: attention_score_matrix
	# 9: attention_weight_matrix, 10: attention_output_matrix, 11: logits_matrix
	# 12: probability_matrix, 13: average loss

	return [input_token_ids, target_token_ids, input_token_embeddings,
			input_position_embeddings, combined_input_embeddings, query_matrix,
			key_matrix, value_matrix, attention_score_matrix,
			attention_weight_matrix, attention_output_matrix, logits_matrix,
			probability_matrix, get_average_loss(probability_matrix, target_token_ids)]

def run_backpropogation(forward_pass_results):
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

	#$ dLoss / dAttention_output_value
	attention_output_gradient = [[0 for _ in range(len(attention_output_matrix[0]))] for _ in range(len(attention_output_matrix))]
	for character in range(len(output_weight_table)):    #number_of_ids
		output_weight_vector = output_weight_table[character]
		for i in range(len(attention_output_gradient)):    #context_length
			correct_id = target_token_ids[i]
			#* The change in loss with respect to the logit, so if it's negative, an increase in the logit is a decrease in loss. simplifies to this expression below. (dLoss/dLogit)
			gradient = (probability_matrix[i][character] - (0 if correct_id != character else 1)) / CONTEXT_LENGTH
			for j in range(len(attention_output_gradient[0])):    #embedding_size
				#* chain rule. change in loss with respect to logit value * change in logit_value with respect to an attention_vector value. so change in loss with respect to attention_vector value
				attention_output_gradient[i][j] += gradient * output_weight_vector[j]

	#$ dLoss / dOutput_weight_values
	for position in range(len(attention_output_matrix)):    #context_length
		attention_vector = attention_output_matrix[position]
		correct_id = target_token_ids[position]
		for i in range(len(output_weight_table)):    #number_of_ids
			#* The change in loss with respect to the logit, so if it's negative, an increase in the logit is a decrease in loss. simplifies to this expression below. (dLoss/dLogit)
			gradient = (probability_matrix[position][i] - (0 if i != correct_id else 1)) / CONTEXT_LENGTH
			for j in range(len(output_weight_table[0])):    #embedding_size
				#* chain rule * learning_rate. change in loss with respect to logit value * change in logit_value with respect to an output_weight_matrix values. so change in loss with respect to output_weight_matrix values
				output_weight_table[i][j] -= learning_rate * gradient * attention_vector[j]

	#% Shape: context_length x embedding_size
	#$ dLoss / dValue_matrix_value
	value_matrix_gradient = [[0 for _ in range(len(value_matrix[0]))] for _ in range(len(value_matrix))]
	for position in range(len(attention_weight_matrix)):    #context_length
		attention_weight_vector = attention_weight_matrix[position]
		for i in range(len(value_matrix)):    #context_length
			for j in range(len(value_matrix[0])):    #embedding_size
				value_matrix_gradient[i][j] += attention_weight_vector[i] * attention_output_gradient[position][j]

	#% Shape: context_length x context_length
	#$ dLoss / dAttention_weight_value
	attention_weight_gradient = [[0 for _ in range(len(attention_weight_matrix[0]))] for _ in range(len(attention_weight_matrix))]
	for position in range(len(value_matrix)):    #context_length
		for i in range(len(attention_weight_matrix)):    #context_length
			value_vector = value_matrix[i]
			for j in range(len(value_vector)):    #embedding_size
				attention_weight_gradient[position][i] += value_vector[j] * attention_output_gradient[position][j]

	#$ dLoss / dAttention_score_value
	attention_score_gradient = [[0 for _ in range(len(attention_score_matrix[0]))] for _ in range(len(attention_score_matrix))]
	for position in range(len(attention_weight_matrix)):    #context_length
		attention_weight_vector = attention_weight_matrix[position]
		weighted_average_blame = 0
		for i in range(len(attention_weight_vector)):    #context_length
			weighted_average_blame += attention_weight_gradient[position][i] * attention_weight_vector[i]

		for i in range(len(attention_score_matrix[0])):    #context_length
			attention_score_gradient[position][i] = attention_weight_vector[i] * (attention_weight_gradient[position][i] - weighted_average_blame)

	#$ dLoss / dQuery_matrix_value
	query_matrix_gradient = [[0 for _ in range(len(query_matrix[0]))] for _ in range(len(query_matrix))]
	for position in range(len(query_matrix)):    #context_length
		for i in range(len(key_matrix)):    #context_length
			for j in range(len(key_matrix[0])):    #embedding_size
				query_matrix_gradient[position][j] += attention_score_gradient[position][i] * key_matrix[i][j] / math.sqrt(embedding_size)

	#$ dLoss / dKey_matrix_value
	key_matrix_gradient = [[0 for _ in range(len(key_matrix[0]))] for _ in range(len(key_matrix))]
	for position in range(len(key_matrix)):    #context_length
		for i in range(len(query_matrix)):    #context_length
			for j in range(len(query_matrix[0])):    #embedding_size
				key_matrix_gradient[i][j] += attention_score_gradient[position][i] * query_matrix[position][j] / math.sqrt(embedding_size)

	#$ dLoss / dCombined_input_value
	combined_input_embeddings_gradient = [[0 for _ in range(len(combined_input_embeddings[0]))] for _ in range(len(combined_input_embeddings))]
	for position in range(len(value_matrix)):    #len(query or key matrix): context_length
		for i in range(len(value_weight_table)):    #len(query or key weight table): embedding_size
			for j in range(len(combined_input_embeddings[0])):    #embedding_size
				combined_input_embeddings_gradient[position][j] += value_matrix_gradient[position][i] * value_weight_table[i][j]
				combined_input_embeddings_gradient[position][j] += key_matrix_gradient[position][i] * key_weight_table[i][j]
				combined_input_embeddings_gradient[position][j] += query_matrix_gradient[position][i] * query_weight_table[i][j]

	#$ dLoss / dValue_weight_table_value
	for position in range(len(value_matrix)):    #context_length
		for i in range(len(value_weight_table)):    #embedding_size
			for j in range(len(value_weight_table[0])):    #embedding_size
				value_weight_table[i][j] -= learning_rate * value_matrix_gradient[position][i] * combined_input_embeddings[position][j]
	
	#$ dLoss / dKey_weight_table_value
	for position in range(len(key_matrix)):    #context_length
		for i in range(len(key_weight_table)):    #embedding_size
			for j in range(len(key_weight_table[0])):    #embedding_size
				key_weight_table[i][j] -= learning_rate * key_matrix_gradient[position][i] * combined_input_embeddings[position][j]

	#$ dLoss / dQuery_weight_table_value
	for position in range(len(query_matrix)):    #context_length
		for i in range(len(query_weight_table)):    #embedding_size
			for j in range(len(query_weight_table[0])):    #embedding_size
				query_weight_table[i][j] -= learning_rate * query_matrix_gradient[position][i] * combined_input_embeddings[position][j]

	#$ dLoss / dInput_position_embeddings_value
	input_position_embeddings_gradient = combined_input_embeddings_gradient
	#$ dLoss / dPositional_embedding_table_value
	for i in range(len(input_position_embeddings)):
		for j in range(len(input_position_embeddings[0])):
			positional_embedding_table[i][j] -= learning_rate * input_position_embeddings_gradient[i][j]

	#$ dLoss / dInput_token_embeddings_value
	input_token_embeddings_gradient = combined_input_embeddings_gradient
	#$ dLoss / dToken_embedding_table_value
	for i in range(len(input_token_embeddings)):
		character_id = input_token_ids[i]
		for j in range(len(input_token_embeddings[0])):
			token_embedding_table[character_id][j] -= learning_rate * input_token_embeddings_gradient[i][j]

def get_predicted_id_for_position(position, probability_matrix):
	number_of_ids = len(probability_matrix[0])
	predicted_id = 0
	current_max_probability = 0
	for i in range(number_of_ids):
		if current_max_probability < probability_matrix[position][i]: 
			predicted_id = i
			current_max_probability = probability_matrix[position][i]

	return predicted_id

def get_prediction_debug_for_position(position, probability_matrix, target_token_ids):
	number_of_ids = len(probability_matrix[0])
	predicted_id = 0
	current_max_probability = 0
	for i in range(number_of_ids):
		if current_max_probability < probability_matrix[position][i]: 
			predicted_id = i
			current_max_probability = probability_matrix[position][i]

	correct_id = target_token_ids[position]
	print(f"Input character at this position: {id_to_character[target_token_ids[position - 1]]}")
	print(f"Correct next character: {id_to_character[correct_id]}")
	print(f"Predicted next character: {id_to_character[predicted_id]}")
	print(f"Probability of correct character: {probability_matrix[position][correct_id]}")
	print(f"Probability of predicted character: {current_max_probability}")

def save_model(path="character_transformer_checkpoint.pt"):
	"""
	Save model parameters, configuration, vocabulary, and random states.
	"""

	checkpoint = {
		"version": 1,

		# Model configuration
		"embedding_size": embedding_size,
		"context_length": CONTEXT_LENGTH,
		"number_of_ids": number_of_ids,
		"learning_rate": learning_rate,

		# Vocabulary
		"character_to_id": character_to_id,
		"id_to_character": id_to_character,

		# Model parameters
		"token_embedding_table": copy.deepcopy(token_embedding_table),
		"positional_embedding_table": copy.deepcopy(positional_embedding_table),
		"query_weight_table": copy.deepcopy(query_weight_table),
		"key_weight_table": copy.deepcopy(key_weight_table),
		"value_weight_table": copy.deepcopy(value_weight_table),
		"output_weight_table": copy.deepcopy(output_weight_table),

		# Optional: preserve randomness between runs
		"python_random_state": random.getstate(),
		"torch_random_state": torch.get_rng_state(),
	}

	torch.save(checkpoint, path)
	print(f"Model saved to: {os.path.abspath(path)}")


def load_model(path="character_transformer_checkpoint.pt"):
	"""
	Load model parameters into the global model tables.

	The model configuration and vocabulary must match the checkpoint.
	"""

	global token_embedding_table
	global positional_embedding_table
	global query_weight_table
	global key_weight_table
	global value_weight_table
	global output_weight_table

	if not os.path.exists(path):
		raise FileNotFoundError(f"No checkpoint found at: {path}")

	checkpoint = torch.load(path, map_location="cpu")

	# Validate that the checkpoint matches this model definition
	if checkpoint["embedding_size"] != embedding_size:
		raise ValueError(
			f"Embedding size mismatch: "
			f"checkpoint={checkpoint['embedding_size']}, "
			f"current={embedding_size}"
		)

	if checkpoint["context_length"] != CONTEXT_LENGTH:
		raise ValueError(
			f"Context length mismatch: "
			f"checkpoint={checkpoint['context_length']}, "
			f"current={CONTEXT_LENGTH}"
		)

	if checkpoint["character_to_id"] != character_to_id:
		raise ValueError(
			"Vocabulary mismatch. The current character_to_id mapping "
			"does not match the checkpoint."
		)

	# Restore model parameters
	token_embedding_table = checkpoint["token_embedding_table"]
	positional_embedding_table = checkpoint["positional_embedding_table"]
	query_weight_table = checkpoint["query_weight_table"]
	key_weight_table = checkpoint["key_weight_table"]
	value_weight_table = checkpoint["value_weight_table"]
	output_weight_table = checkpoint["output_weight_table"]

	# Restore random states if available
	if "python_random_state" in checkpoint:
		random.setstate(checkpoint["python_random_state"])

	if "torch_random_state" in checkpoint:
		torch.set_rng_state(checkpoint["torch_random_state"])

	print(f"Model loaded from: {os.path.abspath(path)}")

	return checkpoint

example_start_position = 128

# test1 = get_input_token_ids(example_start_position)
# test2 = get_target_token_ids(example_start_position)
# print(decode_token_ids(test1))
# print(decode_token_ids(test2))
# print()

#& Settings
#! 2 temporarily, otherwise 16
embedding_size = 24
CONTEXT_LENGTH = 48
number_of_training_windows = len(token_ids) - CONTEXT_LENGTH
learning_rate = 0.0025

#& Embedding and weight table initialization + mask

#% Shape: vocabulary_size × embedding_size. 
#* This is the map for all characters in the vocabulary to their respective vector, initialized random at first and changes later to learn relationships.
token_embedding_table  = [[random.random() * 2 - 1 for _ in range(embedding_size)] for _ in range(len(id_to_character))]

#% Shape: context_length x embedding_size
#* This gives each possible position in the context window its own learnable vector, so the model can distinguish position 0 from position 1, position 2, etc.
positional_embedding_table = [[random.random() * 2 - 1 for _ in range(embedding_size)] for _ in range(CONTEXT_LENGTH)]

#% Shape: embedding_size x embedding_size
#* Turns raw input vectors into queries to ask what it needs
query_weight_table = [[random.random() * 0.6 - 0.3 for _ in range(embedding_size)] for _ in range(embedding_size)]

#% Shape: embedding_size x embedding_size
#* Turns raw input vectors into keys to advertise what a query might need
key_weight_table = [[random.random() * 0.6 - 0.3 for _ in range(embedding_size)] for _ in range(embedding_size)]

#% Shape: embedding_size x embedding_size
#* Turns raw input vectors into values to convey useful information about the next position
value_weight_table = [[random.random() * 0.6 - 0.3 for _ in range(embedding_size)] for _ in range(embedding_size)]

#% Shape: number_of_ids x embedding_size
#* Determines/asks how much an attention_output vector supports a character being next
output_weight_table = [[random.random() * 2 - 1 for _ in range(embedding_size)] for _ in range(number_of_ids)]

#% Shape: context_length x context_length
#* Shows what positions a given position is allowed to look at
causal_attention_mask = [[1 if j <= i else 0 for j in range(CONTEXT_LENGTH)] for i in range(CONTEXT_LENGTH)]

#& Running forward pass
forward_pass_results = run_forward_pass(example_start_position)

#& Assigning each matrix and vector their result from forward pass

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




#? Token embed testing
#print(len(token_embedding_table))

#? Input embed testing
# print(len(positional_embedding_table))
# print(len(positional_embedding_table[0]))

#? Input position testing
# print(len(input_position_embeddings))
# print(len(input_position_embeddings[0]))

# print(len(transform_vector(combined_input_embeddings[0], query_weight_table)))

#? QKV Testing
# print(len(query_matrix), len(query_matrix[0]))
# print(len(key_matrix), len(key_matrix[0]))
# print(len(value_matrix), len(value_matrix[0]))


#? Combined input testing
# print(len(combined_input_embeddings))
# print(len(combined_input_embeddings[0]))

# print(len(causal_attention_mask))
# print(len(causal_attention_mask[0]))
# print(causal_attention_mask[0][:10])
# print(causal_attention_mask[10][:15])

#? Dot product testing
# print(dot_product([2, 1, 3], [4, 0, 5]))
# print(dot_product(combined_input_embeddings[10], combined_input_embeddings[3]))

#? Attention score testing
if (0):
	# attention_scores_for_position_0 = get_attention_scores_for_position(0, query_matrix, key_matrix)
	# attention_scores_for_position_1 = get_attention_scores_for_position(1, query_matrix, key_matrix)
	# attention_scores_for_position_10 = get_attention_scores_for_position(10, query_matrix, key_matrix)
	# attention_scores_for_position_63 = get_attention_scores_for_position(63, query_matrix, key_matrix)

	# print(attention_scores_for_position_0[:8])
	# print(attention_scores_for_position_1[:8])
	# print(attention_scores_for_position_10[:15])
	# print(attention_scores_for_position_63[-8:])

	# print(len(attention_score_matrix))
	# print(len(attention_score_matrix[0]))

	# print(attention_score_matrix[0][:8])
	# print(attention_score_matrix[1][:8])
	# print(attention_score_matrix[10][:15])
	# print(attention_score_matrix[63][-8:])

	# print(attention_score_matrix[10][10])
	# print(attention_score_matrix[10][11])
	pass

#? Attention weight testing
if (0):
	# attention_weights_for_position_10 = masked_softmax(attention_scores_for_position_10)

	# print(len(attention_weights_for_position_10))
	# print(attention_weights_for_position_10[:15])
	# print(attention_weights_for_position_10[10])
	# print(attention_weights_for_position_10[11])
	# print(sum(attention_weights_for_position_10))

	# print(len(attention_weight_matrix))
	# print(len(attention_weight_matrix[0]))

	# print(attention_weight_matrix[0][:8])
	# print(attention_weight_matrix[1][:8])
	print(attention_weight_matrix[10][:15])
	# print(attention_weight_matrix[63][-8:])

	print(sum(attention_weight_matrix[10]))

#? Attention Output testing
if(0):

	# attention_output_for_position_10 = get_attention_output_for_position(10, attention_weight_matrix, value_matrix)
	# attention_output_for_position_0 = get_attention_output_for_position(0, attention_weight_matrix, value_matrix)
	# attention_output_for_position_63 = get_attention_output_for_position(63, attention_weight_matrix, value_matrix)

	# print(len(attention_output_for_position_10))
	# print(attention_output_for_position_10[:5])
	# print(sum(attention_weight_matrix[10]))

	# print(len(attention_output_for_position_0))
	# print(attention_output_for_position_0[:5])
	# print(combined_input_embeddings[0][:5])

	# print(len(attention_output_for_position_63))
	# print(attention_output_for_position_63[:5])
	# print(sum(attention_weight_matrix[63]))

	print(len(combined_input_embeddings[0]))
	print(attention_output_matrix[0][:5])
	print(value_matrix[0][:5])
	print(attention_output_matrix[10][:5])
	# print(len(attention_output_for_position_10))
	pass

# print(len(output_weight_table), len(output_weight_table[0]))
# print(len(logits_matrix), len(logits_matrix[0]))

# print(logits_matrix[10][:15])

# print(len(output_weight_table), len(output_weight_table[0]))
print(len(probability_matrix), len(probability_matrix[0]))

print(probability_matrix[10][:15])
print(sum(probability_matrix[10]))

correct_id = target_token_ids[10]

print(correct_id)
print(id_to_character[correct_id])

print(probability_matrix[10][target_token_ids[10]])

predicted_id = 0
current_max_probability = 0

for i in range(number_of_ids):
	if current_max_probability < probability_matrix[10][i]: 
		predicted_id = i
		current_max_probability = probability_matrix[10][i]

print(predicted_id)
print(id_to_character[predicted_id])
print(probability_matrix[10][predicted_id])

print(get_loss_for_position(10, probability_matrix, target_token_ids))
# print(get_average_loss(probability_matrix, target_token_ids))

print(f"Attention_output_matrix: \n Shape: {CONTEXT_LENGTH} x {embedding_size} \n {[[round(x, 3) for x in row] for row in attention_output_matrix]} \n")
print(f"Old Output_weight_table: \n Shape: {number_of_ids} x {embedding_size} \n {[[round(x, 5) for x in row] for row in output_weight_table]} \n")
run_backpropogation(forward_pass_results)
print(f"New Output_weight_table: \n Shape: {number_of_ids} x {embedding_size} \n {[[round(x, 5) for x in row] for row in output_weight_table]} \n")
print(get_average_loss(probability_matrix, target_token_ids))

load_model("checkpoints/best_model.pt")

# Test the loaded model
forward_pass_results = run_forward_pass(example_start_position)

probability_matrix = forward_pass_results[12]
target_token_ids = forward_pass_results[1]

print("Loss:", forward_pass_results[13])
print(
    "Predicted ID:",
    get_predicted_id_for_position(10, probability_matrix)
)

iteration = 0
loss_window_average = 0.0
window_loss_size = 200
current_least_loss = float("inf")

total_start_time = time.perf_counter()
window_start_time = time.perf_counter()

while True:
    iteration += 1

    start_position = random.randint(
        0,
        number_of_training_windows - 1
    )

    forward_pass_results = run_forward_pass(start_position)

    # Loss calculated from the current randomly selected window
    window_loss = forward_pass_results[13]

    # Update model parameters
    run_backpropogation(forward_pass_results)

    loss_window_average += window_loss

    if iteration % window_loss_size == 0:
        average_loss = loss_window_average / window_loss_size

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
            f"Iterations {iteration - window_loss_size + 1}"
            f"-{iteration} took "
            f"{window_elapsed_time:.2f} seconds"
        )

        print(
            f"Iteration: {iteration}, "
            f"average loss: {average_loss:.6f}, "
            f"speed: {iterations_per_second:.2f} iterations/sec, "
            f"total time: {total_elapsed_time / 3600:.2f} hours"
        )

        # Save a regular checkpoint every 5,000 iterations
        if iteration % 5_000 == 400:
            save_model(
                f"checkpoints/checkpoint_{iteration}.pt"
            )

        # Save the best model seen so far
        if average_loss < current_least_loss:
            current_least_loss = average_loss

            save_model("checkpoints/best_model.pt")

            print(
                f"New best model saved with loss: "
                f"{current_least_loss:.6f}"
            )

        loss_window_average = 0.0
        window_start_time = time.perf_counter()
		
# print(get_loss_for_training_window(1000))

# total_loss_over_many_windows = 0

# step = 1
# while(step <= 2 ** 10):
# 	loss_step = get_loss_for_training_window(step)
# 	total_loss_over_many_windows += loss_step
# 	print(f"Position {step}'s loss is {loss_step}")
# 	step *= 2
# print(f"Avg loss: {total_loss_over_many_windows / 11}")

# get_prediction_debug_for_position(10, probability_matrix, target_token_ids)
# print(get_predicted_id_for_position(10, probability_matrix))
