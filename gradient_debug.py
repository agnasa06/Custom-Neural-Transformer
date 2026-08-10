#~ Gradient debugging
if(0):
	print("\n========== GRADIENT DEBUG ==========")
	print(
		"layer1_output_gradient:",
		"shape =", tuple(layer1_output_gradient.shape),
		"min =", layer1_output_gradient.min().item(),
		"max =", layer1_output_gradient.max().item(),
		"mean =", layer1_output_gradient.mean().item(),
		"norm =", layer1_output_gradient.norm().item(),
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
		"ffn_down_weight_table_layer1_gradient:",
		"shape =", tuple(ffn_down_weight_table_layer1_gradient.shape),
		"min =", ffn_down_weight_table_layer1_gradient.min().item(),
		"max =", ffn_down_weight_table_layer1_gradient.max().item(),
		"mean =", ffn_down_weight_table_layer1_gradient.mean().item(),
		"norm =", ffn_down_weight_table_layer1_gradient.norm().item(),
	)

	print(
		"ffn_upscale_matrix_layer1_gradient:",
		"shape =", tuple(ffn_upscale_matrix_layer1_gradient.shape),
		"min =", ffn_upscale_matrix_layer1_gradient.min().item(),
		"max =", ffn_upscale_matrix_layer1_gradient.max().item(),
		"mean =", ffn_upscale_matrix_layer1_gradient.mean().item(),
		"norm =", ffn_upscale_matrix_layer1_gradient.norm().item(),
	)

	print(
		"ffn_up_weight_table_layer1_gradient:",
		"shape =", tuple(ffn_up_weight_table_layer1_gradient.shape),
		"min =", ffn_up_weight_table_layer1_gradient.min().item(),
		"max =", ffn_up_weight_table_layer1_gradient.max().item(),
		"mean =", ffn_up_weight_table_layer1_gradient.mean().item(),
		"norm =", ffn_up_weight_table_layer1_gradient.norm().item(),
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
