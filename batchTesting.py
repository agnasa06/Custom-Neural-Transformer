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
		"ffn_up_weight_table_layer1",
		"ffn_down_weight_table_layer1",
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
		global ffn_up_weight_table_layer1
		global ffn_down_weight_table_layer1

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

		ffn_up_weight_table_layer1 = snapshot[
			"ffn_up_weight_table_layer1"
		].clone()

		ffn_down_weight_table_layer1 = snapshot[
			"ffn_down_weight_table_layer1"
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