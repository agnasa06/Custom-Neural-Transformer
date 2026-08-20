# VersionOfficial change log

This document records the changes in commit order across `VersionOfficial.py`, `RunningAI.py`, and `DataTokenization.py`. The current model uses a character-level vocabulary with 221 token IDs and 384-token training windows.

## Commit 1: TorchAO / runtime optimization and profiling updates

This change set updates the training runtime and vendorized TorchAO path.

### `VersionOfficial.py`

- Uses `torch.nn.functional.scaled_dot_product_attention` with causal attention instead of manually constructing attention score, mask, softmax, and value product logic.
- Computes the language-model loss with `F.cross_entropy` over flattened logits and targets.
- Groups momentum-gradient calculation alongside the corresponding parameter update so the QKV and FFN layer updates are applied in the same loop.
- Uses `normalized_learning_rate` consistently for embedding, output, gamma, QKV, and FFN updates.
- Enables TorchInductor CUDAGraph support for input mutation through `torch._inductor.config.triton.cudagraph_support_input_mutation`.
- Compiles the training and update functions with `torch.compile(..., mode="max-autotune")`.
- Configures TorchAO FP8 training to avoid recomputing FP8 weights during backward and enables fast accumulation for gradient-input and gradient-weight GEMMs.
- Adds a warmup before profiling and an optimizer profiling phase that exports `optimizer_profile.json`.

### Vendored TorchAO `float8_linear.py`

- The forward pass stores the already-converted FP8 input and weight tensors in the autograd context along with the high-precision tensors.
- The backward pass reuses those saved FP8 tensors for input-gradient and weight-gradient matrix multiplications.
- Backward no longer recasts the weight, gradient output, and input tensors to FP8 for those calculations.

## Commit 2: QKV optimizer split and checkpoint compatibility

This change set refactors the optimizer logic so query, key, and value paths are tracked independently while preserving the downstream QKV code path.

### Split Q/K/V optimizer state

- Replaced aggregate QKV momentum bookkeeping with separate query, key, and value tracking.
- Added separate reference momentum RMS buffers and row-, column-, and layer-level debug tables for each path.
- The optimizer update path slices the QKV projection into query, key, and value sections, each with independent momentum, RMS, log-ratio, z-score, norm-power, ratio, and history calculations.

### Adaptive statistics and checkpoint compatibility

- Added row-wise, column-wise, and layer-wise statistics for Q/K/V, including history ratios and peer multipliers.
- Extended the same adaptive normalization approach to FFN and Wo projections.
- Switched stored layer-level scaling metadata to `pmax_layer`.
- Removed epsilon before the RMS square-root calculation so adaptive statistics remain on the intended scale.
- Updated save/load logic to persist and restore the split Q/K/V reference momentum state and optimizer baselines.

## Commit 3: FineWeb-Edu, model architecture, and inference additions

This change set contains the new uncommitted additions documented below.

### `DataTokenization.py`

- Replaced the local WikiText `.tokens` reader and `torch.save` output with a streaming FineWeb-Edu loader using the `HuggingFaceFW/fineweb-edu` `sample-10BT` split.
- Writes the dataset to `finewebedu_uint8.bin` as raw `numpy.uint8` bytes, allowing the training script to consume the result with `np.memmap` without loading the full corpus into memory.
- Expanded the vocabulary from the original basic character set to 221 IDs, including mathematical symbols, Greek letters, arrows, superscripts, subscripts, fractions, currency symbols, punctuation, and accented Latin characters.
- Added Unicode normalization for smart quotes, dash variants, non-breaking and unusual spaces, zero-width characters, micro signs, ordinal symbols, and common accented characters.
- Uses a full Unicode code-point lookup table and NumPy vectorization instead of Python character-by-character tensor construction.
- Appends two newline characters after every streamed document to preserve document separation.
- Added `max_characters` so a bounded conversion can be requested without changing the streaming logic.
- Tracks document count, total characters, unknown characters, unknown rate, and the most common unknown characters while processing.
- Prints periodic progress every 1,000 documents and unknown-character details every 10,000 documents.

### `VersionOfficial.py`

#### Input and vocabulary

- Updated the in-script vocabulary to match the 221-token tokenizer.
- Replaced the previous serialized WikiText tensor input with `finewebedu_uint8.bin` loaded through `np.memmap`.
- Splits the memory-mapped byte file into training and validation regions without copying the complete dataset.

#### Model forward pass

- Replaced manual matrix multiplications with `nn.Linear` modules for QKV, FFN up, FFN down, and attention output projections.
- Added rotary positional embeddings using precomputed sine and cosine tables.
- Applies RoPE independently to query and key vectors after reshaping each head into pairs.
- Uses fused causal `F.scaled_dot_product_attention` instead of explicitly materializing the causal mask, attention scores, softmax, and value product.
- Removed learned positional embeddings from the input path; token embeddings are now the initial layer state.
- Uses `F.cross_entropy` directly on flattened logits and shifted target IDs.
- Added configurable temperature handling to the logit-to-probability path used for generation.

#### Optimizer and calibration

- Added a fresh-model calibration phase with burn-in steps to establish fixed momentum RMS baselines before adaptive updates begin.
- Tracks optimizer statistics at row, column, and layer levels for Q, K, V, FFN up, FFN down, and Wo projections.
- Keeps Q/K/V momentum, ratios, history ratios, z-scores, norm powers, and update multipliers independent rather than treating the fused QKV matrix as one family.
- Applies the normalized learning rate consistently to embedding, output, gamma, QKV, FFN, and Wo updates.
- Added optimizer history correction using fixed reference momentum RMS values and `global_norm_power`.
- Added periodic debug output for Q/K/V, FFN, and Wo multipliers, including z-score, norm power, ratio, total multiplier, and momentum multiplier.
- Preserves optimizer baselines in checkpoints so resumed training uses the same calibration reference.

#### Runtime and FP8

- Enables TorchInductor compiled autograd and CUDAGraph input-mutation support.
- Compiles training, momentum update, optimizer-statistics, and weight-update functions with the selected autotune modes.
- Converts the QKV, FFN, and Wo modules to TorchAO FP8 training.
- Disables FP8 weight recomputation during backward and enables fast accumulation for gradient-input and gradient-weight GEMMs.
- Adds validation at fixed seeded positions and periodic best-model/checkpoint saves.
- Preserves elapsed training time across restarts and supports a configurable maximum session duration.
- Reads `learning_scale.txt` at startup and during training so the user scale can be adjusted without editing the model script.
- Adds fail-fast single-GPU detection through `nvidia-smi` before training begins.

#### Checkpoint compatibility

- Checkpoints now store module weights and restore them with `nn.Parameter`/`nn.Linear` structures rather than relying only on standalone tensor tables.
- Added validation for vocabulary size, context length, head count, frequency base, layer count, FFN sizes, required momentum buffers, optimizer baselines, and expected QKV/Wo shapes.
- Saves and restores Q/K/V reference momentum RMS values, FFN baselines, Wo baselines, momentum buffers, learning-rate persistence, loss, and elapsed time.
- Writes a baseline checkpoint immediately after calibration so the calibrated optimizer state survives an early restart.

### `RunningAI.py`

- Updated the inference script to the same 221-token vocabulary and checkpoint schema as `VersionOfficial.py`.
- Added robust checkpoint loading with configuration checks, required-state checks, and restoration of module weights, parameters, momentum tensors, and optimizer baselines.
- Migrated inference from raw weight-table multiplication to the `nn.Linear` QKV, FFN, and Wo modules.
- Added RoPE query/key rotation and fused causal scaled dot-product attention.
- Removed the old learned positional-embedding path and uses token embeddings as the initial state.
- Added temperature-controlled generation with configurable `top_k` sampling and an `always_best` greedy mode.
- Keeps the generation context bounded with a sliding window so prompts can grow beyond the model context length.
- Added single-GPU detection and the same TorchAO FP8 module conversion used by the training path.
- Added checkpoint directory handling relative to the script location and clearer model-load/resume diagnostics.
- Retains top-probability inspection and now displays special characters such as spaces and newlines clearly.

## Summary

The previous version trained from a local WikiText tensor using manually managed projection tables and a simpler optimizer state. The current version uses a streamed FineWeb-Edu binary corpus, a substantially expanded Unicode-aware character vocabulary, RoPE plus fused attention, module-based FP8 execution, calibrated adaptive optimization across every major projection family, stronger checkpoint validation, and an inference script that matches the training architecture.
