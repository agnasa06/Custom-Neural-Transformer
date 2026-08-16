# VersionOfficial change log

This document consolidates the main local changes made to `VersionOfficial.py` and keeps each set of updates associated with its own commit area.

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
- Adds a 50-iteration warmup before profiling.
- Adds an optimizer profiling phase that reuses one generated gradient set, profiles 100 compiled optimizer updates, and exports `optimizer_profile.json`.

### Vendored TorchAO `float8_linear.py`
Compared with the corresponding backup from the installed TorchAO environment:
- The forward pass stores the already-converted FP8 input and weight tensors in the autograd context along with the high-precision tensors.
- The backward pass reuses those saved FP8 tensors for the input-gradient and weight-gradient matrix multiplications.
- Backward no longer recasts the weight, gradient output, and input tensors to FP8 for those calculations.
- This reduces repeated FP8 conversion work and aligns with the `force_recompute_fp8_weight_in_bwd=False` configuration used by `VersionOfficial.py`.

## Commit 2: QKV optimizer split and checkpoint compatibility

This change set refactors the optimizer logic so the query, key, and value paths are tracked independently while preserving compatibility with the downstream QKV code path.

### Split Q/K/V optimizer state
- Replaced the previous single aggregate QKV momentum bookkeeping with separate query, key, and value tracking.
- Added separate reference momentum RMS buffers:
  - `query_reference_momentum_rms`
  - `key_reference_momentum_rms`
  - `value_reference_momentum_rms`
- Added separate row- and layer-level debug tables for each path:
  - `query_multiplier`, `key_multiplier`, `value_multiplier`
  - `query_zscore_table`, `key_zscore_table`, `value_zscore_table`
  - `query_norm_power_table`, `key_norm_power_table`, `value_norm_power_table`
  - `query_ratio_table`, `key_ratio_table`, `value_ratio_table`
  - `query_momentum_multiplier`, `key_momentum_multiplier`, `value_momentum_multiplier`

### Independent update logic for Q/K/V
- The optimizer update path now slices the QKV projection into query, key, and value sections.
- Each branch computes its own momentum gradient, row RMS, log-ratio, z-score, norm-power, ratio, and momentum-ratio values.
- This aligns the optimizer behavior with the next code path that consumes Q/K/V independently instead of as a single fused QKV object.

### Row-wise statistics and layer scaling
- Added per-row statistics such as:
  - `query_row_norm_power`
  - `key_row_norm_power`
  - `value_row_norm_power`
  - `query_row_ratio`
  - `key_row_ratio`
  - `value_row_ratio`
  - `query_row_history_ratio`
  - `key_row_history_ratio`
  - `value_row_history_ratio`
- Switched the stored layer-level scaling metadata to `pmax_layer` and updated the checkpoint validation to match.

### RMS fix
- Removed epsilon before the `.sqrt()` call in the RMS calculations so the adaptive statistics remain on the intended scale.

### Checkpoint compatibility
- Updated save/load logic to write and restore the new Q/K/V reference momentum state.
- Kept the merged QKV-compatible debug tables only where needed for downstream compatibility while the actual optimization math remains split by Q/K/V.

## Summary
The repo now contains both the broader runtime optimization work and the narrower QKV optimizer refactor, but they are organized here under separate commit sections so the scope of each change is clear.
