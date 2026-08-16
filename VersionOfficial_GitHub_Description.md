# VersionOfficial local update summary

This local version of VersionOfficial.py differs from the repo copy in the optimizer and checkpoint logic for the QKV path. The goal was to make the query/key/value split explicit and keep the optimizer state compatible with the downstream code path that expects separate Q/K/V behavior.

## Key changes

### 1) Split the QKV optimizer state into query, key, and value paths
- Replaced the old single aggregated QKV momentum bookkeeping with separate query/key/value tracking.
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

### 2) Reworked the update logic to compute Q/K/V separately instead of as one fused QKV block
- The optimizer update path now slices the QKV projection into:
  - query block
  - key block
  - value block
- Each block computes its own momentum gradient, row RMS, log-ratio, z-score, norm-power, ratio, and momentum-ratio values.
- This makes the optimizer behavior align with the next code path that uses Q/K/V independently rather than a single aggregate QKV object.

### 3) Added row-wise Q/K/V statistics and norm-power calculations
- Added per-row values such as:
  - `query_row_norm_power`
  - `key_row_norm_power`
  - `value_row_norm_power`
  - `query_row_ratio`
  - `key_row_ratio`
  - `value_row_ratio`
  - `query_row_history_ratio`
  - `key_row_history_ratio`
  - `value_row_history_ratio`
- The optimizer now calculates row-wise log-ratio, z-score, and norm-power for each of Q/K/V separately.

### 4) Switched the layer-level scaling variable to `pmax_layer`
- The checkpoint metadata now stores `pmax_layer` instead of the old `pmax` field.
- The loader validates for `pmax_layer` and loads the matching reference state.
- This aligns the saved state with the current per-layer adaptive update logic.

### 5) Removed epsilon before `.sqrt()` in the RMS computations
- The RMS calculations were adjusted to avoid applying epsilon before the square root.
- This keeps the adaptive statistics in the intended scale and matches the user-requested optimizer behavior.

### 6) Updated checkpoint compatibility and save/load logic
- Save logic writes the new Q/K/V reference momentum fields.
- Load logic restores those separate buffers and validates the checkpoint baseline keys.
- This preserves compatibility with the refreshed optimizer state format without changing unrelated parts of the training pipeline.

### 7) Kept the combined QKV-compatible variables only where needed for compatibility
- The local version still keeps the QKV aggregate-style debug tables in place for compatibility with downstream code.
- However, the actual optimization state and calculations are now split and applied independently per query, key, and value block.

## Summary
This local version is a targeted optimizer refactor for the attention projection path: the model still behaves as a QKV projection overall, but the update math and tracking state are now separated by query, key, and value so the later code path can consume the correct per-branch statistics without breaking compatibility.
