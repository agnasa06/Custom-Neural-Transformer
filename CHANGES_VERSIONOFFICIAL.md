# VersionOfficial Changes

This document records the changes in the current local `VersionOfficial.py` compared with the previous GitHub version, plus the TorchAO change being vendored in this commit.

## VersionOfficial.py

- Uses `torch.nn.functional.scaled_dot_product_attention` with causal attention instead of constructing the attention score, mask, softmax, and value product manually.
- Computes language-model loss with `F.cross_entropy` over flattened logits and targets.
- Groups the momentum-gradient calculation beside the corresponding parameter update. Each QKV and FFN layer now calculates and applies its own update in the same loop.
- Uses `normalized_learning_rate` consistently for the embedding, output, gamma, QKV, and FFN updates.
- Enables TorchInductor CUDAGraph support for input mutation through `torch._inductor.config.triton.cudagraph_support_input_mutation`.
- Compiles the training and update functions with `torch.compile(..., mode="max-autotune")`.
- Configures TorchAO FP8 training to avoid recomputing FP8 weights during backward and enables fast accumulation for gradient-input and gradient-weight GEMMs.
- Adds a 50-iteration warmup before profiling.
- Adds an optimizer profiling phase that reuses one generated gradient set, profiles 100 compiled optimizer updates, and exports `optimizer_profile.json`.

## Vendored TorchAO float8_linear.py

Compared with the matching `float8_linear_backup.py` from the installed TorchAO environment:

- The forward pass now saves the already-converted FP8 input and weight tensors in the autograd context along with the high-precision tensors.
- The backward pass reuses those saved FP8 tensors for the input-gradient and weight-gradient matrix multiplications.
- Backward no longer recasts the weight, gradient output, and input tensors to FP8 for those calculations.
- This is intended to reduce repeated FP8 conversion work and align with the `force_recompute_fp8_weight_in_bwd=False` configuration used by `VersionOfficial.py`.

The vendored file is copied from the configured `nvidia-test` environment’s TorchAO installation and corresponds to that installed TorchAO/PyTorch version. It is not a replacement for installing TorchAO; its surrounding TorchAO modules are still required at runtime.
