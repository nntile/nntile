# Classic NNTile kernels (`torch_nntile.nn`)

Stock PyTorch on `device=nntile` uses torch-native StarPU codelets
(`TORCH_*` TensorGraph ops). **`torch_nntile.nn`** records classic
`nntile::tensor` ops that lower to `nntile::kernel`.

| `torch_nntile.nn.functional` | TensorGraph / kernel |
|---|---|
| `gemm` / `matmul` | `tensor::gemm` |
| `add` | `tensor::add` |
| `mul` | `tensor::multiply` |
| `mul_scalar` | `tensor::scale` |
| `add_fiber` | `tensor::add_fiber` |
| `sum_slice` / `gap` | `tensor::sum_slice` |
| `relu` / `gelu` / `silu` | classic activations (+ matching backward) |
| `layer_norm` | composed `sum_slice` / `hypot` / `multiply_*` / `add_fiber` |
| `rms_norm` | composed classic RMSNorm |
| `embedding` | `tensor::embedding` |
| `rope` | `tensor::rope` |
| `sdpa_kernel` | classic SDPA (gemm + softmax path) |
| `cross_entropy` / `mse_loss` | classic loss kernels |

Rules:

- **fp32**, **contiguous**, `storage_offset == 0` (densify with
  autograd-tracked `.contiguous()` first).
- Tiling is allowed on classic-only graphs. A pending `TORCH_*` op plus
  tiled axis groups raises at compile.
- C++ `torch_nntile::models` are the nntile-native implementations (ports
  of deleted `nntile::model::*`). Wire them to these kernels; do **not**
  reimplement GPT-2 / Llama / BERT / … from Hugging Face `torch.nn`.
  Tests: `test_models_classic_graph.py` (`_C.cpp_*` fwd+bwd; no
  `TORCH_*` except autograd grad-combine `TORCH_BINARY`).
- Python `torch_nntile.models` are the same stacks for the Python API
  (CI: `.github/scripts/check-model-classic-nn.sh`; graph:
  `test_models_python_classic_graph.py`; tiled smoke:
  `test_models_python_tiled_smoke.py`). They are not a second HF port.
- Stock Hugging Face / `torch.nn` on `device=nntile` (torch-native):
  `test_hf_stock_models_on_nntile.py` and `test_aten_ops_parity.py`.

CMake (both default ON):

- `NNTILE_TORCH_NATIVE_OPS` — torch-native aten / StarPU `TORCH_*`
  codelets (stock `torch.nn` on `device=nntile`).
- `NNTILE_NNTILE_NATIVE_OPS` — classic `nntile::kernel` wrappers, C++
  models, and `torch_nntile.nn`. Turn one flag off to skip that stack
  while developing the other.
