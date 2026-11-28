<!--
 @copyright (c) 2022-present Skolkovo Institute of Science and Technology
                              (Skoltech), Russia. All rights reserved.
                2023-present Artificial Intelligence Research Institute
                              (AIRI), Russia. All rights reserved.

 NNTile is software framework for fast training of big neural networks on
 distributed-memory heterogeneous systems with GPU acceleration.

 @file torch_nntile/README.md
 @version 2.0.0
-->

# torch_nntile

PyTorch `device="nntile"`, backed by PyTorch autograd and NNTile 2.0's
NNHaul runtime. One CUDA GPU per process; chunk-addressable kernels.

## Surface

Only `torch_nntile.nn` is supported:

- `torch_nntile.nn.functional` — classic NNTile autograd ops
- `torch_nntile.nn.module` — classic modules
- `torch_nntile.nn.model` — full models (`GPT2LMHead`, `Llama`, `BERT`, …)

Stock `torch.nn`/`F.*` on `device="nntile"`, CPU/macOS, and flash SDPA are
not part of NNTile 2.0.

## Build

```bash
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake --preset nnhaul -DCMAKE_PREFIX_PATH="$TORCH_PREFIX"
cmake --build --preset nnhaul -j
```

The local wheel lands in `build-nnhaul/wheelhouse` and links the build-tree
`libnntile`, `libnnhaul`, and `libtorch_nntile`. `LD_LIBRARY_PATH` must point
at those build directories unless the wheel is repaired with `auditwheel`.

## Run

```python
import os
os.environ["NNTILE_ENABLE_LOCAL_COMPILER"] = "1"

import torch
import torch_nntile

torch_nntile.init_context(ncpu=0, ncuda=1)
a = torch.randn(4, 8).to("nntile")
b = torch.randn(8, 16).to("nntile")
c = torch_nntile.nn.functional.gemm(a, b, ndim=1)
torch_nntile.compile_graph()
torch_nntile.run()
torch_nntile.wait()
```

Classic GPT-2 training: [`examples/train_gpt2.py`](examples/train_gpt2.py).

## Migration

NNTile 1.x remains the place for stock-`torch.nn`/HF on `nntile`, CPU,
macOS, and flash SDPA. See `docs/torch_nntile.md` in the repository.
