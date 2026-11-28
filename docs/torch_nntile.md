<!--
 @copyright (c) 2022-present Skolkovo Institute of Science and Technology
                              (Skoltech), Russia. All rights reserved.
                2023-present Artificial Intelligence Research Institute
                              (AIRI), Russia. All rights reserved.

 NNTile is software framework for fast training of big neural networks on
 distributed-memory heterogeneous systems with GPU acceleration.

 @file docs/torch_nntile.md
 @version 2.0.0
-->

# torch_nntile (NNTile 2.0)

PyTorch **PrivateUse1** backend registered as `device="nntile"`, backed by
**PyTorch autograd**. Builds link **libnntile**, which runs on **NNHaul** with
chunk-addressable kernels on one CUDA GPU.

NNTile 2.0 has one Python surface:

- `torch_nntile.nn` — classic NNTile kernels and models:
  - `torch_nntile.nn.functional` — autograd functions (`gemm`, `sdpa_*`,
    `cross_entropy`, `layer_norm`, `rope`, …)
  - `torch_nntile.nn.module` — `torch.nn.Module` subclasses
  - `torch_nntile.nn.model` — full models (`GPT2LMHead`, `Llama`, `BERT`, …)

Stock `torch.nn` / `torch.nn.functional` on `device="nntile"` is **not
supported**: NNTile 2.0 has one torch_nntile stack, the classic kernels.
CPU/macOS wheels and flash SDPA are also gone.

## Build

```bash
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake --preset nnhaul -DCMAKE_PREFIX_PATH="$TORCH_PREFIX"
cmake --build --preset nnhaul -j
```

The build needs CUDA, cuBLAS and LibTorch. It produces
`libnntile.so`, `libnnhaul.so`, `libtorch_nntile.so`, and a local
`torch_nntile` wheel under `build-nnhaul/wheelhouse`.

## Train a model

```python
import os
os.environ["NNTILE_ENABLE_LOCAL_COMPILER"] = "1"

import torch
import torch_nntile

torch_nntile.init_context(ncpu=0, ncuda=1)
x = torch.randn(4, 8).to("nntile")
w = torch.randn(8, 16).to("nntile")
y = torch_nntile.nn.functional.gemm(x, w, ndim=1)
torch_nntile.compile_graph()
torch_nntile.run()
torch_nntile.wait()
```

The classic GPT-2 training example is
[`torch_nntile/examples/train_gpt2.py`](../torch_nntile/examples/train_gpt2.py).

## Migration from 1.x

NNTile 1.x stock-`torch.nn`/HF-on-`nntile`, CPU, macOS, and flash-SDPA
paths are not available in 2.0. Stay at the `1.1.0` tag for those products.
