<!--
 @copyright (c) 2022-present Skolkovo Institute of Science and Technology
                              (Skoltech), Russia. All rights reserved.
                2023-present Artificial Intelligence Research Institute
                              (AIRI), Russia. All rights reserved.

 NNTile is software framework for fast training of big neural networks on
 distributed-memory heterogeneous systems with GPU acceleration.

 @file docs/graph.md
 @version 2.0.0
-->

# Execution pipeline

NNTile 2.0 records a deferred PyTorch autograd graph, lowers it to a tensor
graph, tiles it, and executes it on NNHaul.

```text
PyTorch autograd
      │ record
      ▼
TensorGraph  ── compile ──►  TileGraph  ── lower ──►  Runtime
                                                   │
                                                   ▼
                                              NNHaul engine
                                        (chunked tasks, one GPU)
```

- `TensorGraph` stores shape/dtype/op nodes and is sealed per training phase.
- `TileGraph` splits tensor axes into tiles.
- `Runtime` submits tile operations to NNHaul. Every tile is chunk-backed;
  kernels are chunk-addressable.
- `torch_nntile.compile_graph()` / `run()` / `wait()` drive this pipeline.

There is no second backend. See
[backends.md](backends.md) and [torch_nntile.md](torch_nntile.md).
