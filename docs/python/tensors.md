# Tensors: types, construction, and I/O

Import from `nntile.tensor` (re-exports constructors and all functions).

## Tensor types

Defined in `nntile.tensor` and exposed via `nntile.tensor`:

| Class | Role |
|-------|------|
| `Tensor_fp32` | Standard single precision |
| `Tensor_fp64` | Double precision |
| `Tensor_fp16` | Half precision |
| `Tensor_bf16` | Bfloat16 |
| `Tensor_fp32_fast_tf32` | FP32 compute with TF32 |
| `Tensor_fp32_fast_fp16` | FP32 storage, FP16 math |
| `Tensor_fp32_fast_bf16` | FP32 storage, BF16 math |
| `Tensor_int64` | Indices (e.g. embedding) |
| `Tensor_bool` | Masks |

`TensorTraits(shape, basetile_shape)` describes global shape and **basetile**
tiling. `mpi_distr` lists tile ownership (default `[0] * grid.nelems` for single-node).

`TensorMoments` wraps `value`, optional `grad`, and `grad_required` for autodiff-style training.

## Constructors

From [`utils/constructors.py`](../../wrappers/python/nntile/utils/constructors.py):

| Function | Description |
|----------|-------------|
| `empty(shape, basetile_shape=None, dtype=Tensor_fp32, mpi_distr=None)` | Uninitialized tensor |
| `empty_like(A)` | Same layout as `A` |
| `zeros` / `zeros_like` | Allocate and `clear_async` |
| `ones` / `ones_like` | Fill with 1 |
| `full` / `full_like` | Fill with scalar via `fill_async` |
| `from_array(array, basetile_shape=None, mpi_distr=None)` | NumPy → NNTile (dtype from array) |
| `clone(A)` | `empty_like` + `copy_async` |

### Direct construction

```python
from nntile.tensor import Tensor_fp32, TensorTraits

shape = [128, 768]
basetile = [128, 768]
traits = TensorTraits(shape, basetile)
t = Tensor_fp32(traits)  # optional mpi_distr=[0, 0, ...]
```

## Reading and writing host data

| Method / function | Description |
|-----------------|-------------|
| `t.from_array(numpy_array)` | Copy host buffer into tensor (C-order) |
| `t.to_array(numpy_array)` | Copy tensor to preallocated host buffer (blocking) |
| `to_numpy(t)` | Allocate NumPy array and fill (blocking) |
| `to_numpy_async(t)` | Async gather + poll until ready |
| `gather_async(local, gathered)` | MPI gather for multi-tile layouts |

NumPy dtype mapping: `np2nnt_type_mapping` / `nnt2np_type_mapping` in constructors
(e.g. `float32` → `Tensor_fp32`, `int64` → `Tensor_int64`).

### Example

```python
import numpy as np
from nntile.tensor import from_array, to_numpy, TensorTraits

x_np = np.random.randn(64, 128).astype(np.float32)
x = from_array(x_np, basetile_shape=[64, 128])

import nntile
nntile.starpu.wait_for_all()
y_np = to_numpy(x)
```

## GEMM transpose flags

`TransOp`, `trans`, `notrans` from `nntile` select transpose mode for `gemm_async`.

## See also

- [functions.md](functions.md) — operations on tensors
- [layers.md](layers.md) — `TensorMoments` in layers
