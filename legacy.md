# Legacy Functions Analysis

## Overview
This document contains a list of unused legacy functions identified in `wrappers/python/nntile/functions.py`. These functions are not currently being used in the main application code across the following directories:
- `wrappers/python/nntile/inference/`
- `wrappers/python/nntile/layer/`
- `wrappers/python/nntile/loss/`
- `wrappers/python/nntile/model/`
- `wrappers/python/nntile/optimizer/`

## Analysis Summary
- **Total functions in `functions.py`**: 65
- **Functions used in specified directories**: 52
- **Unused legacy functions**: 11

## Unused Legacy Functions

### 1. `add_scalar_async` **[DELETED]**
**Description**: Wrapper for multiprecision add_scalar
```python
def add_scalar_async(alpha: float, beta: float, x: Tensor) -> None:
```

**Status**: Completely removed from the codebase including:
- Python wrapper function
- C++ implementations (tile and tensor)
- C++ header files
- CMake build configurations
- Test configurations

### 2. `addcdiv_async` **[DELETED]**
**Description**: Wrapper for multiprecision addcdiv
```python
def addcdiv_async(alpha: float, eps: float, nom: Tensor, denom: Tensor, src: Tensor) -> None:
```

### 3. `hypot_async`
**Description**: Wrapper for multiprecision hypot
```python
def hypot_async(alpha: float, x: Tensor, beta: float, y: Tensor) -> None:
```

### 4. `is_tensor_of`
**Description**: Type guard for uniformly typed tensors
```python
def is_tensor_of(tensors: Sequence[Any], tensor_type: Type[T]) -> TypeGuard[Sequence[T]]:
```

### 5. `maximum_async`
**Description**: Wrapper for multiprecision elementwise maximum
```python
def maximum_async(x: Tensor, y: Tensor) -> None:
```

### 6. `norm_fiber_async`
**Description**: Wrapper for multiprecision norm_fiber
```python
def norm_fiber_async(alpha: float, x1: Tensor, beta: float, x2: Tensor, norm_fiber: Tensor, axis: int, batch_ndim: int, redux: int = 0) -> None:
```

### 7. `relu_async`
**Description**: Wrapper for multiprecision ReLU
```python
def relu_async(x: Tensor) -> None:
```

### 8. `scal_async`
**Description**: Wrapper for multiprecision scaling
```python
def scal_async(alpha: float, x: Tensor, y: Tensor) -> None:
```

### 9. `scatter_async`
**Description**: Wrapper for multiprecision scatter
```python
def scatter_async(x: TensorFloatOrInt, y: TensorFloatOrInt) -> None:
```

### 10. `sqrt_async`
**Description**: Wrapper for multiprecision square root
```python
def sqrt_async(x: Tensor, y: Tensor) -> None:
```

### 11. `sqrt_inplace_async`
**Description**: Wrapper for multiprecision inplace square root
```python
def sqrt_inplace_async(x: Tensor) -> None:
```

## Recommendations

1. **Review before removal**: Before deleting these functions, verify they are not used in other parts of the codebase (tests, examples, or other modules not covered in this analysis).

2. **Consider deprecation**: If these functions might be needed in the future, consider adding deprecation warnings before removal.

3. **Check dependencies**: Ensure that the underlying C++ implementations are also not needed elsewhere before removing the Python wrappers.

4. **Update imports**: If any of these functions are imported but not used in other files, clean up those imports as well.

## Analysis Date
This analysis was performed on the codebase as of the current date. The functions listed here should be re-evaluated if the codebase has been modified since then.
