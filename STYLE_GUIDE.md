# NNTile C++ Style Guide

This document describes the C++ coding style used in the NNTile project. All code contributions should follow these conventions.

## File Headers

Every source file must start with a copyright/license header:

```cpp
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file path/to/file.hh
 * Brief description of the file
 *
 * @version 1.1.0
 * */
```

For CMakeLists.txt files, use `#` instead of `/*!`:

```cmake
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file path/to/CMakeLists.txt
# Brief description
#
# @version 1.1.0
```

## Includes

- Use quotes for local includes: `#include "nntile/tensor/tensor.hh"`
- Use angle brackets for system/standard library includes: `#include <vector>`
- Group includes logically (system headers, then local headers)

## Namespaces

- Use nested namespace syntax: `namespace nntile::tensor`
- Always close namespaces with a comment: `} // namespace nntile::tensor`
- Use `//!` for namespace documentation comments

## Naming Conventions

- **Functions**: lowercase with underscores (e.g., `gemm_check`, `gemm_async`)
- **Classes/Structs**: PascalCase (e.g., `Tensor`, `TensorTraits`, `LogicalGraph`)
- **Enums**: PascalCase for enum name, UPPER_CASE for values (e.g., `enum class OpType { GEMM, GELU }`)
- **Variables**: lowercase with underscores (e.g., `tensor_shape`, `batch_ndim`)
- **Constants**: UPPER_CASE with underscores (e.g., `MAX_DIM`)
- **Template parameters**: Single uppercase letter (e.g., `T`, `U`)

## Formatting

### Braces

- Opening braces on the same line for functions, classes, and control structures:
```cpp
void function_name()
{
    // code
}

if(condition)
{
    // code
}
```

### Indentation

- Use 4 spaces for indentation (no tabs)
- Continuation lines should align with the opening parenthesis or use 4-space indentation

### Spacing

- Spaces around operators: `a + b`, `a == b`
- Spaces after commas: `func(a, b, c)`
- No space after opening parenthesis or before closing parenthesis: `func(a, b)`
- Spaces around control flow keywords: `if(condition)`, `for(Index i = 0; i < n; ++i)`

### Function Parameters

- Align parameters when they wrap to multiple lines:
```cpp
void gemm(Scalar alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, Scalar beta,
        const Tensor<T> &C, Index ndim, Index batch_ndim)
```

### Line Length

- Maximum line length is 79 characters
- Break long lines at logical points (operators, commas, etc.)
- Indent continuation lines appropriately
- Prefer BLACK style for function calls that exceed line length:
```cpp
auto x = long_function_name(
    argument_number_1,
    argument_number_2_as_a_function(
        arg1, arg2, arg3
    )
);
```

## Comments

- Use `//!` for documentation comments (Doxygen-style)
- Use `//` for regular comments
- Place documentation comments immediately before the item being documented
- Use `/*! ... */` for multi-line documentation blocks

Example:
```cpp
//! Check if tensors match gemm
void gemm_check(const TransOp &transA, const TensorTraits &A,
        const TransOp &transB, const TensorTraits &B, const TensorTraits &C,
        Index ndim, Index batch_ndim)
```

## Control Flow

- Use braces for all control structures, even single-line bodies
- Prefer `++i` over `i++` in loops
- Use `Index` type for loop indices and dimensions

Example:
```cpp
for(Index i = 0; i < ndim; ++i)
{
    if(condition)
    {
        // code
    }
}
```

## Switch Statements

- Always include a `default` case or comment explaining why it's omitted
- Use `switch(enum_value.value)` pattern when accessing enum values
- Comment out unused cases rather than omitting them:
```cpp
switch(transA.value)
{
    case TransOp::NoTrans:
        // code
        break;
    // This parameter was already checked in gemm_check_opA_opB
    //case TransOp::Trans:
    default:
        // code
        break;
}
```

## Error Handling

- Use `std::runtime_error` or `std::invalid_argument` for errors
- Provide descriptive error messages
- Check preconditions at function entry

Example:
```cpp
if(ndim < 0)
{
    throw std::runtime_error("ndim < 0");
}
```

## Template Instantiations

- Place explicit template instantiations at the end of `.cc` files
- Use consistent formatting:
```cpp
// Explicit instantiation
template
void gemm<fp32_t>(Scalar alpha, const TransOp &transA,
        const Tensor<fp32_t> &A,
        const TransOp &transB, const Tensor<fp32_t> &B, Scalar beta,
        const Tensor<fp32_t> &C, Index ndim, Index batch_ndim);
```

## Data Types

- Use NNTile type aliases: `fp32_t`, `fp64_t`, `fp16_t`, `bf16_t`, `Index`, `Scalar`
- Support all standard data types when implementing operations (FP32, FP64, FP32_FAST_TF32, FP32_FAST_FP16, FP32_FAST_BF16, FP16, BF16)

## StarPU Integration

- **Never use raw pointers** for data that StarPU manages
- Use StarPU handles and tile acquisition/release patterns
- Operations are asynchronous by default; provide both `*_async` and blocking versions

Example:
```cpp
auto tile = tensor.get_tile(0);
auto tile_local = tile.acquire(STARPU_W);
// ... use tile_local ...
tile_local.release();
```

## GEMM Operations

- Use `gemm` (not `matmul`) for matrix multiplication operations
- Follow BLAS convention: `C = alpha * op(A) * op(B) + beta * C`
- Always include `alpha` and `beta` parameters
- Use `ndim` and `batch_ndim` parameters from the logical graph, don't hardcode them

## Code Organization

- Separate interface (`.hh`) and implementation (`.cc`) files
- Group related functions together
- Use static helper functions for internal operations
- Keep functions focused and single-purpose

## CMake Style

- Use `#` for comments in CMakeLists.txt
- Include license header at the top
- Use consistent indentation (4 spaces)
- Group related targets together
