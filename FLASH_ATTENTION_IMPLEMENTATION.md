# Flash Attention Implementation for NNTile

## Overview

This document describes the implementation of flash attention wrappers for NNTile at the StarPU level. The implementation includes kernel-level functions, StarPU wrappers, and comprehensive tests following the catch2 testing framework.

## Implementation Structure

### 1. Kernel-Level Implementation

Located in `src/kernel/flash_attention/` and `include/nntile/kernel/flash_attention/`:

#### CPU Implementation (`cpu.cc`)
- **Purpose**: Reference implementation of vanilla attention mechanism
- **Algorithm**: Implements standard scaled dot-product attention
  ```
  O = softmax(Q @ K^T / scale) @ V
  ```
- **Features**:
  - Works with batch, multi-head attention
  - Numerically stable softmax using max subtraction
  - Supports fp32, fp64, fp16, bf16 types
  - Shape: `[batch, num_heads, seq_len, head_dim]`

#### CUDA Implementation (`cuda.cu`)
- **Purpose**: Placeholder for cuDNN-based flash attention
- **Status**: Stubbed out (requires CUDA toolkit and cuDNN 8.9.0+)
- **Future Integration**: 
  - Will use `cudnnMultiHeadAttnForward()` or cudnn_frontend SDPA APIs
  - Optimized for memory efficiency using tiling
  - Will support both training and inference modes

#### Headers
- `flash_attention.hh`: Main header aggregating CPU/CUDA implementations
- `cpu.hh`: CPU function declarations
- `cuda.hh`: CUDA function declarations

### 2. StarPU-Level Wrappers

Located in `src/starpu/flash_attention.cc` and `include/nntile/starpu/flash_attention.hh`:

#### Features
- **Template-based design**: `FlashAttention<std::tuple<T>>` specialization
- **Codelet registration**: Registers CPU and CUDA implementations with StarPU
- **Task submission**: `submit()` method for inserting tasks into StarPU queue
- **Footprint calculation**: Hash-based task identification for scheduling
- **Type support**: fp32, fp64, fp16, bf16, and accelerated types (tf32, fast_fp16, fast_bf16)

#### Arguments Structure
```cpp
struct args_t {
    Index batch;       // Batch size
    Index num_heads;   // Number of attention heads
    Index seq_len;     // Sequence length
    Index head_dim;    // Head dimension
    Scalar scale;      // Scaling factor (typically 1/sqrt(head_dim))
};
```

#### Buffer Layout
- Input: Q (Query), K (Key), V (Value) - Read-only
- Output: O (Output) - Write-only
- Shape: All tensors are `[batch, num_heads, seq_len, head_dim]`

### 3. Test Suite

#### Kernel Tests (`tests/kernel/flash_attention.cc`)
- **Framework**: Catch2 with template test cases
- **Coverage**:
  - Multiple data types: fp64, fp32, fp16, bf16
  - Various tensor sizes: batch=[1,2], heads=[1,2], seq_len=[4,8], head_dim=[4,8]
  - Data generation strategies: PRESET, RANDOM, IDENTITY
  - Both CPU and CUDA paths (CUDA currently skipped pending cuDNN integration)
  - Benchmarks for performance testing

- **Reference Implementation**: Uses double precision for validation
- **Tolerance**: Type-dependent epsilon (bf16: 1e-1, fp16: 1e-2, fp32: 1e-4, fp64: 1e-10)

#### StarPU Tests (`tests/starpu/flash_attention.cc`)
- **Framework**: Catch2 with StarPU integration
- **Coverage**:
  - End-to-end StarPU task submission and execution
  - Handle management and data movement
  - Multi-type support via template tests
  - Reference validation against CPU implementation

## Build System Integration

### Modified Files

1. **`src/CMakeLists.txt`**:
   - Added `kernel/flash_attention/cpu.cc` to `KERNEL_CPU_SRC`
   - Added `kernel/flash_attention/cuda.cu` to `KERNEL_CUDA_SRC`
   - Added `starpu/flash_attention.cc` to `STARPU_CODELET_SRC`

2. **`include/CMakeLists.txt`**:
   - Added kernel headers to `KERNEL_HDR`
   - Added StarPU header to `STARPU_HDR`

3. **`tests/kernel/CMakeLists.txt`**:
   - Added `flash_attention` to `TESTS` list
   - Added `flash_attention` to `TESTS_CATCH2` list

4. **`tests/starpu/CMakeLists.txt`**:
   - Added `flash_attention` to `TESTS` list

5. **`include/nntile/kernel.hh`** and **`include/nntile/starpu.hh`**:
   - Added include directives for new headers

## Algorithm Details

### Vanilla Attention (CPU Reference)

The CPU implementation follows standard attention computation:

1. **Compute Attention Scores**:
   ```
   scores[j] = (Q[i] · K[j]) * scale
   ```

2. **Stable Softmax**:
   ```
   max_score = max(scores)
   scores[j] = exp(scores[j] - max_score)
   sum_exp = sum(scores)
   scores[j] = scores[j] / sum_exp
   ```

3. **Weighted Sum**:
   ```
   O[i] = sum_j(scores[j] * V[j])
   ```

### Future cuDNN Integration (CUDA)

The CUDA implementation is prepared for cuDNN's optimized SDPA:

```cpp
// Pseudocode for future integration
cudnnHandle_t handle;
cudnnAttnDescriptor_t attnDesc;
// Setup descriptors with tensor shapes
// Configure attention parameters (scale, dropout, etc.)
cudnnMultiHeadAttnForward(handle, attnDesc, ..., Q, K, V, ..., O, ...);
```

Benefits of cuDNN implementation:
- Memory-efficient tiling (flash attention algorithm)
- Fused operations (reduces memory bandwidth)
- Optimized for various GPU architectures
- Support for causal masking and dropout

## Usage Example

```cpp
#include <nntile/starpu/flash_attention.hh>

// Initialize StarPU
starpu::Config config(1, 0, 0);

// Setup parameters
Index batch = 2;
Index num_heads = 8;
Index seq_len = 512;
Index head_dim = 64;
Scalar scale = 1.0 / std::sqrt(head_dim);

// Create and register data handles
Handle Q_handle, K_handle, V_handle, O_handle;
// ... register data with handles ...

// Submit flash attention task
flash_attention.template get<std::tuple<fp32_t>>().submit(
    batch, num_heads, seq_len, head_dim, scale,
    Q_handle, K_handle, V_handle, O_handle
);

// Wait for completion
O_handle.acquire(STARPU_R);
// ... use results ...
O_handle.release();
```

## Testing

### Run Kernel Tests
```bash
ctest -R tests_kernel_flash_attention
```

### Run StarPU Tests
```bash
ctest -R tests_starpu_flash_attention
```

### Run Benchmarks
```bash
ctest -R "tests_kernel_flash_attention.*benchmark"
```

## Limitations and Future Work

### Current Limitations
1. **No CUDA Device**: CUDA implementation is a placeholder
2. **No cuDNN**: Requires cuDNN 8.9.0+ for optimized implementation
3. **No Masking**: Causal masking not yet implemented
4. **No Dropout**: Attention dropout not implemented
5. **Memory**: CPU version not memory-optimized (loads entire attention matrix)

### Future Enhancements
1. Integrate cuDNN backend for CUDA path
2. Add causal masking support
3. Add attention dropout
4. Implement flash attention tiling on CPU
5. Add backward pass for training
6. Support for mixed-precision computation
7. Add support for variable sequence lengths
8. Optimize for specific sequence length patterns

## Technical Details

### Type System
- Uses NNTile's wrapper types (fp32_t, fp64_t, fp16_t, bf16_t)
- Automatic type conversion via `repr_t` for computation
- Fallback from accelerated types (tf32, fast_fp16, fast_bf16) to base types

### Memory Layout
- Row-major layout: `[batch, num_heads, seq_len, head_dim]`
- Contiguous memory access patterns
- Optimized for cache locality in inner loops

### Error Handling
- Uses exceptions for error reporting
- CUDA operations checked with `CUDA_CHECK` macro
- StarPU task submission failures throw `std::runtime_error`

## References

1. Flash Attention: [Dao et al., 2022](https://arxiv.org/abs/2205.14135)
2. cuDNN Documentation: NVIDIA cuDNN Library
3. StarPU Documentation: [starpu.gitlabpages.inria.fr](https://starpu.gitlabpages.inria.fr)
4. NNTile Framework: Project documentation

## Author Notes

This implementation provides a complete framework for flash attention in NNTile, with:
- ✅ Full kernel-level CPU implementation (vanilla attention as reference)
- ✅ StarPU wrapper infrastructure
- ✅ Comprehensive catch2-based tests
- ✅ Build system integration
- ⏳ CUDA/cuDNN integration (prepared but not compiled)

The CPU implementation serves as a working reference and validation baseline, while the CUDA path is ready for cuDNN integration when a CUDA-enabled environment becomes available.
