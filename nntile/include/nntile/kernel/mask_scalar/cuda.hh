/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/mask_scalar/cuda.hh
 * Mask operation with scalar on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::mask_scalar
{

template<typename T>
void cuda(cudaStream_t stream, Index nrows, Index ncols, const bool_t *mask,
        Scalar val, T *data)
    noexcept;

//! Set masked matrix entries to a scalar on a chunked CUDA buffer
/*! Same operation as cuda() above (data is nrows-by-ncols Fortran with
 * element (i, j) at flat i + j*nrows; mask is the nrows-element row
 * operand and element (i, j) becomes val when mask[i] is false), but both
 * operands' flat element ranges are backed by a chain of equal-size chunk
 * allocations (a shorter tail on the last chunk). The WHOLE nrows-by-ncols
 * region is issued as ONE kernel launch: the two chunk base-pointer tables
 * are read from the device chunk-pointer ring (no per-tile or handle cap)
 * and each written element resolves its data chunk from its flat offset
 * f = i + j*nrows via f / elems_data, f % elems_data, and its mask chunk
 * from the row index i via i / elems_mask, i % elems_mask. Writes are
 * idempotent val assignments, so the result is BITWISE the whole-buffer
 * cuda() call. A launch whose pointer set exceeds the ring falls back to
 * the per-column driver.
 *
 * @param[in] stream: CUDA stream
 * @param[in] nrows: Number of rows of data (length of mask)
 * @param[in] ncols: Number of columns of data
 * @param[in] val: Value to set if the mask element is false
 * @param[in] data_chunks: Device-ring view of data's chunk base pointers
 * @param[in] ndata: Number of data chunks
 * @param[in] elems_data: Elements in a full data chunk (uniform)
 * @param[in] mask_chunks: Device-ring view of mask's chunk base pointers
 * @param[in] nmask: Number of mask chunks
 * @param[in] elems_mask: Elements in a full mask chunk (uniform)
 * */
template<typename T>
void chunked(cudaStream_t stream, Index nrows, Index ncols, Scalar val,
    T *const *data_chunks, Index ndata, Index elems_data,
    const bool_t *const *mask_chunks, Index nmask, Index elems_mask)
    noexcept;

} // namespace nntile::kernel::mask_scalar
