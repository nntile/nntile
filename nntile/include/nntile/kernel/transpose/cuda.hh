/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/transpose/cuda.hh
 * Transpose operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::transpose
{

// Apply transpose for buffers on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Scalar alpha, const T* src, T* dst)
    noexcept;

// Transpose an nr-by-nc sub-block of an m-by-n buffer on CUDA
template<typename T>
void panel(cudaStream_t stream, Index m, Index n, Index nr, Index nc,
    Scalar alpha, const T* src, T* dst)
    noexcept;

//! Transpose an m-by-n buffer whose element space is stored in chunks on CUDA
/*! Same operation as cuda() above, but both operands' flat element ranges
 * are backed by a chain of equal-size chunk allocations (a shorter tail on
 * the last chunk). The whole m-by-n transpose is issued as ONE kernel
 * launch: the chunk base-pointer tables are read from the device
 * chunk-pointer ring (no per-tile or handle cap), and each element's chunk
 * index is derived from its flat offset. Elementwise and deterministic, so
 * the result is bitwise the whole-buffer cuda() call. A launch whose
 * pointer set exceeds the ring falls back to the per-panel driver.
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Number of rows of src (columns of dst)
 * @param[in] n: Number of columns of src (rows of dst)
 * @param[in] alpha: Scalar multiplier
 * @param[in] src_chunks: Device-ring view of src's chunk base pointers
 * @param[in] nsrc: Number of src chunks
 * @param[in] dst_chunks: Device-ring view of dst's chunk base pointers
 * @param[in] ndst: Number of dst chunks
 * @param[in] elems_src: Elements in a full src chunk (uniform)
 * @param[in] elems_dst: Elements in a full dst chunk (uniform)
 * */
template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Scalar alpha,
    const T* const* src_chunks, Index nsrc,
    T* const* dst_chunks, Index ndst,
    Index elems_src, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::transpose
