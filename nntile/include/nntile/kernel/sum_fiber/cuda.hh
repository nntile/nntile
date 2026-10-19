/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sum_fiber/cuda.hh
 * Sums over slices into a fiber of a buffer on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::sum_fiber
{

// Sums over slices along the first and last axes into a fiber of a tensor
template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index batch, Scalar alpha,
        const T *src, Scalar beta, T *dst)
    noexcept;

//! Sum fibers of a chunked tensor on CUDA
/*! Same operation as cuda() above, but the source's (and destination's)
 * flat element ranges are backed by a chain of equal-size chunk
 * allocations (a shorter tail on the last chunk). The whole reduction is
 * issued as ONE kernel launch: the two chunk base-pointer tables are read
 * from the device chunk-pointer ring (no per-tile or handle cap), and
 * every element derives its chunk index from its own flat offset (chunk =
 * offset / elems, in-chunk offset = offset % elems; a shift/mask when elems
 * is resolved by the class shift/mask). The kernel reproduces cuda()'s thread geometry and
 * per-thread accumulation order (i1 outer, i0 inner), so each thread sums
 * the same values in the same order and the shared-memory atomicAdd sees
 * the same partial sums: the result is cuda()'s, up to the same atomic
 * ordering nondeterminism cuda() itself already has. A launch whose pointer
 * set exceeds the ring falls back to the per-piece driver.
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of src
 * @param[in] n: Size of the last mode of src
 * @param[in] k: Size of the middle mode of src and the only mode of dst
 * @param[in] batch: Size of the batch dimension
 * @param[in] alpha: Scaling factor for src
 * @param[in] src_chunks: Device-ring view of src's chunk base pointers
 * @param[in] nsrc: Number of src chunks
 * @param[in] beta: Scaling factor for dst
 * @param[in] dst_chunks: Device-ring view of dst's chunk base pointers
 * @param[in] ndst: Number of dst chunks
 * @param[in] elems_src: Elements in a full src chunk (uniform)
 * @param[in] elems_dst: Elements in a full dst chunk (uniform)
 * */
template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T* const* src_chunks, Index nsrc, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::sum_fiber
