/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/softmax_inplace/cuda.hh
 * softmax_inplace operation on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::softmax_inplace
{

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *maxsumexp,
        Scalar alpha, T *dst)
    noexcept;

//! Inplace elementwise softmax over chunked operands on CUDA
/*! Same operation as cuda() above, but each operand's flat element range is
 * backed by a chain of equal-size chunk allocations (a shorter, exact-size
 * tail on the last chunk). The whole tensor is processed by ONE kernel: the
 * chunk base-pointer arrays are the device-ring views of device-ring arrays
 * built by the driver, and each block copies them into shared memory once.
 * Every element's chunk is derived from its own flat offset, so the
 * underoccupied tail needs no length array; the elementwise formula is
 * byte-for-byte the one cuda() computes, so the result is bitwise cuda()'s.
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode
 * @param[in] n: Size of the last mode
 * @param[in] k: Size of the middle mode
 * @param[in] mse_chunks: Device view of the host maxsumexp pointer array
 * @param[in] nmse: Number of maxsumexp chunks
 * @param[in] alpha: Scalar multiplier for the output
 * @param[in] dst_chunks: Device view of the host dst pointer array
 * @param[in] ndst: Number of dst chunks
 * @param[in] elems_mse: Elements in a full maxsumexp chunk (uniform)
 * @param[in] elems_dst: Elements in a full dst chunk (uniform)
 */
template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k,
        const T *const *mse_chunks, Index nmse, Scalar alpha,
        T *const *dst_chunks, Index ndst, Index elems_mse, Index elems_dst)
    noexcept;



} // namespace nntile::kernel::softmax_inplace
