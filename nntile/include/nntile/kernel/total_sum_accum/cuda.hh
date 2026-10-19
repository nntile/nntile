/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/total_sum_accum/cuda.hh
 * total_sum_accum operation for buffers on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::total_sum_accum
{

template<typename T>
void cuda(cudaStream_t stream, Scalar alpha, Index n_labels, Index n_outputs,
        Index ignore_index, const T *logsumexp, const T *src, const int64_t *labels,
        float *val)
    noexcept;

//! total_sum_accum over chunked operands on CUDA
/*! Same operation as cuda() above, but each operand's flat element range is
 * backed by a chain of equal-size chunk allocations (a shorter, exact-size
 * tail on the last chunk). The whole op is ONE kernel: the chunk
 * base-pointer arrays are the device-ring views of device-ring arrays built
 * by the driver, and each block copies them into shared memory once. Each
 * thread's per-column read (logsumexp[i], labels[i], src[labels[i], i]) is
 * identical to cuda()'s, so the block partial and the global float
 * atomicAdds see the same values in the same block grouping: the result is
 * cuda()'s up to the same float-atomic ordering nondeterminism.
 *
 * @param[in] stream: CUDA stream
 * @param[in] alpha: Scalar multiplier
 * @param[in] n_labels: Number of possible labels
 * @param[in] n_outputs: Number of columns to sum up
 * @param[in] ignore_index: Label value to skip
 * @param[in] lse_chunks: Device view of the host logsumexp pointer array
 * @param[in] nlse: Number of logsumexp chunks
 * @param[in] src_chunks: Device view of the host src pointer array
 * @param[in] nsrc: Number of src chunks
 * @param[in] labels_chunks: Device view of the host labels pointer array
 * @param[in] nlab: Number of labels chunks
 * @param[in] val_chunks: Device view of the host val pointer array
 * @param[in] nval: Number of val chunks
 * @param[in] elems_lse: Elements in a full logsumexp chunk (uniform)
 * @param[in] elems_src: Elements in a full src chunk (uniform)
 * @param[in] elems_lab: Labels per full labels chunk (uniform)
 * @param[in] elems_val: Elements in a full val chunk (uniform)
 */
template<typename T>
void chunked(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, Index ignore_index,
        const T *const *lse_chunks, Index nlse,
        const T *const *src_chunks, Index nsrc,
        const int64_t *const *labels_chunks, Index nlab,
        float *const *val_chunks, Index nval,
        Index elems_lse, Index elems_src, Index elems_lab, Index elems_val,
        const T *lse0 = nullptr, const int64_t *labels0 = nullptr,
        float *val0 = nullptr)
    noexcept;



} // namespace nntile::kernel::total_sum_accum
