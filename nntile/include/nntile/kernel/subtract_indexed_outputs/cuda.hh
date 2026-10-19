/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/subtract_indexed_outputs/cuda.hh
 * subtract_indexed_outputs operation for buffers on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::subtract_indexed_outputs
{

template<typename T>
void cuda(cudaStream_t stream, Index n_labels, Index n_outputs, Index ignore_index,
          Scalar val, const int64_t *labels, T *dst)
    noexcept;

//! subtract_indexed_outputs over chunked operands on CUDA
/*! Same operation as cuda() above, but each operand's flat element range is
 * backed by a chain of equal-size chunk allocations (a shorter, exact-size
 * tail on the last chunk). The whole op is ONE kernel: the chunk
 * base-pointer arrays are the device-ring views of device-ring arrays built
 * by the driver, and each block copies them into shared memory once. Output
 * column i's actions (the ignore_index zero-fill, or the single
 * dst[labels[i], i] -= val) are entirely column-local, so the result is
 * bitwise cuda()'s under any chunking.
 *
 * @param[in] stream: CUDA stream
 * @param[in] n_labels: Number of possible labels
 * @param[in] n_outputs: Number of columns
 * @param[in] ignore_index: Label value whose column is zeroed
 * @param[in] val: Value subtracted from the indexed element
 * @param[in] labels_chunks: Device view of the host labels pointer array
 * @param[in] nlab: Number of labels chunks
 * @param[in] dst_chunks: Device view of the host dst pointer array
 * @param[in] ndst: Number of dst chunks
 * @param[in] elems_lab: Labels per full labels chunk (uniform)
 * @param[in] elems_dst: Elements in a full dst chunk (uniform)
 */
template<typename T>
void chunked(cudaStream_t stream, Index n_labels, Index n_outputs,
        Index ignore_index, Scalar val,
        const int64_t *const *labels_chunks, Index nlab,
        T *const *dst_chunks, Index ndst, Index elems_lab, Index elems_dst,
        const int64_t *labels0 = nullptr)
    noexcept;



} // namespace nntile::kernel::subtract_indexed_outputs
