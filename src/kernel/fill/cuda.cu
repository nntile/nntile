/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/fill/cuda.cu
 * Fill operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/fill/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::fill
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar val_, T *data)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y val{val_};
    if(i < nelems)
    {
        data[i] = T{val};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar val, T *data)
    noexcept
//! Fill operation on CUDA
/*! Sets all elements to the provided value
 * @params[in] nelems: Number of elements in a buffer
 * @param[in] val: Input value
 * @params[out] data: Output buffer
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, val, data);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar val, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar val, fp64_t *data)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar val, bf16_t *data)
    noexcept;

} // namespace nntile::kernel::fill
