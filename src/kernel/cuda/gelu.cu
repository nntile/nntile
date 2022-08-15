/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cuda/gelu.cu
 * GeLU operation on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-15
 * */

#include "nntile/kernel/cuda/gelu.hh"

namespace nntile
{
namespace kernel
{
namespace cuda
{

template<typename T>
static __global__
void gelu_kernel(Index nelems, T *data)
{
    int start = threadIdx.x + blockIdx.x*blockDim.x,
        step = blockDim.x * gridDim.x;
    constexpr T mone = -1, pt5 = 0.5;
    const T f1 = mone / std::sqrt(T{2.0});
    for(Index i = start; i < nelems; i += step)
    {
        T z = data[i];
        T y = erfc(f1 * z);
        data[i] = pt5 * z * y;
    }
}

template<typename T>
void gelu(cudaStream_t stream, Index nelems, T *data)
    noexcept
//! Inplace GeLU operation
/*! Uses very slow std::erfc() function, so consider using approximated version
 * nntile::kernel::cpu::gelutanh(). Does the following per-element operation:
 * GeLU(z) = 0.5 z erfc(-z/sqrt(2))
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data: Buffer to apply GeLU
 * */
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    dim3 blocks(256), threads(32);
    (gelu_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, src, dst);
}

// Explicit instantiation
template
void gelu<fp32_t>(cudaStream_t stream, Index nelems, fp32_t *data)
    noexcept;

template
void gelu<fp64_t>(cudaStream_t stream, Index nelems, fp64_t *data)
    noexcept;

} // namespace cuda
} // namespace kernel
} // namespace nntile

