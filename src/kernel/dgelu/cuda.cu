/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/dgelu/cuda.cu
 * Derivative of GeLU operation on a buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-24
 * */

#include "nntile/kernel/dgelu/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace dgelu
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, T *data)
{
    int start = threadIdx.x + blockIdx.x*blockDim.x,
        step = blockDim.x * gridDim.x;
    constexpr T pi = 3.141592653589793238462643383279502884L,
        one = 1, mone = -1, pt5 = 0.5;
    const T f1 = mone / std::sqrt(T{2.0}), f2 = one / std::sqrt(2*pi);
    for(Index i = start; i < nelems; i += step)
    {
        T z = data[i];
        T x = std::exp(-pt5 * z * z);
        T y = erfc(f1 * z);
        data[i] = z*f2*x + pt5*y;
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T *data)
    noexcept
//! Inplace derivative of GeLU operation performed on CUDA
/*! Uses very slow std::erfc() function, so consider using approximated version
 * nntile::kernel::dgelutanh::cuda(). Does the following per-element operation:
 * GeLU'(z) = [0.5 z erfc(-z/sqrt(2))]'
 * GeLU'(z) = 0.5 erfc(-z/sqrt(2)) + [0.5 z (1+erf(z/sqrt(2))']
 * GeLU'(z) = 0.5 erfc(-z/sqrt(2)) + z 1/sqrt(2pi) e^(-z*z/2)
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data: Buffer to apply derivative of GeLU
 * */
{
    dim3 blocks(256), threads(32);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, data);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, fp64_t *data)
    noexcept;

} // namespace dgelu
} // namespace kernel
} // namespace nntile

