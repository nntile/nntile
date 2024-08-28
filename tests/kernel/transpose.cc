/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/transpose.cc
 * Transpose operation
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/transpose.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::transpose;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index m, Index n, Scalar alpha, const std::vector<T> &src,
        std::vector<T> &dst)
{
    // Copy to device
    T *dev_src, *dev_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(T)*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level CUDA kernel
    cuda<T>(stream, m, n, alpha, dev_src, dev_dst);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*m*n,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index m, Index n)
{
    using Y = typename T::repr_t;
    const Y eps = T::epsilon();
    // Init test input
    std::vector<T> src(m*n), dst(m*n);
    for(Index i0 = 0; i0 < m*n; ++i0)
    {
        dst[i0] = Y{0.0};
    }
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            src[i1*m+i0] = Y(i0+2*i1+1) / Y{20};
        }
    }
    // Save original dst
    std::vector<T> dst_save(dst);
    // Check low-level CPU kernel
    std::cout << "Run kernel::transpose::cpu<" << T::type_repr << ">\n";
    cpu<T>(m, n, -2.0, &src[0], &dst[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Y val(dst[i0*n+i1]);
            Y val_ref = -2.0 * Y(src[i1*m+i0]);
            TEST_ASSERT(std::abs(val/val_ref-Y{1}) <= 10*eps);
        }
    }
    std::cout << "OK: kernel::transpose::cpu<" << T::type_repr << ">\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    dst = dst_save;
    std::cout << "Run kernel::transpose::cuda<" << T::type_repr << ">\n";
    run_cuda<T>(m, n, -2.0, src, dst);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Y val(dst[i0*n+i1]);
            Y val_ref = -2.0 * Y(src[i1*m+i0]);
            TEST_ASSERT(std::abs(val/val_ref-Y{1}) <= 10*eps);
        }
    }
    std::cout << "OK: kernel::transpose::cuda<" << T::type_repr << ">\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    validate<fp32_t>(1, 9);
    validate<fp32_t>(8, 9);
    validate<fp32_t>(8, 1);
    validate<fp32_t>(4, 7);
    validate<fp64_t>(1, 9);
    validate<fp64_t>(8, 9);
    validate<fp64_t>(8, 1);
    validate<fp64_t>(4, 7);
    return 0;
}
