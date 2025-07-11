/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/norm_fiber.cc
 * Euclidean norms over slices into a fiber of a product of buffers
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/norm_fiber.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::norm_fiber;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index m, Index n, Index k, Index batch, Scalar alpha,
        const std::vector<T> &src1, Scalar beta, const std::vector<T> &src2, std::vector<T> &dst)
{
    // Copy to device
    T *dev_src1, *dev_src2, *dev_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_src1, sizeof(T)*m*n*k*batch);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*k*batch);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_src2, sizeof(T)*k*batch);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src1, &src1[0], sizeof(T)*m*n*k*batch,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*k*batch,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src2, &src2[0], sizeof(T)*k*batch,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);

    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level kernel
    cuda<T>(stream, m, n, k, batch, alpha, dev_src1, beta, dev_src2, dev_dst);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*k*batch,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src1);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src2);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index m, Index n, Index k, Index batch, Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    const Y eps = T::epsilon;
    // Init test input
    std::vector<T> src1(m*n*k*batch);
    std::vector<T> src2(k*batch);
    std::vector<T> dst(k*batch);
    T *src_pointer = &src1[0];
    for(Index b = 0; b < batch; ++b) {
        for(Index i2 = 0; i2 < k; ++i2)
        {
            dst[b*batch+i2] = Y{1.0};
            src2[b*batch+i2] = Y{10.0};
            for(Index i1 = 0; i1 < n; ++i1)
            {
                T *src_slice = src_pointer + ((i1+b*n)*k+i2)*m;
                for(Index i0 = 0; i0 < m; ++i0)
                {
                    src_slice[i0] = Y{1.0};
                }
            }
        }
    }

    constexpr Y zero{0.0}, one{1.0};
    std::vector<T> dst_copy(dst);
    // Check low-level kernel
    std::cout << "OK: kernel::norm_fiber::cpu<" << T::short_name << ">\n";
    cpu<T>(m, n, k, batch, alpha, &src1[0], beta, &src2[0], &dst[0]);
    Y ref = std::sqrt(m*n + beta*10.0*10.0);
    for(Index i = 0; i < dst.size(); ++i)
    {
        Y val{dst[i]};
        if(ref == Y{0})
        {
            TEST_ASSERT(std::abs(val) <= 10*eps);
        }
        else
        {
            TEST_ASSERT(std::abs(val/ref-Y{1}) <= 10*eps);
        }
    }
    std::cout << "OK: kernel::norm_fiber::cpu<" << T::short_name << ">\n";

#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    std::vector<T> dst_cuda(dst_copy);
    std::cout << "Run kernel::norm_fiber::cuda<" << T::short_name << ">\n";
    run_cuda<T>(m, n, k, batch, alpha, src1, beta, src2, dst_cuda);
    for(Index i = 0; i < dst_cuda.size(); ++i)
    {
        Y val_cuda{dst_cuda[i]};
        if(ref == Y{0})
        {
            TEST_ASSERT(std::abs(val_cuda) <= 10*eps);
        }
        else
        {
            TEST_ASSERT(std::abs(val_cuda/ref-Y{1}) <= 10*eps);
        }
    }
    std::cout << "OK: kernel::norm_fiber::cuda<" << T::short_name << ">\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    validate<fp64_t>(32, 32, 10, 1, 1.0, 1.0);
    validate<fp64_t>(32, 9, 10, 1, 1.0, 1.0);
    validate<fp32_t>(32, 32, 10, 1, 1.0, 1.0);
    validate<fp32_t>(32, 9, 10, 1, 1.0, 1.0);
    validate<fp64_t>(32, 32, 10, 1, 1.0, 0.0);
    validate<fp64_t>(32, 9, 10, 1, 1.0, 0.0);
    validate<fp32_t>(32, 32, 10, 1, 1.0, 0.0);
    validate<fp32_t>(32, 9, 10, 1, 1.0, 0.0);
    return 0;
}
