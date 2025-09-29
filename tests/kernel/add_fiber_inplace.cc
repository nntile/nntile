/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/add_fiber_inplace.cc
 * Per-element addition of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/add_fiber_inplace.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>
#include "nntile/kernel/cpu.hh"
#include "nntile/kernel/cuda.hh"

#ifdef NNTILE_USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif // NNTILE_USE_CUDA

using namespace nntile;
using namespace nntile::kernel;
using namespace nntile::kernel::add_fiber_inplace;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index m, Index n, Index k, Index batch, Scalar alpha, const std::vector<T> &src,
        Scalar beta, std::vector<T> &dst)
{
    // Copy to device
    T *dev_src, *dev_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(T)*k*batch);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n*k*batch);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, src.data(), sizeof(T)*k*batch,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, dst.data(), sizeof(T)*m*n*k*batch,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level CUDA kernel
    cuda<T>(stream, m, n, k, batch, alpha, dev_src, beta, dev_dst);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(dst.data(), dev_dst, sizeof(T)*m*n*k*batch,
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
void validate(Index m, Index n, Index k, Index batch, int test_index_a, int test_index_b)
{
    using Y = typename T::repr_t;
    const Y eps = 2 * T::epsilon;
    // Init test input
    Scalar alpha = (1.0)/Scalar(test_index_a);
    Scalar beta = (1.0)/Scalar(test_index_b);
    std::vector<T> src(k*batch);
    std::vector<T> dst(m*n*k*batch);
    for(Index i = 0; i < k*batch; ++i)
    {
        src[i] = Y(2*i+1-k*batch);
    }
    for(Index i = 0; i < m*n*k*batch; ++i)
    {
        dst[i] = Y(5*m*n*k*batch-2*i);
    }
    std::vector<T> dst_save(dst);
    // Check CPU kernel
    std::cout << "Run kernel::add_fiber_inplace::cpu<" << T::short_name << ">\n";
    cpu<T>(m, n, k, batch, alpha, src.data(), beta, dst.data());
    for(Index b = 0; b < batch; ++b)
    {
        for(Index i2 = 0; i2 < k; ++i2)
        {
            Y src_val = Y{src[i2+b*k]};
            for(Index i1 = 0; i1 < n; ++i1)
            {
                for(Index i0 = 0; i0 < m; ++i0)
                {
                    Index l = ((i1+b*n)*k+i2)*m+i0;
                    Y val_ref = alpha*src_val + beta*Y{dst_save[l]};
                    Y val = Y{dst[l]};
                    Y abs_error = std::abs(val - val_ref);
                    Y val_ref_abs = std::abs(val_ref);
                    Y tol = eps * std::max(Y{1.0}, val_ref_abs);
                    TEST_ASSERT(abs_error <= tol);
                }
            }
        }
    }
    std::cout << "OK: kernel::add_fiber_inplace::cpu<" << T::short_name << ">\n";
#ifdef NNTILE_USE_CUDA
    // Check CUDA kernel
    dst = dst_save;
    std::cout << "Run kernel::add_fiber_inplace::cuda<" << T::short_name << ">\n";
    run_cuda<T>(m, n, k, batch, alpha, src, beta, dst);
    for(Index b = 0; b < batch; ++b)
    {
        for(Index i2 = 0; i2 < k; ++i2)
        {
            Y src_val = Y{src[i2+b*k]};
            for(Index i1 = 0; i1 < n; ++i1)
            {
                for(Index i0 = 0; i0 < m; ++i0)
                {
                    Index l = ((i1+b*n)*k+i2)*m+i0;
                    Y val_ref = alpha*src_val + beta*Y{dst_save[l]};
                    Y val = Y{dst[l]};
                    Y abs_error = std::abs(val - val_ref);
                    Y val_ref_abs = std::abs(val_ref);
                    Y tol = eps * std::max(Y{1.0}, val_ref_abs);
                    TEST_ASSERT(abs_error <= tol);
                }
            }
        }
    }
    std::cout << "OK: kernel::add_fiber_inplace::cuda<" << T::short_name << ">\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    const Index test_m[] = {0, 1, 5};
    const Index test_n[] = {0, 1, 3};
    const Index test_k[] = {0, 1, 10};
    const Index test_batch[] = {0, 1, 4};
    int ti = 1;
    for(Index m: test_m)
    {
        for(Index n: test_n)
        {
            for(Index k: test_k)
            {
                for(Index batch: test_batch)
                {
                    validate<fp32_t>(m, n, k, batch, ti, ti+1);
                    validate<fp64_t>(m, n, k, batch, ti, ti+1);
                    validate<bf16_t>(m, n, k, batch, ti, ti+1);
                    validate<fp16_t>(m, n, k, batch, ti, ti+1);
                    ++ti;
                }
            }
        }
    }
}