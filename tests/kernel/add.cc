/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/add.cc
 * Per-element addition of tensors
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/add.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>
#include "nntile/kernel/cpu.hh"
#include "nntile/kernel/cuda.hh"

#ifdef NNTILE_USE_CUDA
//#include <cuda_fp16.h>
#endif // NNTILE_USE_CUDA

using namespace nntile;
using namespace nntile::kernel;

#ifdef NNTILE_USE_CUDA

template<typename T>
void run_cuda(Index nelems, Scalar alpha, const std::vector<T> &src,
        Scalar beta, std::vector<T> &dst)
{
    // Copy to device
    T *dev_src, *dev_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(T)*nelems);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*nelems);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*nelems,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*nelems,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level CUDA kernel
    add::cuda<T>(stream, nelems, alpha, dev_src, beta, dev_dst);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*nelems,
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
void validate(Index nelems, int test_index_a, int test_index_b)
{
    using Y = typename T::repr_t;
    const Y eps = 2 * T::epsilon();
    // Init test input
    Scalar alpha = (1.0)/Scalar(test_index_a);
    Scalar beta = (1.0)/Scalar(test_index_b);
    std::vector<T> src(nelems), dst(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src[i] = Y(2*i+1-nelems);
        dst[i] = Y(2*nelems-i);
    }
    std::vector<T> dst_save(dst);
    std::cout << "Run kernel::add::cpu<" << T::type_repr << ">\n";
    add::cpu<T>(nelems, alpha, &src[0], beta, &dst[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        Y val_ref = alpha*Y(2*i+1-nelems) + beta*Y(2*nelems-i);
        TEST_ASSERT(std::abs(Y{dst[i]}-val_ref)/std::abs(val_ref) <= eps);
    }
    std::cout << "OK: kernel::add::cpu<" << T::type_repr << ">\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    dst = dst_save;
    std::cout << "Run kernel::add::cuda<" << T::type_repr << ">\n";
    run_cuda<T>(nelems, alpha, src, beta, dst);
    for(Index i = 0; i < nelems; ++i)
    {
        Y val_ref = alpha*Y(2*i+1-nelems) + beta*Y(2*nelems-i);
        TEST_ASSERT(std::abs(Y{dst[i]}-val_ref)/std::abs(val_ref) <= eps);
    }
    std::cout << "OK: kernel::add::cuda<" << T::type_repr << ">\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    const Index test_nelems[] = {0, 3, 999};
    for(Index j = 0; j < 3; ++j)
    {
        Index nelems = test_nelems[j];
        int i = int(j)+1;
        validate<fp64_t>(nelems, i, i);
        validate<fp32_t>(nelems, i, i);
        validate<bf16_t>(nelems, i, i);
    }
}
