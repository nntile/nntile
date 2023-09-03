/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/add.cc
 * Per-element addition of tensors
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Konstantin Sozykin
 * @date 2023-09-03
 * */

#include "nntile/kernel/add.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::add;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index nelems, T alpha, const std::vector<T> &src, T beta, std::vector<T> &dst)
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
    cuda<T>(stream, nelems, alpha, dev_src, beta, dev_dst);
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
void validate(Index nelems, Index test_index)
{
    constexpr T eps = 2 * std::numeric_limits<T>::epsilon();
    // Init test input
    T alpha = (1.0)/T(test_index); 
    T beta = (1.0)/T(test_index);
    std::vector<T> src(nelems), dst(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src[i] = T(2*i+1-nelems);
        dst[i] = T(2*nelems-i);
    }
    std::vector<T> dst_save(dst);
    // Check low-level CPU kernel
    std::cout << "Run kernel::add::cpu<T>\n";
    cpu<T>(nelems, alpha, &src[0], beta, &dst[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        T val_ref = alpha*T(2*i+1-nelems) + beta*T(2*nelems-i);
        TEST_ASSERT(std::abs(dst[i]-val_ref) <= eps);
    }
    std::cout << "OK: kernel::add::cpu<T>\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    dst = dst_save;
    std::cout << "Run kernel::add::cuda<T>\n";
    run_cuda<T>(nelems, alpha, src, beta, dst);
    for(Index i = 0; i < nelems; ++i)
    {
        T val_ref = alpha*T(2*i+1-nelems) + beta*T(2*nelems-i);
        TEST_ASSERT(std::abs(dst[i]-val_ref) <= eps);
    }
    std::cout << "OK: kernel::add::cuda<T>\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    const Index test_nelems[] = { 0, 3, 8888 };
    for(Index i = 0; i < 3; ++i)
    {
        validate<fp32_t>(test_nelems[i],i);
        validate<fp64_t>(test_nelems[i],i);
        
        validate<fp32_t>(test_nelems[i],-i);
        validate<fp64_t>(test_nelems[i],i);
        
        validate<fp32_t>(test_nelems[i],i);
        validate<fp64_t>(test_nelems[i],-i);
        
        validate<fp32_t>(test_nelems[i],-i);
        validate<fp64_t>(test_nelems[i],-i);
    }
}

