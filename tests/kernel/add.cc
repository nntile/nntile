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
 * @version 1.0.0
 * */

#include "nntile/kernel/add.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>

#ifdef NNTILE_USE_CUDA
//#include <cuda_fp16.h>
#endif // NNTILE_USE_CUDA

using namespace nntile;
using namespace nntile::kernel;

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

//void run_cuda16(Index nelems, fp32_t alpha, const std::vector<fp16_t> &src, fp32_t beta, std::vector<fp16_t> &dst)
//{
//    // Copy to device
//    fp16_t *dev_src, *dev_dst;
//    cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(fp16_t)*nelems);
//    TEST_ASSERT(cuda_err == cudaSuccess);
//    cuda_err = cudaMalloc(&dev_dst, sizeof(fp16_t)*nelems);
//    TEST_ASSERT(cuda_err == cudaSuccess);
//    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(fp16_t)*nelems,
//            cudaMemcpyHostToDevice);
//    TEST_ASSERT(cuda_err == cudaSuccess);
//    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(fp16_t)*nelems,
//            cudaMemcpyHostToDevice);
//    TEST_ASSERT(cuda_err == cudaSuccess);
//    // Init stream
//    cudaStream_t stream;
//    cuda_err = cudaStreamCreate(&stream);
//    TEST_ASSERT(cuda_err == cudaSuccess);
//    // Launch low-level CUDA kernel
//    add::cuda16(stream, nelems, alpha, dev_src, beta, dev_dst);
//    cuda_err = cudaStreamSynchronize(stream);
//    TEST_ASSERT(cuda_err == cudaSuccess);
//    // Copy result and deallocate device memory
//    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(fp16_t)*nelems,
//            cudaMemcpyDeviceToHost);
//    TEST_ASSERT(cuda_err == cudaSuccess);
//    cuda_err = cudaFree(dev_src);
//    TEST_ASSERT(cuda_err == cudaSuccess);
//    cuda_err = cudaFree(dev_dst);
//    TEST_ASSERT(cuda_err == cudaSuccess);
//    cuda_err = cudaStreamDestroy(stream);
//    TEST_ASSERT(cuda_err == cudaSuccess);
//}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index nelems, int test_index_a, int test_index_b)
{
    constexpr T eps = 20 * std::numeric_limits<T>::epsilon();
    // Init test input
    T alpha = (1.0)/T(test_index_a);
    T beta = (1.0)/T(test_index_b);
    std::vector<T> src(nelems), dst(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src[i] = T(2*i+1-nelems);
        dst[i] = T(2*nelems-i);
    }
    std::vector<T> dst_save(dst);
    // Check low-level CPU kernel
    std::cout << "Run kernel::add::cpu<T>\n";
    add::cpu<T>(nelems, alpha, &src[0], beta, &dst[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        T val_ref = alpha*T(2*i+1-nelems) + beta*T(2*nelems-i);
        TEST_ASSERT(std::abs(dst[i]-val_ref)/std::abs(val_ref) <= eps);
    }
    //std::cout << "OK: kernel::add::cpu<" << typeid(T).name() << ">\n";
    std::cout << "OK: kernel::add::cpu<T>\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    dst = dst_save;
    std::cout << "Run kernel::add::cuda<T>\n";
    run_cuda<T>(nelems, alpha, src, beta, dst);
    for(Index i = 0; i < nelems; ++i)
    {
        T val_ref = alpha*T(2*i+1-nelems) + beta*T(2*nelems-i);
        TEST_ASSERT(std::abs(dst[i]-val_ref)/std::abs(val_ref) <= eps);
    }
    std::cout << "OK: kernel::add::cuda<T>\n";
#endif // NNTILE_USE_CUDA
}

//template<>
//void validate<fp16_t>(Index nelems, int test_index_a, int test_index_b)
//
//{
//#ifdef NNTILE_USE_CUDA
//    constexpr fp32_t eps = 0.001;
//    // Init test input
//    fp32_t alpha = (1.0)/fp32_t(test_index_a);
//    fp32_t beta = (1.0)/fp32_t(test_index_b);
//    std::vector<__half> src_half(nelems), dst_half(nelems);
//    for(Index i = 0; i < nelems; ++i)
//    {
//        src_half[i] = __float2half(fp32_t(2*i+1-nelems));
//        dst_half[i] = __float2half(fp32_t(2*nelems-i));
//    }
//
//    std::vector<fp16_t> &src16 = * reinterpret_cast<std::vector<fp16_t> *>(&src_half);
//    std::vector<fp16_t> &dst16 = * reinterpret_cast<std::vector<fp16_t> *>(&dst_half);
//    // Check low-level CUDA kernel
//    std::cout << "Run kernel::add::cuda<fp16_t>\n";
//    run_cuda16(nelems, alpha, src16, beta, dst16);
//    std::vector<__half> &dst_result_half = * reinterpret_cast<std::vector<__half> *>(&dst16);
//    for(Index i = 0; i < nelems; ++i)
//    {
//        fp32_t val_ref = alpha*fp32_t(2*i+1-nelems) + beta*fp32_t(2*nelems-i);
//        fp32_t val_dst32 = __half2float(dst_result_half[i]);
//        std::cout << val_ref <<"\t"<< val_dst32  << "\n";
//        TEST_ASSERT(std::abs(val_dst32-val_ref) <= eps);
//    }
//    std::cout << "OK: kernel::add::cuda<fp16_t>\n";
//#endif // NNTILE_USE_CUDA
//}

int main(int argc, char **argv)
{
    const Index test_nelems[] = { 0, 3, 999 };
    for(Index j = 0; j < 3; ++j)
    {
        Index nelems = test_nelems[j];
        int i = int(j)+1;
        validate<fp64_t>(nelems,i,i); // run with alhpa == 1/i and beta == 1/i etc
        //validate<fp64_t>(nelems,i,-i);
        //validate<fp64_t>(nelems,-i,i);
        //validate<fp64_t>(nelems,-i,-i);

        validate<fp32_t>(nelems,i,i);
        //validate<fp32_t>(nelems,i,-i);
        //validate<fp32_t>(nelems,-i,i);
        //validate<fp32_t>(nelems,-i,-i);

        i = 1;
        //validate<fp16_t>(nelems,i,i);
        //validate<fp16_t>(nelems,i,-i);
        //validate<fp16_t>(nelems,-i,i);
        //validate<fp16_t>(nelems,-i,-i);
    }
}
