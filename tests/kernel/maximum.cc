/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/maximum.cc
 * Per-element maximum of two buffers
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-05-05
 * */

#include "nntile/kernel/maximum.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::maximum;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index nelems, const std::vector<T> &src, std::vector<T> &dst)
{
    // // Copy to device
    // T *dev_src, *dev_dst;
    // cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(T)*nelems);
    // TEST_ASSERT(cuda_err == cudaSuccess);
    // cuda_err = cudaMalloc(&dev_dst, sizeof(T)*nelems);
    // TEST_ASSERT(cuda_err == cudaSuccess);
    // cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*nelems,
    //         cudaMemcpyHostToDevice);
    // TEST_ASSERT(cuda_err == cudaSuccess);
    // cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*nelems,
    //         cudaMemcpyHostToDevice);
    // TEST_ASSERT(cuda_err == cudaSuccess);
    // // Init stream
    // cudaStream_t stream;
    // cuda_err = cudaStreamCreate(&stream);
    // TEST_ASSERT(cuda_err == cudaSuccess);
    // // Launch low-level CUDA kernel
    // cuda<T>(stream, nelems, dev_src, dev_dst);
    // cuda_err = cudaStreamSynchronize(stream);
    // TEST_ASSERT(cuda_err == cudaSuccess);
    // // Copy result and deallocate device memory
    // cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*nelems,
    //         cudaMemcpyDeviceToHost);
    // TEST_ASSERT(cuda_err == cudaSuccess);
    // cuda_err = cudaFree(dev_src);
    // TEST_ASSERT(cuda_err == cudaSuccess);
    // cuda_err = cudaFree(dev_dst);
    // TEST_ASSERT(cuda_err == cudaSuccess);
    // cuda_err = cudaStreamDestroy(stream);
    // TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index nelems)
{
    constexpr T eps = 2 * std::numeric_limits<T>::epsilon();
    // Init test input
    std::vector<T> src(nelems), dst(nelems);
    T sign_factor = T(-1.);
    for(Index i = 0; i < nelems; ++i)
    {
        src[i] = T(2*i+1-nelems) * sign_factor;
        dst[i] = T(nelems-i);
        sign_factor *= T(-1.);
    }
    std::vector<T> dst_save(dst);
    // Check low-level CPU kernel
    std::cout << "Run kernel::max::cpu<T>\n";
    cpu<T>(nelems, &src[0], &dst[0]);
    T ref_val;
    for(Index i = 0; i < nelems; ++i)
    {
        if (dst_save[i] <= src[i])
            ref_val = src[i];
        else
            ref_val = dst_save[i];
        // NaN-aware comparisons
        TEST_ASSERT(dst[i] == ref_val);
    }
    std::cout << "OK: kernel::max::cpu<T>\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    // dst = dst_save;
    // std::cout << "Run kernel::prod::cuda<T>\n";
    // run_cuda<T>(nelems, src, dst);
    // for(Index i = 0; i < nelems; ++i)
    // {
    //     T x = dst_save[i];
    //     T val_ref = T((2*i+1-nelems)*(nelems-i)) / T{1000000};
    //     // Obtain range of correct values
    //     T val_ref_min, val_ref_max;
    //     if(val_ref < 0)
    //     {
    //         val_ref_min = val_ref * (T{1}+eps) - eps;
    //         val_ref_max = val_ref * (T{1}-eps) + eps;
    //     }
    //     else
    //     {
    //         val_ref_min = val_ref * (T{1}-eps) - eps;
    //         val_ref_max = val_ref * (T{1}+eps) + eps;
    //     }
    //     // NaN-aware comparisons
    //     TEST_ASSERT(dst[i] >= val_ref_min and dst[i] <= val_ref_max);
    // }
    // std::cout << "OK: kernel::prod::cuda<T>\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    validate<fp32_t>(0);
    validate<fp32_t>(1);
    validate<fp32_t>(80000);
    validate<fp64_t>(0);
    validate<fp64_t>(1);
    validate<fp64_t>(80000);
    return 0;
}
