/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/rope_backward.cc
 * Backward RoPE operation on a buffer
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/rope_backward.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::rope_backward;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index nrows, Index ncols, const std::vector<T> &sin,
    const std::vector<T> &cos, const std::vector<T> &dy, std::vector<T> &dx)
{
    T *dev_sin, *dev_cos;
    T *dev_dy, *dev_dx;
    cudaError_t cuda_err = cudaMalloc(&dev_dy, sizeof(T)*2*ncols*nrows);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dx, sizeof(T)*2*ncols*nrows);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_sin, sizeof(T)*ncols);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_cos, sizeof(T)*ncols);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dy, &dy[0], sizeof(T)*2*ncols*nrows,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dx, &dx[0], sizeof(T)*2*ncols*nrows,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_sin, &sin[0], sizeof(T)*ncols,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_cos, &cos[0], sizeof(T)*ncols,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda<T>(stream, nrows, ncols, dev_sin, dev_cos, dev_dy, dev_dx);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(&dx[0], dev_dx, sizeof(T)*2*ncols*nrows,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);

    cuda_err = cudaFree(dev_dy);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dx);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_sin);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_cos);
    TEST_ASSERT(cuda_err == cudaSuccess);

    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

template<typename T>
void validate(Index ncols, Index nrows)
{
    using Y = typename T::repr_t;
    const Y eps = 6 * T::epsilon;
    Index const num_data_elems{2*ncols*nrows};
    std::vector<T> sin(ncols);
    std::vector<T> cos(ncols);
    std::vector<T> dy(num_data_elems);
    std::vector<T> dx(num_data_elems);
    for(Index i = 0; i < num_data_elems; ++i)
    {
        dy[i] = Y(2*i+1-num_data_elems) / Y{1000};
        dx[i] = Y(2*i+1-num_data_elems) / Y{1000};
    }
    std::vector<T> dx_copy(dx);
    for(Index i = 0; i < ncols; ++i)
    {
        T const s = Y(2*i+1-ncols) / Y(ncols);
        T const c = Y(std::sqrt(1 - Y(s) * Y(s)));
        sin[static_cast<size_t>(i)] = s;
        cos[static_cast<size_t>(i)] = c;
    }

    std::cout << "Run kernel::rope_backward::cpu<" << T::short_name << ">\n";
    cpu<T>(nrows, ncols, &sin[0], &cos[0], &dy[0], &dx[0]);
    for(Index j = 0; j < nrows; ++j)
    {
        for(Index i = 0; i < ncols; ++i)
        {
            Index const l0 = 2 * (i + j * ncols);
            Index const l1 = l0 + 1;
            Y c{cos[i]}, s{sin[i]};
            Y a{dy[l0]}, b{dy[l1]};
            Y val_ref_a{c*a + s*b};
            Y val_ref_b{c*b - s*a};

            Y val_ref_a_min, val_ref_a_max;
            Y val_ref_b_min, val_ref_b_max;
            if(val_ref_a < 0)
            {
                val_ref_a_min = val_ref_a * (Y{1}+eps) - eps;
                val_ref_a_max = val_ref_a * (Y{1}-eps) + eps;
            }
            else
            {
                val_ref_a_min = val_ref_a * (Y{1}-eps) - eps;
                val_ref_a_max = val_ref_a * (Y{1}+eps) + eps;
            }

            if(val_ref_b < 0)
            {
                val_ref_b_min = val_ref_b * (Y{1}+eps) - eps;
                val_ref_b_max = val_ref_b * (Y{1}-eps) + eps;
            }
            else
            {
                val_ref_b_min = val_ref_b * (Y{1}-eps) - eps;
                val_ref_b_max = val_ref_b * (Y{1}+eps) + eps;
            }

            TEST_ASSERT(Y(dx[l0]) >= val_ref_a_min and Y(dx[l0]) <= val_ref_a_max);
            TEST_ASSERT(Y(dx[l1]) >= val_ref_b_min and Y(dx[l1]) <= val_ref_b_max);
        }
    }
    std::cout << "OK: kernel::rope_backward::cpu<" << T::short_name << ">\n";
#ifdef NNTILE_USE_CUDA
    dx = dx_copy;
    std::cout << "Run kernel::rope_backward::cuda<" << T::short_name << ">\n";
    run_cuda<T>(nrows, ncols, sin, cos, dy, dx);
    for(Index j = 0; j < nrows; ++j)
    {
        for(Index i = 0; i < ncols; ++i)
        {
            Index const l0 = 2 * (i + j * ncols);
            Index const l1 = l0 + 1;
            Y c{cos[i]}, s{sin[i]};
            Y a{dy[l0]}, b{dy[l1]};
            Y val_ref_a{c*a + s*b};
            Y val_ref_b{c*b - s*a};

            Y val_ref_a_min, val_ref_a_max;
            Y val_ref_b_min, val_ref_b_max;
            if(val_ref_a < 0)
            {
                val_ref_a_min = val_ref_a * (Y{1}+eps) - eps;
                val_ref_a_max = val_ref_a * (Y{1}-eps) + eps;
            }
            else
            {
                val_ref_a_min = val_ref_a * (Y{1}-eps) - eps;
                val_ref_a_max = val_ref_a * (Y{1}+eps) + eps;
            }

            if(val_ref_b < 0)
            {
                val_ref_b_min = val_ref_b * (Y{1}+eps) - eps;
                val_ref_b_max = val_ref_b * (Y{1}-eps) + eps;
            }
            else
            {
                val_ref_b_min = val_ref_b * (Y{1}-eps) - eps;
                val_ref_b_max = val_ref_b * (Y{1}+eps) + eps;
            }

            TEST_ASSERT(Y(dx[l0]) >= val_ref_a_min and Y(dx[l0]) <= val_ref_a_max);
            TEST_ASSERT(Y(dx[l1]) >= val_ref_b_min and Y(dx[l1]) <= val_ref_b_max);
        }
    }
    std::cout << "OK: kernel::rope_backward::cuda<" << T::short_name << ">\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    validate<fp32_t>(0, 5);
    validate<fp32_t>(1, 10);
    validate<fp32_t>(10, 100);
    validate<fp64_t>(0, 5);
    validate<fp64_t>(1, 10);
    validate<fp64_t>(10, 100);
    return 0;
}
