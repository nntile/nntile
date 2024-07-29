/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/conv2d.cc
 * StarPU wrappers for 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * */

#include "nntile/starpu/conv2d.hh"
#include "../testing.hh"
#include "nntile/kernel/conv2d.hh"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace nntile;
using namespace nntile::starpu;

template <typename T>
void validate_cpu()
{
    using Y = typename T::repr_t;
    Index nx = 20, ny = 21, mx = 3, my = 4, kx = nx + mx - 1, ky = ny + my - 1;
    Index offset_x = 1, offset_y = 2;
    Index batch = 3, out_channels = 4, in_channels = 5;
    Index padding_n = 6, limit_n = nx - 7, padding_m = 8, limit_m = ny - 9;

    Index src_size = nx * ny * in_channels * batch;
    Index kernel_size = mx * my * in_channels * out_channels;
    Index dst_size = kx * ky * out_channels * batch;
    // Init all the data
    std::vector<T> src(src_size);
    for(Index i = 0; i < nx * ny; ++i)
    {
        src[i] = Y(1);
    }
    std::vector<T> kernel(kernel_size);
    for(Index i = 0; i < mx * my; ++i)
    {
        kernel[i] = Y(1);
    }
    std::vector<T> dst(dst_size);
    for(Index i = 0; i < (nx + mx - 1) * (ny + my - 1); ++i)
    {
        dst[i] = Y(0);
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    std::cout << "Run kernel::conv2d::cpu<" << T::type_repr << ">\n";
    kernel::conv2d::cpu<T>(offset_x, offset_y, batch, out_channels, in_channels,
                           0, nx, 0, ny, nx, ny, &src[0], mx, my, &kernel[0],
                           kx, ky, &dst[0]);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T) * src_size, STARPU_R),
        kernel_handle(&kernel[0], sizeof(T) * kernel_size, STARPU_R),
        dst2_handle(&dst2[0], sizeof(T) * dst_size, STARPU_W);
    conv2d::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::conv2d::submit<" << T::type_repr << "> restricted to CPU\n";
    conv2d::submit<T>(offset_x, offset_y, batch, out_channels, in_channels, 0,
                      nx, 0, ny, nx, ny, src_handle, mx, my, kernel_handle, kx,
                      ky, dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < dst_size; ++i)
    {
        TEST_ASSERT(Y(dst[i]) == Y(dst2[i]));
    }
    std::cout << "OK: starpu::conv2d::submit<" << T::type_repr << "> restricted to CPU\n";
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 1, 0);
    // Init codelet
    conv2d::init();
    // Launch all tests
    validate_cpu<fp32_t>();
    validate_cpu<fp64_t>();

    return 0;
}
