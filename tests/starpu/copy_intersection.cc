/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/copy_intersection.cc
 * Copy intersection of 2 StarPU buffers from one into another
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-12
 * */

#include "nntile/starpu/copy_intersection.hh"
#include "nntile/kernel/copy_intersection/cpu.hh"
#include "common.hh"
#include <array>
#include <vector>
#include <stdexcept>
#include <iostream>

using namespace nntile;

template<typename T, std::size_t NDIM>
void validate_cpu(std::array<Index, NDIM> src, std::array<Index, NDIM> dst,
        std::array<Index, NDIM> shape)
{
    // Location of copy area in source and target buffers and their shapes
    std::vector<Index> src_start(NDIM), dst_start(NDIM),
        copy_shape(shape.cbegin(), shape.cend()),
        src_shape(NDIM), dst_shape(NDIM);
    Index src_nelems = 1, dst_nelems = 1, copy_nelems = 1;
    for(Index i = 0; i < NDIM; ++i)
    {
        // Offset from the beginning
        if(src[i] >= 0)
        {
            src_start[i] = src[i];
            src_shape[i] = shape[i] + src[i];
        }
        // Offset from the end
        else
        {
            src_start[i] = 0;
            src_shape[i] = shape[i] - src[i] - 1;
        }
        src_nelems *= src_shape[i];
        // Offset from the beginning
        if(dst[i] >= 0)
        {
            dst_start[i] = dst[i];
            dst_shape[i] = shape[i] + dst[i];
        }
        // Offset from the end
        else
        {
            dst_start[i] = 0;
            dst_shape[i] = shape[i] - dst[i] - 1;
        }
        dst_nelems *= dst_shape[i];
        // Total number of elements to be copied
        copy_nelems *= shape[i];
    }
    // Strides
    std::vector<Index> src_stride(NDIM), dst_stride(NDIM);
    src_stride[0] = 1;
    dst_stride[0] = 1;
    for(Index i = 1; i < NDIM; ++i)
    {
        src_stride[i] = src_stride[i-1] * src_shape[i-1];
        dst_stride[i] = dst_stride[i-1] * dst_shape[i-1];
    }
    // Init all the data
    std::vector<T> src_data(src_nelems);
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = T(i+1);
    }
    std::vector<T> dst_data(dst_nelems);
    for(Index i = 0; i < dst_nelems; ++i)
    {
        dst_data[i] = T(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2_data(dst_data);
    // Launch low-level kernel
    std::vector<Index> tmp_index(2*NDIM);
    std::cout << "Run kernel::copy_intersection::cpu<T>\n";
    kernel::copy_intersection::cpu<T>(NDIM, &src_start[0], &src_stride[0],
            &copy_shape[0], &src_data[0], &dst_start[0], &dst_stride[0],
            &dst_data[0], &tmp_index[0]);
    // Check by actually submitting a task
    StarpuVariableHandle src_handle(&src_data[0], sizeof(T)*src_nelems),
        dst2_handle(&dst2_data[0], sizeof(T)*dst_nelems),
        tmp_handle(&tmp_index[0], sizeof(Index)*NDIM*2);
    starpu::copy_intersection::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::copy_intersection::submit<T> restricted to "
        "CPU\n";
    starpu::copy_intersection::submit<T>(NDIM, src_start, src_stride,
            dst_start, dst_stride, copy_shape, src_handle, dst2_handle,
            tmp_handle, STARPU_RW);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < dst_nelems; ++i)
    {
        if(dst_data[i] != dst2_data[i])
        {
            throw std::runtime_error("StarPU submission wrong result");
        }
    }
    std::cout << "OK: starpu::copy_intersection::submit<T> restricted to "
        "CPU\n";
}

// Run multiple tests for a given precision
template<typename T>
void validate_many()
{
    validate_cpu<T, 1>({0}, {0}, {2});
    validate_cpu<T, 1>({2}, {0}, {2});
    validate_cpu<T, 1>({-2}, {0}, {2});
    validate_cpu<T, 1>({0}, {2}, {2});
    validate_cpu<T, 1>({0}, {-2}, {2});
    validate_cpu<T, 3>({0, 0, 0}, {0, 0, 0}, {2, 3, 4});
    validate_cpu<T, 3>({1, 0, 0}, {0, 0, 0}, {2, 3, 4});
    validate_cpu<T, 3>({1, 0, 0}, {-1, 0, 0}, {2, 3, 4});
    validate_cpu<T, 3>({0, 1, -1}, {3, -4, 5}, {2, 3, 4});
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    StarpuTest starpu;
    // Init codelet
    starpu::copy_intersection::init();
    // Launch all tests
    validate_many<fp32_t>();
    validate_many<fp64_t>();
    return 0;
}

