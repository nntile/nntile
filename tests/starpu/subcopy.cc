/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/subcopy.cc
 * Copy subarray based on contiguous indices
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/subcopy.hh"
#include "nntile/kernel/subcopy.hh"
#include "../testing.hh"
#include <array>
#include <vector>
#include <stdexcept>
#include <iostream>

using namespace nntile;
using namespace nntile::starpu;

template<typename T, std::size_t NDIM>
void validate_cpu(std::array<Index, NDIM> src, std::array<Index, NDIM> dst,
        std::array<Index, NDIM> shape)
{
    using Y = typename T::repr_t;
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
        src_data[i] = Y(i+1);
    }
    std::vector<T> dst_data(dst_nelems);
    for(Index i = 0; i < dst_nelems; ++i)
    {
        dst_data[i] = Y(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2_data(dst_data);
    // Launch low-level kernel
    std::vector<nntile::int64_t> tmp_index(2*NDIM);
    std::cout << "Run kernel::subcopy::cpu<" << T::type_repr << ">\n";
    kernel::subcopy::cpu<T>(NDIM, &src_start[0], &src_stride[0],
            &copy_shape[0], &src_data[0], &dst_start[0], &dst_stride[0],
            &dst_data[0], &tmp_index[0]);
    // Check by actually submitting a task
    VariableHandle src_handle(&src_data[0], sizeof(T)*src_nelems,
            STARPU_R),
        dst2_handle(&dst2_data[0], sizeof(T)*dst_nelems, STARPU_RW),
        tmp_handle(&tmp_index[0], sizeof(Index)*NDIM*2, STARPU_R);
    subcopy::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::subcopy::submit<" << T::type_repr << "> restricted to "
        "CPU\n";
    subcopy::submit<T>(NDIM, src_start, src_stride,
            dst_start, dst_stride, copy_shape, src_handle, dst2_handle,
            tmp_handle, STARPU_RW);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < dst_nelems; ++i)
    {
        TEST_ASSERT(Y(dst_data[i]) == Y(dst2_data[i]));
    }
    std::cout << "OK: starpu::subcopy::submit<" << T::type_repr << "> restricted to CPU\n";
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
    Config starpu(1, 0, 0);
    // Init codelet
    subcopy::init();
    // Launch all tests
    validate_many<fp32_t>();
    validate_many<fp64_t>();
    validate_many<nntile::int64_t>();
    return 0;
}
