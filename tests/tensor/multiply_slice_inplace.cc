/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/multiply_slice_inplace.cc
 * Test for tensor::multiply_slice_inplace<T> C++ wrapper
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/multiply_slice_inplace.hh"
#include "nntile/tile/multiply_slice_inplace.hh"
#include "nntile/starpu/multiply_slice_inplace.hh"

#include <cmath>

#include "testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void test_multiply_slice_inplace()
{
    // Set up types
    using Y = typename T::repr_t;
    const Y alpha{1.5}, beta{0.5};
    // Set up shapes and axis
    std::vector<Index> shape_src{2, 3}, shape_dst{2, 4, 3};
    Index axis{1};
    // Create tensors
    Tensor<T> src(shape_src), dst(shape_dst);
    // Generate random arrays
    std::vector<Y> rand_src(src.nelem), rand_dst(dst.nelem);
    for(auto &x : rand_src)
    {
        x = Y{drandn()};
    }
    for(auto &x : rand_dst)
    {
        x = Y{drandn()};
    }
    // Set up random arrays
    src.set_scalar(Y{0});
    dst.set_scalar(Y{0});
    // Copy random arrays to tensors
    src.acquire(std::begin(rand_src));
    dst.acquire(std::begin(rand_dst));
    // Perform tensor-wise and tile-wise multiply_slice_inplace operations
    multiply_slice_inplace<T>(alpha, src, beta, dst, axis);
    // Check results for the first tile
    auto tile_src = src.get_tile(0);
    auto tile_dst = dst.get_tile(0);
    tile::multiply_slice_inplace<T>(alpha, tile_src, beta, tile_dst, axis);
    // Check result
    std::vector<Y> result_dst(dst.nelem);
    dst.acquire(std::end(result_dst));
    // Check if result is correct
    for(Index i0 = 0; i0 < shape_dst[0]; ++i0)
    {
        for(Index i1 = 0; i1 < shape_dst[1]; ++i1)
        {
            for(Index i2 = 0; i2 < shape_dst[2]; ++i2)
            {
                Index linear = i0*shape_dst[1]*shape_dst[2] + i1*shape_dst[2] + i2;
                Y expected = beta * rand_dst[linear] * alpha * rand_src[i0*shape_src[1] + i2];
                TEST_ASSERT(std::abs(Y{result_dst[linear]} - expected) < 1e-5);
            }
        }
    }
}

template<typename T>
void test_multiply_slice_inplace_async()
{
    // Set up types
    using Y = typename T::repr_t;
    const Y alpha{2.0}, beta{0.25};
    // Set up shapes and axis
    std::vector<Index> shape_src{3, 4}, shape_dst{3, 5, 4};
    Index axis{1};
    // Create tensors
    Tensor<T> src(shape_src), dst(shape_dst);
    // Generate random arrays
    std::vector<Y> rand_src(src.nelem), rand_dst(dst.nelem);
    for(auto &x : rand_src)
    {
        x = Y{drandn()};
    }
    for(auto &x : rand_dst)
    {
        x = Y{drandn()};
    }
    // Set up random arrays
    src.set_scalar(Y{0});
    dst.set_scalar(Y{0});
    // Copy random arrays to tensors
    src.acquire(std::begin(rand_src));
    dst.acquire(std::begin(rand_dst));
    // Perform tensor-wise and tile-wise multiply_slice_inplace operations
    multiply_slice_inplace_async<T>(alpha, src, beta, dst, axis);
    starpu_task_wait_for_all();
    // Check results for the first tile
    auto tile_src = src.get_tile(0);
    auto tile_dst = dst.get_tile(0);
    tile::multiply_slice_inplace_async<T>(alpha, tile_src, beta, tile_dst, axis);
    starpu_task_wait_for_all();
    // Check result
    std::vector<Y> result_dst(dst.nelem);
    dst.acquire(std::end(result_dst));
    // Check if result is correct
    for(Index i0 = 0; i0 < shape_dst[0]; ++i0)
    {
        for(Index i1 = 0; i1 < shape_dst[1]; ++i1)
        {
            for(Index i2 = 0; i2 < shape_dst[2]; ++i2)
            {
                Index linear = i0*shape_dst[1]*shape_dst[2] + i1*shape_dst[2] + i2;
                Y expected = beta * rand_dst[linear] * alpha * rand_src[i0*shape_src[1] + i2];
                TEST_ASSERT(std::abs(Y{result_dst[linear]} - expected) < 1e-5);
            }
        }
    }
}

template<typename T>
void test_multiply_slice_inplace_errors()
{
    using Y = typename T::repr_t;
    // Create tensors with wrong shapes
    Tensor<T> A({2, 3}), B({2, 4, 3}), C({2, 3}), D({2, 4, 5}), E({2, 4, 3}), F({2, 3}), G({2, 4, 3});
    // Test various error conditions
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, B, 0));
    TEST_THROW(multiply_slice_inplace<T>(1.0, F, 1.0, F, 0));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, B, -1));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, B, 2));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, D, 0));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, E, 0));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, B, 0));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, B, 1));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, G, 0));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, G, 1));
}

int main(int argc, char **argv)
{
    test_multiply_slice_inplace<fp32_t>();
    test_multiply_slice_inplace<fp64_t>();
    test_multiply_slice_inplace<bf16_t>();
    test_multiply_slice_inplace<fp16_t>();

    test_multiply_slice_inplace_async<fp32_t>();
    test_multiply_slice_inplace_async<fp64_t>();
    test_multiply_slice_inplace_async<bf16_t>();
    test_multiply_slice_inplace_async<fp16_t>();

    test_multiply_slice_inplace_errors<fp32_t>();
    test_multiply_slice_inplace_errors<fp64_t>();
    test_multiply_slice_inplace_errors<bf16_t>();
    test_multiply_slice_inplace_errors<fp16_t>();

    return 0;
}