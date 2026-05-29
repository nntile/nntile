/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/hypot.cc
 * Per-element hypot function of tiles
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/hypot.hh"
#include "nntile/starpu/hypot.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;

    // Test with single element
    Tile<T> tile1_src1({}), tile1_src2({}), tile1_dst({}), tile1_src1_copy({}), tile1_src2_copy({}), tile1_dst_copy({});
    Tile<T> tile2_src1({2, 3, 4}), tile2_src2({2, 3, 4}), tile2_dst({2, 3, 4}), tile2_src1_copy({2, 3, 4}), tile2_src2_copy({2, 3, 4}), tile2_dst_copy({2, 3, 4});

    // Test with single element
    auto tile1_src1_local = tile1_src1.acquire(STARPU_W);
    auto tile1_src2_local = tile1_src2.acquire(STARPU_W);
    auto tile1_dst_local = tile1_dst.acquire(STARPU_W);
    tile1_src1_local[0] = Y(-1);
    tile1_src2_local[0] = Y(2);
    tile1_dst_local[0] = Y(0);
    tile1_src1_local.release();
    tile1_src2_local.release();
    tile1_dst_local.release();

    auto tile1_src1_copy_local = tile1_src1_copy.acquire(STARPU_W);
    auto tile1_src2_copy_local = tile1_src2_copy.acquire(STARPU_W);
    auto tile1_dst_copy_local = tile1_dst_copy.acquire(STARPU_W);
    tile1_src1_copy_local[0] = Y(-1);
    tile1_src2_copy_local[0] = Y(2);
    tile1_dst_copy_local[0] = Y(0);
    tile1_src1_copy_local.release();
    tile1_src2_copy_local.release();
    tile1_dst_copy_local.release();

    starpu::hypot.submit<std::tuple<T>>(1, 1.0, tile1_src1, -1.5, tile1_src2, tile1_dst);
    hypot<T>(1.0, tile1_src1_copy, -1.5, tile1_src2_copy, tile1_dst_copy);

    tile1_dst_local.acquire(STARPU_R);
    tile1_dst_copy_local.acquire(STARPU_R);
    TEST_ASSERT(Y(tile1_dst_local[0]) == Y(tile1_dst_copy_local[0]));
    tile1_dst_local.release();
    tile1_dst_copy_local.release();

    // Test with multiple elements
    auto tile2_src1_local = tile2_src1.acquire(STARPU_W);
    auto tile2_src2_local = tile2_src2.acquire(STARPU_W);
    auto tile2_dst_local = tile2_dst.acquire(STARPU_W);
    auto tile2_src1_copy_local = tile2_src1_copy.acquire(STARPU_W);
    auto tile2_src2_copy_local = tile2_src2_copy.acquire(STARPU_W);
    auto tile2_dst_copy_local = tile2_dst_copy.acquire(STARPU_W);

    for(Index i = 0; i < tile2_src1.nelems; ++i)
    {
        tile2_src1_local[i] = Y(i+1);
        tile2_src2_local[i] = Y(i+2);
        tile2_dst_local[i] = Y(0);
        tile2_src1_copy_local[i] = Y(i+1);
        tile2_src2_copy_local[i] = Y(i+2);
        tile2_dst_copy_local[i] = Y(0);
    }
    tile2_src1_local.release();
    tile2_src2_local.release();
    tile2_dst_local.release();
    tile2_src1_copy_local.release();
    tile2_src2_copy_local.release();
    tile2_dst_copy_local.release();

    starpu::hypot.submit<std::tuple<T>>(tile2_src1.nelems, 2.0, tile2_src1, 0.5, tile2_src2, tile2_dst);
    hypot<T>(2.0, tile2_src1_copy, 0.5, tile2_src2_copy, tile2_dst_copy);

    tile2_dst_local.acquire(STARPU_R);
    tile2_dst_copy_local.acquire(STARPU_R);
    for(Index i = 0; i < tile2_src1.nelems; ++i)
    {
        TEST_ASSERT(Y(tile2_dst_local[i]) == Y(tile2_dst_copy_local[i]));
    }
    tile2_dst_local.release();
    tile2_dst_copy_local.release();

    // Test edge cases with zero alpha/beta
    Tile<T> tile3_src1({3}), tile3_src2({3}), tile3_dst({3}), tile3_src1_copy({3}), tile3_src2_copy({3}), tile3_dst_copy({3});

    auto tile3_src1_local = tile3_src1.acquire(STARPU_W);
    auto tile3_src2_local = tile3_src2.acquire(STARPU_W);
    auto tile3_dst_local = tile3_dst.acquire(STARPU_W);
    auto tile3_src1_copy_local = tile3_src1_copy.acquire(STARPU_W);
    auto tile3_src2_copy_local = tile3_src2_copy.acquire(STARPU_W);
    auto tile3_dst_copy_local = tile3_dst_copy.acquire(STARPU_W);

    for(Index i = 0; i < tile3_src1.nelems; ++i)
    {
        tile3_src1_local[i] = Y(i+1);
        tile3_src2_local[i] = Y(i+2);
        tile3_dst_local[i] = Y(0);
        tile3_src1_copy_local[i] = Y(i+1);
        tile3_src2_copy_local[i] = Y(i+2);
        tile3_dst_copy_local[i] = Y(0);
    }
    tile3_src1_local.release();
    tile3_src2_local.release();
    tile3_dst_local.release();
    tile3_src1_copy_local.release();
    tile3_src2_copy_local.release();
    tile3_dst_copy_local.release();

    // Test case: alpha=0, beta=0 -> should result in 0
    starpu::hypot.submit<std::tuple<T>>(tile3_src1.nelems, 0.0, tile3_src1, 0.0, tile3_src2, tile3_dst);
    hypot<T>(0.0, tile3_src1_copy, 0.0, tile3_src2_copy, tile3_dst_copy);

    tile3_dst_local.acquire(STARPU_R);
    tile3_dst_copy_local.acquire(STARPU_R);
    for(Index i = 0; i < tile3_src1.nelems; ++i)
    {
        TEST_ASSERT(Y(tile3_dst_local[i]) == Y(0.0));
        TEST_ASSERT(Y(tile3_dst_copy_local[i]) == Y(0.0));
    }
    tile3_dst_local.release();
    tile3_dst_copy_local.release();
}

int main(int argc, char **argv)
{
    // Initialize StarPU
    int ncpu=1, ncuda=0, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    validate<bf16_t>();
    validate<fp16_t>();

    return 0;
}
