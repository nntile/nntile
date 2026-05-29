/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/gelu.cc
 * GeLU operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/gelu.hh"
#include "nntile/starpu/gelu.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> tile1_src({}), tile1_dst({}), tile1_src_copy({}), tile1_dst_copy({});
    Tile<T> tile2_src({2, 3, 4}), tile2_dst({2, 3, 4}), tile2_src_copy({2, 3, 4}), tile2_dst_copy({2, 3, 4});

    // Test with single element
    auto tile1_src_local = tile1_src.acquire(STARPU_W);
    auto tile1_dst_local = tile1_dst.acquire(STARPU_W);
    tile1_src_local[0] = Y(-1);
    tile1_dst_local[0] = Y(0);
    tile1_src_local.release();
    tile1_dst_local.release();

    auto tile1_src_copy_local = tile1_src_copy.acquire(STARPU_W);
    auto tile1_dst_copy_local = tile1_dst_copy.acquire(STARPU_W);
    tile1_src_copy_local[0] = Y(-1);
    tile1_dst_copy_local[0] = Y(0);
    tile1_src_copy_local.release();
    tile1_dst_copy_local.release();

    starpu::gelu.submit<std::tuple<T>>(1, tile1_src, tile1_dst);
    gelu<T>(tile1_src_copy, tile1_dst_copy);

    tile1_dst_local.acquire(STARPU_R);
    tile1_dst_copy_local.acquire(STARPU_R);
    TEST_ASSERT(Y(tile1_dst_local[0]) == Y(tile1_dst_copy_local[0]));
    tile1_dst_local.release();
    tile1_dst_copy_local.release();

    // Test with multiple elements
    auto tile2_src_local = tile2_src.acquire(STARPU_W);
    auto tile2_dst_local = tile2_dst.acquire(STARPU_W);
    auto tile2_src_copy_local = tile2_src_copy.acquire(STARPU_W);
    auto tile2_dst_copy_local = tile2_dst_copy.acquire(STARPU_W);
    for(Index i = 0; i < tile2_src.nelems; ++i)
    {
        tile2_src_local[i] = Y(i+1);
        tile2_dst_local[i] = Y(0);
        tile2_src_copy_local[i] = Y(i+1);
        tile2_dst_copy_local[i] = Y(0);
    }
    tile2_src_local.release();
    tile2_dst_local.release();
    tile2_src_copy_local.release();
    tile2_dst_copy_local.release();

    starpu::gelu.submit<std::tuple<T>>(tile2_src.nelems, tile2_src, tile2_dst);
    gelu<T>(tile2_src_copy, tile2_dst_copy);

    tile2_dst_local.acquire(STARPU_R);
    tile2_dst_copy_local.acquire(STARPU_R);
    for(Index i = 0; i < tile2_src.nelems; ++i)
    {
        TEST_ASSERT(Y(tile2_dst_local[i]) == Y(tile2_dst_copy_local[i]));
    }
    tile2_dst_local.release();
    tile2_dst_copy_local.release();
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

    return 0;
}
