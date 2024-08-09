/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/copy.cc
 * Copy one tile into another
 *
 * @version 1.1.0
 * */

#include "nntile/tile/copy.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> tile1({}), tile2({2, 2, 3}), tile3({2, 3, 4});
    // Check full copying, that is delegated to starpu_data_cpy internally
    auto tile1_local = tile1.acquire(STARPU_W);
    tile1_local[0] = Y(-1);
    tile1_local.release();
    auto tile2_local = tile2.acquire(STARPU_W);
    for(Index i = 0; i < tile2.nelems; ++i)
    {
        tile2_local[i] = Y(i+1);
    }
    tile2_local.release();
    auto tile3_local = tile3.acquire(STARPU_W);
    for(Index i = 0; i < tile3.nelems; ++i)
    {
        tile3_local[i] = Y(2*i+2);
    }
    tile3_local.release();
    Tile<T> tile1_copy({});
    copy<T>(tile1, tile1_copy);
    auto tile1_copy_local = tile1_copy.acquire(STARPU_R);
    TEST_ASSERT(Y(tile1_copy_local[0]) == Y(-1));
    tile1_copy_local.release();
    Tile<T> tile2_copy(tile2.shape);
    copy<T>(tile2, tile2_copy);
    auto tile2_copy_local = tile2_copy.acquire(STARPU_RW);
    for(Index i = 0; i < tile2.nelems; ++i)
    {
        TEST_ASSERT(Y(tile2_copy_local[i]) == Y(i+1));
        tile2_copy_local[i] = Y(-2);
    }
    tile2_copy_local.release();
    copy<T>(tile2, tile2_copy);
    tile2_copy_local.acquire(STARPU_R);
    for(Index i = 0; i < tile2.nelems; ++i)
    {
        TEST_ASSERT(Y(tile2_copy_local[i]) == Y(i+1));
    }
    tile2_copy_local.release();
    // Checking throwing exceptions
    TEST_THROW(copy<T>(Tile<T>({1}), Tile<T>({2})));
    TEST_THROW(copy<T>(Tile<T>({1}), Tile<T>({})));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    validate<nntile::int64_t>();
    return 0;
}
