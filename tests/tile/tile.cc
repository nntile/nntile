/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/tile.cc
 * Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/tile.hh"
#include <limits>
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check_tile(const std::vector<Index> &shape)
{
    using Y = typename T::repr_t;
    // Check temporary tile with allocation done by StarPU
    Tile<T> tile1(shape);
    TEST_ASSERT(static_cast<starpu_data_handle_t>(tile1) != nullptr);
    auto tile1_local = tile1.acquire(STARPU_W);
    TEST_ASSERT(tile1_local.get_ptr() != nullptr);
    for(Index i = 0; i < tile1.nelems; ++i)
    {
        tile1_local[i] = Y(i);
    }
    tile1_local.release();
    // Check copy construction
    Tile<T> tile2(tile1);
    TEST_ASSERT(static_cast<starpu_data_handle_t>(tile2) ==
            static_cast<starpu_data_handle_t>(tile1));
    // Check constructor from TileTraits
    Tile<T> tile3(static_cast<TileTraits>(tile2));
    TEST_ASSERT(static_cast<starpu_data_handle_t>(tile3) != nullptr);
    TEST_ASSERT(static_cast<starpu_data_handle_t>(tile2) !=
            static_cast<starpu_data_handle_t>(tile3));
    // Check if acquire, release and copy are working together
    starpu_data_cpy(static_cast<starpu_data_handle_t>(tile3),
            static_cast<starpu_data_handle_t>(tile2), 0, nullptr, nullptr);
    auto tile3_local = tile3.acquire(STARPU_R);
    for(Index i = 0; i < tile2.nelems; ++i)
    {
        TEST_ASSERT(Y(tile3_local[i]) == Y(i));
    }
    tile3_local.release();
    // Check with shape and pointer
    std::vector<T> data(tile1.nelems);
    for(Index i = 0; i < tile1.nelems; ++i)
    {
        data[i] = Y(i+1);
    }
    TEST_THROW(Tile<T>(shape, &data[0], tile1.nelems-1));
    Tile<T> tile4(shape, &data[0], tile1.nelems);
    TEST_ASSERT(static_cast<starpu_data_handle_t>(tile4) != nullptr);
    starpu_data_cpy(static_cast<starpu_data_handle_t>(tile3),
            static_cast<starpu_data_handle_t>(tile4), 0, nullptr, nullptr);
    tile3_local.acquire(STARPU_R);
    for(Index i = 0; i < tile2.nelems; ++i)
    {
        TEST_ASSERT(Y(tile3_local[i]) == Y(i+1));
    }
    tile3_local.release();
    // Check with TileTraits and pointer
    TEST_THROW(Tile<T>(tile4, &data[0], tile4.nelems-1));
    Tile<T> tile5(tile4, &data[0], tile4.nelems);
    TEST_ASSERT(static_cast<starpu_data_handle_t>(tile5) != nullptr);
    TEST_ASSERT(static_cast<starpu_data_handle_t>(tile5) !=
            static_cast<starpu_data_handle_t>(tile4));
    starpu_data_cpy(static_cast<starpu_data_handle_t>(tile3),
            static_cast<starpu_data_handle_t>(tile5), 0, nullptr, nullptr);
    tile3_local.acquire(STARPU_RW);
    for(Index i = 0; i < tile2.nelems; ++i)
    {
        TEST_ASSERT(Y(tile3_local[i]) == Y(i+1));
        tile3_local[i] = Y(tile3_local[i]) + Y(1);
    }
    tile3_local.release();
}

template<typename T>
void validate_tile()
{
    check_tile<T>({});
    check_tile<T>({3});
    check_tile<T>({3, 2});
    check_tile<T>({3, 2, 1});
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Launch all tests
    validate_tile<fp64_t>();
    validate_tile<fp32_t>();
    return 0;
}
