/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/clear.cc
 * Clear Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-06
 * */

#include "nntile/tile/clear.hh"
#include "nntile/starpu/clear.hh"
#include "../testing.hh"
#include "../starpu/common.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check()
{
    TileTraits traits({4, 5, 6, 7});
    std::vector<T> data(traits.nelems);
    for(Index i = 0; i < traits.nelems; ++i)
    {
        data[i] = T(i+1);
    }
    Tile<T> tile(traits, &data[0], traits.nelems);
    clear(tile);
    auto tile_local = tile.acquire(STARPU_R);
    constexpr T zero = 0;
    for(Index i = 0; i < traits.nelems; ++i)
    {
        TEST_ASSERT(tile_local[i] == zero);
    }
    tile_local.release();
}

template<typename T>
void validate()
{
    starpu::clear::restrict_where(STARPU_CPU);
    check<T>();
#ifdef NNTILE_USE_CUDA
    starpu::clear::restrict_where(STARPU_CUDA);
    check<T>();
#endif
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    StarpuTest starpu;
    // Init codelet
    starpu::clear::init();
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

