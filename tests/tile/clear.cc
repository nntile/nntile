/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/clear.cc
 * Clear Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/clear.hh"
#include "nntile/starpu/clear.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    TileTraits traits({4, 5, 6, 7});
    std::vector<T> data(traits.nelems);
    for(Index i = 0; i < traits.nelems; ++i)
    {
        data[i] = Y(i+1);
    }
    Tile<T> tile(traits, &data[0], traits.nelems);
    clear(tile);
    auto tile_local = tile.acquire(STARPU_R);
    constexpr Y zero = 0;
    for(Index i = 0; i < traits.nelems; ++i)
    {
        TEST_ASSERT(Y(tile_local[i]) == zero);
    }
    tile_local.release();
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
