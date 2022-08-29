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
 * @date 2022-08-23
 * */

#include "nntile/tile/clear.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    TileTraits traits({4, 5, 6, 7});
    std::vector<T> data(traits.nelems);
    for(Index i = 0; i < traits.nelems; ++i)
    {
        data[i] = T(i+1);
    }
    Tile<T> tile(traits, &data[0], traits.nelems, STARPU_RW);
    starpu_resume();
    clear(tile);
    starpu_pause();
    auto tile_local = tile.acquire(STARPU_R);
    constexpr T zero = 0;
    for(Index i = 0; i < traits.nelems; ++i)
    {
        if(tile_local[i] != zero)
        {
            throw std::runtime_error("Data is not zero");
        }
    }
}

int main(int argc, char **argv)
{
    // Init StarPU configuration and set number of CPU workers to 1
    starpu_conf conf;
    int ret = starpu_conf_init(&conf);
    if(ret != 0)
    {
        throw std::runtime_error("starpu_conf_init error");
    }
    conf.ncpus = 1;
    // No CUDA workers since we are checking against results of CPU
    // implementation
    conf.ncuda = 0;
    ret = starpu_init(&conf);
    if(ret != 0)
    {
        throw std::runtime_error("starpu_init error");
    }
    starpu_pause();
    validate<fp32_t>();
    validate<fp64_t>();
    starpu_resume();
    starpu_shutdown();
    return 0;
}

