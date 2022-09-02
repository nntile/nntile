/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/gelu.cc
 * GeLU operation on Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-02
 * */

#include "nntile/tile/gelu.hh"
#include "nntile/starpu/gelu.hh"
#include "../testing.hh"
#include "../starpu/common.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    Tile<T> tile1({}), tile1_copy({}), tile2({2, 3, 4}), tile2_copy({2, 3, 4});
    auto tile1_local = tile1.acquire(STARPU_W);
    tile1_local[0] = T{-1};
    tile1_local.release();
    auto tile1_copy_local = tile1_copy.acquire(STARPU_W);
    tile1_copy_local[0] = T{-1};
    tile1_copy_local.release();
    auto tile2_local = tile2.acquire(STARPU_W);
    auto tile2_copy_local = tile2_copy.acquire(STARPU_W);
    for(Index i = 0; i < tile2.nelems; ++i)
    {
        tile2_local[i] = T(i+1);
        tile2_copy_local[i] = T(i+1);
    }
    tile2_local.release();
    tile2_copy_local.release();
    starpu_resume();
    starpu::gelu::submit<T>(1, tile1);
    gelu<T>(tile1_copy);
    starpu_pause();
    tile1_local.acquire(STARPU_R);
    tile1_copy_local.acquire(STARPU_R);
    TEST_ASSERT(tile1_local[0] == tile1_copy_local[0]);
    tile1_local.release();
    tile1_copy_local.release();
    starpu_resume();
    starpu::gelu::submit<T>(tile2.nelems, tile2);
    gelu<T>(tile2_copy);
    tile2_local.acquire(STARPU_R);
    tile2_copy_local.acquire(STARPU_R);
    for(Index i = 0; i < tile2.nelems; ++i)
    {
        TEST_ASSERT(tile2_local[i] == tile2_copy_local[i]);
    }
    tile2_local.release();
    tile2_copy_local.release();
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    StarpuTest starpu;
    // Init codelet
    starpu::gelu::init();
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

