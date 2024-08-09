/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/gelutanh_inplace.cc
 * Approximate GeLU operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/gelutanh_inplace.hh"
#include "nntile/starpu/gelutanh_inplace.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> tile1({}), tile1_copy({}), tile2({2, 3, 4}), tile2_copy({2, 3, 4});
    auto tile1_local = tile1.acquire(STARPU_W);
    tile1_local[0] = Y(-1);
    tile1_local.release();
    auto tile1_copy_local = tile1_copy.acquire(STARPU_W);
    tile1_copy_local[0] = Y(-1);
    tile1_copy_local.release();
    auto tile2_local = tile2.acquire(STARPU_W);
    auto tile2_copy_local = tile2_copy.acquire(STARPU_W);
    for(Index i = 0; i < tile2.nelems; ++i)
    {
        tile2_local[i] = Y(i+1);
        tile2_copy_local[i] = Y(i+1);
    }
    tile2_local.release();
    tile2_copy_local.release();
    starpu::gelutanh_inplace::submit<T>(1, tile1);
    gelutanh_inplace<T>(tile1_copy);
    tile1_local.acquire(STARPU_R);
    tile1_copy_local.acquire(STARPU_R);
    TEST_ASSERT(Y(tile1_local[0]) == Y(tile1_copy_local[0]));
    tile1_local.release();
    tile1_copy_local.release();
    starpu::gelutanh_inplace::submit<T>(tile2.nelems, tile2);
    gelutanh_inplace<T>(tile2_copy);
    tile2_local.acquire(STARPU_R);
    tile2_copy_local.acquire(STARPU_R);
    for(Index i = 0; i < tile2.nelems; ++i)
    {
        TEST_ASSERT(Y(tile2_local[i]) == Y(tile2_copy_local[i]));
    }
    tile2_local.release();
    tile2_copy_local.release();
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::gelutanh_inplace::init();
    starpu::gelutanh_inplace::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
