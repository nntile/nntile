/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/addcdiv.cc
 * Addcdiv operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/addcdiv.hh"
#include "nntile/starpu/addcdiv.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate(Scalar val, Scalar eps)
{
    using Y = typename T::repr_t;
    Tile<T> src1({}), src1_copy({}), nom1({}), denom1({});
    Tile<T> src2({2, 3, 4}), src2_copy({2, 3, 4});

    auto src1_local = src1.acquire(STARPU_W);
    src1_local[0] = Y{-1};
    src1_local.release();
    auto src1_copy_local = src1_copy.acquire(STARPU_W);
    src1_copy_local[0] = Y{-1};
    src1_copy_local.release();

    auto nom1_local = nom1.acquire(STARPU_W);
    nom1_local[0] = Y(100);
    nom1_local.release();

    auto denom1_local = denom1.acquire(STARPU_W);
    denom1_local[0] = Y(-10);
    denom1_local.release();

    // auto tile2_local = tile2.acquire(STARPU_W);
    // auto tile2_copy_local = tile2_copy.acquire(STARPU_W);
    // for(Index i = 0; i < tile2.nelems; ++i)
    // {
    //     tile2_local[i] = T(i+1);
    //     tile2_copy_local[i] = T(i+1);
    // }
    // tile2_local.release();
    // tile2_copy_local.release();

    starpu::addcdiv::submit<T>(val, eps, src1.nelems, nom1, denom1, src1);
    addcdiv<T>(val, eps, nom1, denom1, src1_copy);
    src1_local.acquire(STARPU_R);
    src1_copy_local.acquire(STARPU_R);
    TEST_ASSERT(Y(src1_local[0]) == Y(src1_copy_local[0]));
    src1_local.release();
    src1_copy_local.release();

    // starpu::addcdiv::submit<T>(tile2.nelems, tile2);
    // addcdiv<T>(tile2_copy);
    // tile2_local.acquire(STARPU_R);
    // tile2_copy_local.acquire(STARPU_R);
    // for(Index i = 0; i < tile2.nelems; ++i)
    // {
    //     TEST_ASSERT(tile2_local[i] == tile2_copy_local[i]);
    // }
    // tile2_local.release();
    // tile2_copy_local.release();
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::addcdiv::init();
    starpu::addcdiv::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>(-1, 1e-3);
    validate<fp64_t>(1000, 1e-9);
    return 0;
}
