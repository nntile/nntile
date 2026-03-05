/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/conv2d_bwd_weight_inplace.cc
 * Backward 2D-Convolution of two tiles in WHCN format to get weight grad
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/conv2d_bwd_weight_inplace.hh"
#include "nntile/starpu/conv2d_bwd_weight_inplace.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> X({3, 3, 1, 1}), dY({2, 2, 1, 1}), dC_ref({2, 2, 1, 1}),
        dC_tile({2, 2, 1, 1});

    auto X_local = X.acquire(STARPU_W);
    auto dY_local = dY.acquire(STARPU_W);
    auto dC_ref_local = dC_ref.acquire(STARPU_W);
    auto dC_tile_local = dC_tile.acquire(STARPU_W);
    for(Index i = 0; i < X.nelems; ++i)
    {
        X_local[i] = Y(i+1);
    }
    for(Index i = 0; i < dY.nelems; ++i)
    {
        dY_local[i] = Y(i+1);
    }
    for(Index i = 0; i < dC_ref.nelems; ++i)
    {
        dC_ref_local[i] = Y(0);
        dC_tile_local[i] = Y(0);
    }
    X_local.release();
    dY_local.release();
    dC_ref_local.release();
    dC_tile_local.release();

    starpu::conv2d_bwd_weight_inplace.submit<std::tuple<T>>(3, 3, 1, 1, 2, 2, 1,
            1, 1, 0, 0, 1.0, X, dY, 2, 2, 1, 1, 0.0, dC_ref);
    conv2d_bwd_weight_inplace<T>(3, 3, 1, 1, 2, 2, 1, 1, 1, 0, 0, 1.0, X, dY,
            2, 2, 1, 1, 0.0, dC_tile);

    dC_ref_local.acquire(STARPU_R);
    dC_tile_local.acquire(STARPU_R);
    for(Index i = 0; i < dC_ref.nelems; ++i)
    {
        TEST_ASSERT(Y(dC_ref_local[i]) == Y(dC_tile_local[i]));
    }
    dC_ref_local.release();
    dC_tile_local.release();
}

int main(int argc, char **argv)
{
    int ncpu=1, ncuda=0, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
