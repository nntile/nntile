/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/conv2d_inplace.cc
 * Forward 2D-Convolution of two tiles in WHCN format
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/conv2d_inplace.hh"
#include "nntile/starpu/conv2d_inplace.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> X({3, 3, 1, 1}), C({2, 2, 1, 1}), Y_ref({2, 2, 1, 1}),
        Y_tile({2, 2, 1, 1});

    auto X_local = X.acquire(STARPU_W);
    auto C_local = C.acquire(STARPU_W);
    auto Y_ref_local = Y_ref.acquire(STARPU_W);
    auto Y_tile_local = Y_tile.acquire(STARPU_W);
    for(Index i = 0; i < X.nelems; ++i)
    {
        X_local[i] = Y(i+1);
    }
    for(Index i = 0; i < C.nelems; ++i)
    {
        C_local[i] = Y(i+1);
    }
    for(Index i = 0; i < Y_ref.nelems; ++i)
    {
        Y_ref_local[i] = Y(0);
        Y_tile_local[i] = Y(0);
    }
    X_local.release();
    C_local.release();
    Y_ref_local.release();
    Y_tile_local.release();

    starpu::conv2d_inplace.submit<std::tuple<T>>(3, 3, 1, 1, 2, 2, 1, 1, 1, 0,
            0, 1.0, X, C, 2, 2, 1, 1, 0.0, Y_ref);
    conv2d_inplace<T>(3, 3, 1, 1, 2, 2, 1, 1, 1, 0, 0, 1.0, X, C, 2, 2, 1, 1,
            0.0, Y_tile);

    Y_ref_local.acquire(STARPU_R);
    Y_tile_local.acquire(STARPU_R);
    for(Index i = 0; i < Y_ref.nelems; ++i)
    {
        TEST_ASSERT(Y(Y_ref_local[i]) == Y(Y_tile_local[i]));
    }
    Y_ref_local.release();
    Y_tile_local.release();
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
