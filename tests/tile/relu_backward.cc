/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/relu_backward.cc
 * Backward ReLU on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/relu_backward.hh"
#include "nntile/starpu/relu_backward.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> x({3, 4}), dy({3, 4}), dx({3, 4}), dxr({3, 4});
    auto xl = x.acquire(STARPU_W);
    auto dyl = dy.acquire(STARPU_W);
    auto dxl = dx.acquire(STARPU_W);
    auto dxrl = dxr.acquire(STARPU_W);
    for(Index i = 0; i < x.nelems; ++i)
    {
        xl[i] = Y(i - 4);
        dyl[i] = Y(0.5);
        dxl[i] = Y(0);
        dxrl[i] = Y(0);
    }
    xl.release();
    dyl.release();
    dxl.release();
    dxrl.release();

    starpu::relu_backward.submit<std::tuple<T>>(x.nelems, x, dy, dx);
    relu_backward<T>(x, dy, dxr);

    dxl.acquire(STARPU_R);
    dxrl.acquire(STARPU_R);
    for(Index i = 0; i < x.nelems; ++i)
    {
        TEST_ASSERT(Y(dxl[i]) == Y(dxrl[i]));
    }
    dxl.release();
    dxrl.release();
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
