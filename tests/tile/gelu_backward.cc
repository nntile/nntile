/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/gelu_backward.cc
 * Backward GeLU on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/gelu_backward.hh"
#include "nntile/starpu/gelu_backward.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> x({2, 3}), dy({2, 3}), dx({2, 3}), dxr({2, 3});
    auto xl = x.acquire(STARPU_W);
    auto dyl = dy.acquire(STARPU_W);
    auto dxl = dx.acquire(STARPU_W);
    auto dxrl = dxr.acquire(STARPU_W);
    for(Index i = 0; i < x.nelems; ++i)
    {
        xl[i] = Y(0.1 * (i + 1));
        dyl[i] = Y(0.2 * (i + 1));
        dxl[i] = Y(0);
        dxrl[i] = Y(0);
    }
    xl.release();
    dyl.release();
    dxl.release();
    dxrl.release();

    starpu::gelu_backward.submit<std::tuple<T>>(x.nelems, x, dy, dx);
    gelu_backward<T>(x, dy, dxr);

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
