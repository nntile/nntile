/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/silu_inplace.cc
 * In-place SiLU on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/silu_inplace.hh"
#include "nntile/starpu/silu_inplace.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> a({}), b({});
    auto al = a.acquire(STARPU_W);
    auto bl = b.acquire(STARPU_W);
    al[0] = Y(-0.25);
    bl[0] = Y(-0.25);
    al.release();
    bl.release();

    starpu::silu_inplace.submit<std::tuple<T>>(1, a);
    silu_inplace<T>(b);
    al.acquire(STARPU_R);
    bl.acquire(STARPU_R);
    TEST_ASSERT(Y(al[0]) == Y(bl[0]));
    al.release();
    bl.release();

    Tile<T> x({2, 3}), y({2, 3});
    auto xl = x.acquire(STARPU_W);
    auto yl = y.acquire(STARPU_W);
    for(Index i = 0; i < x.nelems; ++i)
    {
        xl[i] = Y(i - 1);
        yl[i] = Y(i - 1);
    }
    xl.release();
    yl.release();

    starpu::silu_inplace.submit<std::tuple<T>>(x.nelems, x);
    silu_inplace<T>(y);
    xl.acquire(STARPU_R);
    yl.acquire(STARPU_R);
    for(Index i = 0; i < x.nelems; ++i)
    {
        TEST_ASSERT(Y(xl[i]) == Y(yl[i]));
    }
    xl.release();
    yl.release();
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
