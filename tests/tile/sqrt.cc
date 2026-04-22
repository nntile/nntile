/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/sqrt.cc
 * Sqrt operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/sqrt.hh"
#include "nntile/starpu/sqrt.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> s1({}), d1({}), s1c({}), d1c({});
    auto s1l = s1.acquire(STARPU_W);
    auto d1l = d1.acquire(STARPU_W);
    auto s1cl = s1c.acquire(STARPU_W);
    auto d1cl = d1c.acquire(STARPU_W);
    s1l[0] = Y(4);
    d1l[0] = Y(0);
    s1cl[0] = Y(4);
    d1cl[0] = Y(0);
    s1l.release();
    d1l.release();
    s1cl.release();
    d1cl.release();

    starpu::sqrt.submit<std::tuple<T>>(1, s1, d1);
    sqrt<T>(s1c, d1c);
    d1l.acquire(STARPU_R);
    d1cl.acquire(STARPU_R);
    TEST_ASSERT(Y(d1l[0]) == Y(d1cl[0]));
    d1l.release();
    d1cl.release();

    Tile<T> s2({2, 3}), d2({2, 3}), s2c({2, 3}), d2c({2, 3});
    auto s2l = s2.acquire(STARPU_W);
    auto d2l = d2.acquire(STARPU_W);
    auto s2cl = s2c.acquire(STARPU_W);
    auto d2cl = d2c.acquire(STARPU_W);
    for(Index i = 0; i < s2.nelems; ++i)
    {
        s2l[i] = Y((i + 1) * (i + 1));
        d2l[i] = Y(0);
        s2cl[i] = Y((i + 1) * (i + 1));
        d2cl[i] = Y(0);
    }
    s2l.release();
    d2l.release();
    s2cl.release();
    d2cl.release();

    starpu::sqrt.submit<std::tuple<T>>(s2.nelems, s2, d2);
    sqrt<T>(s2c, d2c);
    d2l.acquire(STARPU_R);
    d2cl.acquire(STARPU_R);
    for(Index i = 0; i < s2.nelems; ++i)
    {
        TEST_ASSERT(Y(d2l[i]) == Y(d2cl[i]));
    }
    d2l.release();
    d2cl.release();
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
