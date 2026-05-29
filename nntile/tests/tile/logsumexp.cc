/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/logsumexp.cc
 * logsumexp operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/logsumexp.hh"
#include "nntile/starpu/logsumexp.hh"
#include "../testing.hh"
#include <cmath>

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> src({2, 3, 4, 5}), dst({3, 4, 5}), dst2({3, 4, 5});
    auto sl = src.acquire(STARPU_W);
    auto dl = dst.acquire(STARPU_W);
    auto d2l = dst2.acquire(STARPU_W);
    for(Index i = 0; i < src.nelems; i += 2)
    {
        sl[i] = Y(0.5 * (i / 2 + 1));
        sl[i+1] = Y(std::exp(Y(i + 1) / Y{20}));
    }
    for(Index i = 0; i < dst.nelems; ++i)
    {
        dl[i] = Y(0);
        d2l[i] = Y(0);
    }
    sl.release();
    dl.release();
    d2l.release();

    starpu::logsumexp.submit<std::tuple<T>>(dst.nelems, src, dst);
    logsumexp<T>(src, dst2);

    dl.acquire(STARPU_R);
    d2l.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        TEST_ASSERT(Y(dl[i]) == Y(d2l[i]));
    }
    dl.release();
    d2l.release();
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
