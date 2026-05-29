/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/softmax.cc
 * softmax operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/softmax.hh"
#include "nntile/starpu/softmax.hh"
#include "../testing.hh"
#include <cmath>

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check()
{
    using Y = typename T::repr_t;
    constexpr Scalar alpha = 1.0;
    Tile<T> src({3, 4, 5}), dst({3, 4, 5}), dst2({3, 4, 5});
    Tile<T> maxsumexp[3] = {Tile<T>({2, 4, 5}), Tile<T>({2, 3, 5}),
        Tile<T>({2, 3, 4})};
    auto sl = src.acquire(STARPU_W);
    auto dl = dst.acquire(STARPU_W);
    auto d2l = dst2.acquire(STARPU_W);
    for(Index i = 0; i < src.nelems; ++i)
    {
        sl[i] = Y(0.01 * (i + 1));
        dl[i] = Y(0);
        d2l[i] = Y(0);
    }
    sl.release();
    dl.release();
    d2l.release();
    for(Index i = 0; i < 3; ++i)
    {
        auto ml = maxsumexp[i].acquire(STARPU_W);
        for(Index j = 0; j < maxsumexp[i].nelems; j += 2)
        {
            ml[j] = Y(j + 1);
            ml[j+1] = std::exp(Y(j + 2) / Y{10});
        }
        ml.release();
    }
    {
        starpu::softmax.submit<std::tuple<T>>(1, 20, 3, maxsumexp[0], src, alpha,
                dst);
        softmax<T>(maxsumexp[0], src, alpha, dst2, 0);
        dl.acquire(STARPU_R);
        d2l.acquire(STARPU_R);
        for(Index i = 0; i < dst.nelems; ++i)
        {
            TEST_ASSERT(Y(dl[i]) == Y(d2l[i]));
        }
        dl.release();
        d2l.release();
    }
    {
        starpu::softmax.submit<std::tuple<T>>(3, 5, 4, maxsumexp[1], src, alpha,
                dst);
        softmax<T>(maxsumexp[1], src, alpha, dst2, 1);
        dl.acquire(STARPU_R);
        d2l.acquire(STARPU_R);
        for(Index i = 0; i < dst.nelems; ++i)
        {
            TEST_ASSERT(Y(dl[i]) == Y(d2l[i]));
        }
        dl.release();
        d2l.release();
    }
    {
        starpu::softmax.submit<std::tuple<T>>(12, 1, 5, maxsumexp[2], src, alpha,
                dst);
        softmax<T>(maxsumexp[2], src, alpha, dst2, 2);
        dl.acquire(STARPU_R);
        d2l.acquire(STARPU_R);
        for(Index i = 0; i < dst.nelems; ++i)
        {
            TEST_ASSERT(Y(dl[i]) == Y(d2l[i]));
        }
        dl.release();
        d2l.release();
    }
}

template<typename T>
void validate()
{
    check<T>();
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
