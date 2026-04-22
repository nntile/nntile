/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/sumprod_fiber.cc
 * sumprod_fiber operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/sumprod_fiber.hh"
#include "nntile/starpu/sumprod_fiber.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check(Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    Tile<T> src1({3, 4, 5}), src2({3, 4, 5});
    Tile<T> dst[3] = {Tile<T>({3}), Tile<T>({4}), Tile<T>({5})};
    Tile<T> dst2[3] = {Tile<T>({3}), Tile<T>({4}), Tile<T>({5})};
    auto a = src1.acquire(STARPU_W);
    auto b = src2.acquire(STARPU_W);
    for(Index i = 0; i < src1.nelems; ++i)
    {
        a[i] = Y(i + 1);
        b[i] = Y(0.5 * (i + 1));
    }
    a.release();
    b.release();
    Y zero = 0;
    for(Index i = 0; i < 3; ++i)
    {
        auto dl = dst[i].acquire(STARPU_W);
        auto d2l = dst2[i].acquire(STARPU_W);
        for(Index j = 0; j < dst[i].nelems; ++j)
        {
            dl[j] = zero;
            d2l[j] = zero;
        }
        dl.release();
        d2l.release();
    }
    for(Index ax = 0; ax < 3; ++ax)
    {
        Index m = src1.stride[ax];
        Index n = src1.matrix_shape[ax+1][1];
        Index k = src1.shape[ax];
        starpu::sumprod_fiber.submit<std::tuple<T>>(m, n, k, alpha, src1, src2,
                beta, dst[ax]);
        sumprod_fiber<T>(alpha, src1, src2, beta, dst2[ax], ax);
        auto dl = dst[ax].acquire(STARPU_R);
        auto d2l = dst2[ax].acquire(STARPU_R);
        for(Index i = 0; i < dst[ax].nelems; ++i)
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
    check<T>(1.0, 0.0);
    check<T>(2.0, -0.5);
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
