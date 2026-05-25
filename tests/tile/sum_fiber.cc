/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/sum_fiber.cc
 * Sum over fibers into a slice of a Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/sum_fiber.hh"
#include "nntile/starpu/sum_fiber.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check(Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    Tile<T> src({3, 4, 5});
    Tile<T> dst[3] = {Tile<T>({3}), Tile<T>({4}), Tile<T>({5})};
    Tile<T> dst2[3] = {Tile<T>({3}), Tile<T>({4}), Tile<T>({5})};
    auto src_local = src.acquire(STARPU_W);
    for(Index i = 0; i < src.nelems; ++i)
    {
        src_local[i] = Y(i + 1);
    }
    src_local.release();
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
    // axis=0 -> fiber along first dim, dst is last dims as 1D fiber of len 5
    {
        Index batch = dst[0].matrix_shape[1][1];
        Index m = src.stride[0];
        Index n = src.matrix_shape[1][1] / batch;
        Index k = src.shape[0];
        starpu::sum_fiber.submit<std::tuple<T>>(m, n, k, batch, alpha, src, beta,
                dst[0]);
        sum_fiber<T>(alpha, src, beta, dst2[0], 0, 0);
        auto dl = dst[0].acquire(STARPU_R);
        auto d2l = dst2[0].acquire(STARPU_R);
        for(Index i = 0; i < dst[0].nelems; ++i)
        {
            TEST_ASSERT(Y(dl[i]) == Y(d2l[i]));
        }
        dl.release();
        d2l.release();
    }
    {
        Index batch = dst[1].matrix_shape[1][1];
        Index m = src.stride[1];
        Index n = src.matrix_shape[2][1] / batch;
        Index k = src.shape[1];
        starpu::sum_fiber.submit<std::tuple<T>>(m, n, k, batch, alpha, src, beta,
                dst[1]);
        sum_fiber<T>(alpha, src, beta, dst2[1], 1, 0);
        auto dl = dst[1].acquire(STARPU_R);
        auto d2l = dst2[1].acquire(STARPU_R);
        for(Index i = 0; i < dst[1].nelems; ++i)
        {
            TEST_ASSERT(Y(dl[i]) == Y(d2l[i]));
        }
        dl.release();
        d2l.release();
    }
    {
        Index batch = dst[2].matrix_shape[1][1];
        Index m = src.stride[2];
        Index n = src.matrix_shape[3][1] / batch;
        Index k = src.shape[2];
        starpu::sum_fiber.submit<std::tuple<T>>(m, n, k, batch, alpha, src, beta,
                dst[2]);
        sum_fiber<T>(alpha, src, beta, dst2[2], 2, 0);
        auto dl = dst[2].acquire(STARPU_R);
        auto d2l = dst2[2].acquire(STARPU_R);
        for(Index i = 0; i < dst[2].nelems; ++i)
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
    check<T>(-0.5, 1.5);
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
