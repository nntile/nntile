/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/norm_slice.cc
 * Euclidean norms of fibers into a slice of a Tile<T> (out-of-place version)
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/norm_slice.hh"
#include "nntile/starpu/norm_slice.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check()
{
    using Y = typename T::repr_t;
    Tile<T> src1({3, 4, 5});
    Tile<T> src2[3] = {Tile<T>({4, 5}), Tile<T>({3, 5}), Tile<T>({3, 4})};
    Tile<T> dst[3] = {Tile<T>({4, 5}), Tile<T>({3, 5}), Tile<T>({3, 4})};
    Tile<T> dst2[3] = {Tile<T>({4, 5}), Tile<T>({3, 5}), Tile<T>({3, 4})};
    auto s1 = src1.acquire(STARPU_W);
    Scalar alpha = -1.0, beta = 0.5;
    for(Index i = 0; i < src1.nelems; ++i)
    {
        s1[i] = Y(i + 1);
    }
    s1.release();
    Y zero = 0;
    for(Index ax = 0; ax < 3; ++ax)
    {
        auto s2l = src2[ax].acquire(STARPU_W);
        auto dl = dst[ax].acquire(STARPU_W);
        auto d2l = dst2[ax].acquire(STARPU_W);
        for(Index j = 0; j < src2[ax].nelems; ++j)
        {
            s2l[j] = zero;
            dl[j] = zero;
            d2l[j] = zero;
        }
        s2l.release();
        dl.release();
        d2l.release();
    }
    // axis=0
    {
        Index m = 1, n = 20, k = 3;
        starpu::norm_slice.submit<std::tuple<T>>(m, n, k, alpha, src1, beta,
                src2[0], dst[0], 0);
        norm_slice<T>(alpha, src1, beta, src2[0], dst2[0], 0, 0);
        auto dl = dst[0].acquire(STARPU_R);
        auto d2l = dst2[0].acquire(STARPU_R);
        for(Index i = 0; i < dst[0].nelems; ++i)
        {
            TEST_ASSERT(Y(dl[i]) == Y(d2l[i]));
        }
        dl.release();
        d2l.release();
    }
    // axis=1
    {
        Index m = 3, n = 5, k = 4;
        starpu::norm_slice.submit<std::tuple<T>>(m, n, k, alpha, src1, beta,
                src2[1], dst[1], 0);
        norm_slice<T>(alpha, src1, beta, src2[1], dst2[1], 1, 0);
        auto dl = dst[1].acquire(STARPU_R);
        auto d2l = dst2[1].acquire(STARPU_R);
        for(Index i = 0; i < dst[1].nelems; ++i)
        {
            TEST_ASSERT(Y(dl[i]) == Y(d2l[i]));
        }
        dl.release();
        d2l.release();
    }
    // axis=2
    {
        Index m = 12, n = 1, k = 5;
        starpu::norm_slice.submit<std::tuple<T>>(m, n, k, alpha, src1, beta,
                src2[2], dst[2], 0);
        norm_slice<T>(alpha, src1, beta, src2[2], dst2[2], 2, 0);
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
    check<T>();
    Tile<T> src({3, 4, 5});
    Tile<T> slice({4, 5});
    Tile<T> wrong1({2, 4, 5});
    Tile<T> wrong2({2, 3, 5});
    TEST_THROW(norm_slice<T>(1.0, src, 1.0, wrong1, slice, 0, 0));
    TEST_THROW(norm_slice<T>(1.0, src, 1.0, wrong2, slice, 1, 0));
    TEST_THROW(norm_slice<T>(1.0, src, 1.0, slice, slice, -1, 0));
    TEST_THROW(norm_slice<T>(1.0, src, 1.0, slice, slice, 3, 0));
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
