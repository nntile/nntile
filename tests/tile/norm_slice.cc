/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/norm_slice.cc
 * Euclidean norms of fibers into a slice of a Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/norm_slice.hh"
#include "nntile/starpu/norm_slice.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check()
{
    using Y = typename T::repr_t;
    // Init data for checking
    Tile<T> src({3, 4, 5});
    Tile<T> dst[3] = {Tile<T>({4, 5}), Tile<T>({3, 5}),
        Tile<T>({3, 4})};
    Tile<T> dst2[3] = {Tile<T>({4, 5}), Tile<T>({3, 5}),
        Tile<T>({3, 4})};
    auto src_local = src.acquire(STARPU_W);
    Scalar alpha = -1.0, beta = 0.5;
    for(Index i = 0; i < src.nelems; ++i)
    {
        src_local[i] = Y(i+1);
    }
    src_local.release();
    Y zero = 0;
    for(Index i = 0; i < 3; ++i)
    {
        auto dst_local = dst[i].acquire(STARPU_W);
        auto dst2_local = dst2[i].acquire(STARPU_W);
        for(Index j = 0; j < dst[i].nelems; ++j)
        {
            dst_local[j] = zero;
            dst2_local[j] = zero;
        }
        dst_local.release();
        dst2_local.release();
    }
    // Check axis=0
    {
        starpu::norm_slice::submit<T>(1, 20, 3, alpha, src, beta, dst[0]);
        norm_slice<T>(alpha, src, beta, dst2[0], 0);
        auto dst_local = dst[0].acquire(STARPU_R);
        auto dst2_local = dst2[0].acquire(STARPU_R);
        for(Index i = 0; i < dst[0].nelems; ++i)
        {
            TEST_ASSERT(Y(dst_local[i]) == Y(dst2_local[i]));
        }
        dst_local.release();
        dst2_local.release();
    }
    // Check axis=1
    {
        starpu::norm_slice::submit<T>(3, 5, 4, alpha, src, beta, dst[1]);
        norm_slice<T>(alpha, src, beta, dst2[1], 1);
        auto dst_local = dst[1].acquire(STARPU_R);
        auto dst2_local = dst2[1].acquire(STARPU_R);
        for(Index i = 0; i < dst[1].nelems; ++i)
        {
            TEST_ASSERT(Y(dst_local[i]) == Y(dst2_local[i]));
        }
        dst_local.release();
        dst2_local.release();
    }
    // Check axis=2
    {
        starpu::norm_slice::submit<T>(12, 1, 5, alpha, src, beta, dst[2]);
        norm_slice<T>(alpha, src, beta, dst2[2], 2);
        auto dst_local = dst[2].acquire(STARPU_R);
        auto dst2_local = dst2[2].acquire(STARPU_R);
        for(Index i = 0; i < dst[2].nelems; ++i)
        {
            TEST_ASSERT(Y(dst_local[i]) == Y(dst2_local[i]));
        }
        dst_local.release();
        dst2_local.release();
    }
}

template<typename T>
void validate()
{
    // Check normal execution
    check<T>();
    // Check throwing exceptions
    Tile<T> src({3, 4, 5});
    Tile<T> dst[3] = {Tile<T>({2, 4, 5}), Tile<T>({2, 3, 5}),
        Tile<T>({2, 3, 4})};
    Tile<T> empty({});
    TEST_THROW(norm_slice<T>(1.0, src, 1.0, empty, 0));
    TEST_THROW(norm_slice<T>(1.0, empty, 1.0, empty, 0));
    TEST_THROW(norm_slice<T>(1.0, src, 1.0, dst[0], -1));
    TEST_THROW(norm_slice<T>(1.0, src, 1.0, dst[0], 3));
    TEST_THROW(norm_slice<T>(1.0, src, 1.0, src, 0));
    TEST_THROW(norm_slice<T>(1.0, src, 1.0, dst[0], 1));
    TEST_THROW(norm_slice<T>(1.0, src, 1.0, dst[2], 1));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::norm_slice::init();
    starpu::norm_slice::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
