/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/sumprod_slice.cc
 * sumprod_slice operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/sumprod_slice.hh"
#include "nntile/starpu/sumprod_slice.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check(Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    // Init data for checking
    Tile<T> src1({3, 4, 5}), src2({3, 4, 5});
    Tile<T> dst[3] = {Tile<T>({4, 5}), Tile<T>({3, 5}), Tile<T>({3, 4})};
    Tile<T> dst2[3] = {Tile<T>({4, 5}), Tile<T>({3, 5}), Tile<T>({3, 4})};
    auto src1_local = src1.acquire(STARPU_W);
    auto src2_local = src2.acquire(STARPU_W);
    for(Index i = 0; i < src1.nelems; ++i)
    {
        src1_local[i] = Y(i+1) / Y(i*i+1);
        src2_local[i] = Y(i*i+1);
    }
    src1_local.release();
    src2_local.release();
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
        starpu::sumprod_slice::submit<T>(1, 20, 3, alpha, src1, src2, beta,
                dst[0]);
        sumprod_slice<T>(alpha, src1, src2, beta, dst2[0], 0);
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
        starpu::sumprod_slice::submit<T>(3, 5, 4, alpha, src1, src2, beta,
                dst[1]);
        sumprod_slice<T>(alpha, src1, src2, beta, dst2[1], 1);
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
        starpu::sumprod_slice::submit<T>(12, 1, 5, alpha, src1, src2, beta,
                dst[2]);
        sumprod_slice<T>(alpha, src1, src2, beta, dst2[2], 2);
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
    check<T>(-1.0, 0.0);
    check<T>(2.0, 2.0);
    // Check throwing exceptions
    Tile<T> src1({3, 4, 5}), src2({3, 4, 5});
    Tile<T> dst[3] = {Tile<T>({2, 4, 5}), Tile<T>({2, 3, 5}),
        Tile<T>({2, 3, 4})};
    Tile<T> empty({});
    TEST_THROW(sumprod_slice<T>(1.0, src1, src2, 0.0, empty, 0));
    TEST_THROW(sumprod_slice<T>(1.0, empty, empty, 0.0, empty, 0));
    TEST_THROW(sumprod_slice<T>(1.0, src1, src2, 0.0, dst[0], -1));
    TEST_THROW(sumprod_slice<T>(1.0, src1, src2, 0.0, dst[0], 3));
    TEST_THROW(sumprod_slice<T>(1.0, src1, src2, 0.0, src1, 0));
    TEST_THROW(sumprod_slice<T>(1.0, src1, src2, 0.0, dst[0], 1));
    TEST_THROW(sumprod_slice<T>(1.0, src1, src2, 0.0, dst[2], 1));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::sumprod_slice::init();
    starpu::sumprod_slice::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
