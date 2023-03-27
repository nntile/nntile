/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/sum.cc
 * Sum operation on Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Konstantin Sozykin
 * @date 2023-02-19
 * */

#include "nntile/tile/sum.hh"
#include "nntile/starpu/sum.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check()
{
    // Init data for checking
    Tile<T> src({3, 4, 5});
    Tile<T> dst[3] = {Tile<T>({4, 5}), Tile<T>({3, 5}),
        Tile<T>({3, 4})};
    Tile<T> dst2[3] = {Tile<T>({4, 5}), Tile<T>({3, 5}),
        Tile<T>({3, 4})};
    auto src_local = src.acquire(STARPU_W);
    for(Index i = 0; i < src.nelems; ++i)
    {
        src_local[i] = T(i+1);
    }
    src_local.release();
    T zero = 0;
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
        starpu::sum::submit<T>(1, 20, 3, src, dst[0]);
        sum<T>(src, dst2[0], 0);
        auto dst_local = dst[0].acquire(STARPU_R);
        auto dst2_local = dst2[0].acquire(STARPU_R);
        for(Index i = 0; i < dst[0].nelems; ++i)
        {
            TEST_ASSERT(dst_local[i] == dst2_local[i]);
        }
        dst_local.release();
        dst2_local.release();
    }
    // Check axis=1
    {
        starpu::sum::submit<T>(3, 5, 4, src, dst[1]);
        sum<T>(src, dst2[1], 1);
        auto dst_local = dst[1].acquire(STARPU_R);
        auto dst2_local = dst2[1].acquire(STARPU_R);
        for(Index i = 0; i < dst[1].nelems; ++i)
        {
            TEST_ASSERT(dst_local[i] == dst2_local[i]);
        }
        dst_local.release();
        dst2_local.release();
    }
    // Check axis=2
    {
        starpu::sum::submit<T>(12, 1, 5, src, dst[2]);
        sum<T>(src, dst2[2], 2);
        auto dst_local = dst[2].acquire(STARPU_R);
        auto dst2_local = dst2[2].acquire(STARPU_R);
        for(Index i = 0; i < dst[2].nelems; ++i)
        {
            TEST_ASSERT(dst_local[i] == dst2_local[i]);
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
    TEST_THROW(sum<T>(src, empty, 0));
    TEST_THROW(sum<T>(empty, empty, 0));
    TEST_THROW(sum<T>(src, dst[0], -1));
    TEST_THROW(sum<T>(src, dst[0], 3));
    TEST_THROW(sum<T>(src, src, 0));
    TEST_THROW(sum<T>(src, dst[0], 1));
    TEST_THROW(sum<T>(src, dst[2], 1));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::sum::init();
    starpu::sum::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

