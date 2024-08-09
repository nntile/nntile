/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/softmax_inplace.cc
 * softmax_inplace operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/softmax_inplace.hh"
#include "nntile/starpu/softmax_inplace.hh"
#include "../testing.hh"
#include <cmath>

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check()
{
    using Y = typename T::repr_t;
    constexpr Scalar alpha = 1.0;
    // Init data for checking
    Tile<T> dst({3, 4, 5}), dst2({3, 4, 5});
    Tile<T> maxsumexp[3] = {Tile<T>({2, 4, 5}), Tile<T>({2, 3, 5}),
        Tile<T>({2, 3, 4})};
    auto dst_local = dst.acquire(STARPU_W);
    auto dst2_local = dst2.acquire(STARPU_W);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        dst_local[i] = Y(i+1);
        dst2_local[i] = Y(i+1);
    }
    dst_local.release();
    dst2_local.release();
    for(Index i = 0; i < 3; ++i)
    {
        auto maxsumexp_local = maxsumexp[i].acquire(STARPU_W);
        for(Index j = 0; j < maxsumexp[i].nelems; j += 2)
        {
            maxsumexp_local[j] = Y(j+1);
            maxsumexp_local[j+1] = std::exp(Y(j+2) / Y{10});
        }
        maxsumexp_local.release();
    }
    // Check axis=0
    {
        starpu::softmax_inplace::submit<T>(1, 20, 3, maxsumexp[0], alpha, dst);
        softmax_inplace<T>(maxsumexp[0], alpha, dst2, 0);
        dst_local.acquire(STARPU_R);
        dst2_local.acquire(STARPU_R);
        for(Index i = 0; i < dst.nelems; ++i)
        {
            TEST_ASSERT(Y(dst_local[i]) == Y(dst2_local[i]));
        }
        dst_local.release();
        dst2_local.release();
    }
    // Check axis=1
    {
        starpu::softmax_inplace::submit<T>(3, 5, 4, maxsumexp[1], alpha, dst);
        softmax_inplace<T>(maxsumexp[1], alpha, dst2, 1);
        dst_local.acquire(STARPU_R);
        dst2_local.acquire(STARPU_R);
        for(Index i = 0; i < dst.nelems; ++i)
        {
            TEST_ASSERT(Y(dst_local[i]) == Y(dst2_local[i]));
        }
        dst_local.release();
        dst2_local.release();
    }
    // Check axis=2
    {
        starpu::softmax_inplace::submit<T>(12, 1, 5, maxsumexp[2], alpha, dst);
        softmax_inplace<T>(maxsumexp[2], alpha, dst2, 2);
        dst_local.acquire(STARPU_R);
        dst2_local.acquire(STARPU_R);
        for(Index i = 0; i < dst.nelems; ++i)
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
    constexpr Scalar alpha = 1.0;
    // Check normal execution
    check<T>();
    // Check throwing exceptions
    Tile<T> empty({});
    Tile<T> dst({3, 4, 5});
    Tile<T> maxsumexp[3] = {Tile<T>({2, 4, 5}), Tile<T>({2, 3, 5}),
        Tile<T>({2, 3, 4})};
    TEST_THROW(softmax_inplace<T>(empty, alpha, empty, 0));
    TEST_THROW(softmax_inplace<T>(maxsumexp[0], alpha, dst, 1));
    TEST_THROW(softmax_inplace<T>(maxsumexp[0], alpha, dst, 2));
    TEST_THROW(softmax_inplace<T>(dst, alpha, dst, 0));
    TEST_THROW(softmax_inplace<T>(maxsumexp[2], alpha, dst, 0));
    TEST_THROW(softmax_inplace<T>(maxsumexp[1], alpha, dst, 2));
    TEST_THROW(softmax_inplace<T>(maxsumexp[0], alpha, dst, -1));
    TEST_THROW(softmax_inplace<T>(maxsumexp[0], alpha, dst, 3));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::softmax_inplace::init();
    starpu::softmax_inplace::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
