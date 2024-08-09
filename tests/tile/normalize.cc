/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/normalize.cc
 * Normalize operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/normalize.hh"
#include "nntile/starpu/normalize.hh"
#include "../testing.hh"
#include <cmath>

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check()
{
    using Y = typename T::repr_t;
    // Init data for checking
    Tile<T> gamma_beta({2}), dst({3, 4, 5}), dst2({3, 4, 5});
    Tile<T> sumnorm[3] = {Tile<T>({2, 4, 5}), Tile<T>({2, 3, 5}),
        Tile<T>({2, 3, 4})};
    Y one = 1, zero = 0;
    auto gamma_beta_local = gamma_beta.acquire(STARPU_W);
    gamma_beta_local[0] = one;
    gamma_beta_local[1] = zero;
    gamma_beta_local.release();
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
        auto sumnorm_local = sumnorm[i].acquire(STARPU_W);
        for(Index j = 0; j < sumnorm[i].nelems; j += 2)
        {
            sumnorm_local[j] = Y(j+1) * Y(dst.shape[i]);
            sumnorm_local[j+1] = Y(j+2) * std::sqrt(Y(dst.shape[i]));
        }
        sumnorm_local.release();
    }
    // Check axis=0
    {
        starpu::normalize::submit<T>(1, 20, 3, 3, 1.0, gamma_beta, sumnorm[0],
                dst);
        normalize<T>(gamma_beta, sumnorm[0], dst2, 3, 1.0, 0);
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
        starpu::normalize::submit<T>(3, 5, 4, 4, 1.0, gamma_beta, sumnorm[1],
                dst);
        normalize<T>(gamma_beta, sumnorm[1], dst2, 4, 1.0, 1);
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
        starpu::normalize::submit<T>(12, 1, 5, 5, 1.0, gamma_beta, sumnorm[2],
                dst);
        normalize<T>(gamma_beta, sumnorm[2], dst2, 5, 1.0, 2);
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
    // Check normal execution
    check<T>();
    // Check throwing exceptions
    Tile<T> empty({});
    Tile<T> gamma_beta({2}), dst({3, 4, 5}), dst2({3, 4, 5});
    Tile<T> sumnorm[3] = {Tile<T>({2, 4, 5}), Tile<T>({2, 3, 5}),
        Tile<T>({2, 3, 4})};
    Scalar one = 1, zero = 0, eps=1e-16;
    TEST_THROW(normalize<T>(gamma_beta, gamma_beta, dst, dst.shape[0], eps,
                0));
    TEST_THROW(normalize<T>(gamma_beta, empty, empty, dst.shape[0], eps, 0));
    TEST_THROW(normalize<T>(gamma_beta, sumnorm[0], dst, 0, eps, 0));
    TEST_THROW(normalize<T>(gamma_beta, sumnorm[0], dst, dst.shape[0], zero,
                0));
    TEST_THROW(normalize<T>(gamma_beta, sumnorm[0], dst, dst.shape[0], -1, 0));
    TEST_THROW(normalize<T>(gamma_beta, dst, dst, dst.shape[0], eps, 0));
    TEST_THROW(normalize<T>(gamma_beta, sumnorm[2], dst, dst.shape[0], eps,
                0));
    TEST_THROW(normalize<T>(gamma_beta, sumnorm[1], dst, dst.shape[1], eps,
                2));
    TEST_THROW(normalize<T>(gamma_beta, sumnorm[0], dst, 1, eps, -1));
    TEST_THROW(normalize<T>(gamma_beta, sumnorm[0], dst, 1, eps, 3));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::normalize::init();
    starpu::normalize::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
