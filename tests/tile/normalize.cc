/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/normalize.cc
 * Normalize operation on Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-03
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
    // Init data for checking
    Tile<T> gamma_beta({2}), dst({3, 4, 5}), dst2({3, 4, 5});
    Tile<T> sumnorm[3] = {Tile<T>({2, 4, 5}), Tile<T>({2, 3, 5}),
        Tile<T>({2, 3, 4})};
    T one = 1, zero = 0;
    auto gamma_beta_local = gamma_beta.acquire(STARPU_W);
    gamma_beta_local[0] = one;
    gamma_beta_local[1] = zero;
    gamma_beta_local.release();
    auto dst_local = dst.acquire(STARPU_W);
    auto dst2_local = dst2.acquire(STARPU_W);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        dst_local[i] = T(i+1);
        dst2_local[i] = T(i+1);
    }
    dst_local.release();
    dst2_local.release();
    for(Index i = 0; i < 3; ++i)
    {
        auto sumnorm_local = sumnorm[i].acquire(STARPU_W);
        for(Index j = 0; j < sumnorm[i].nelems; j += 2)
        {
            sumnorm_local[j] = T(j+1) * T(dst.shape[i]);
            sumnorm_local[j+1] = T(j+2) * std::sqrt(T(dst.shape[i]));
        }
        sumnorm_local.release();
    }
    // Check axis=0
    {
        starpu::normalize::submit<T>(1, 20, 3, 3, one, gamma_beta, sumnorm[0],
                dst);
        normalize<T>(gamma_beta, sumnorm[0], dst2, 3, one, 0);
        dst_local.acquire(STARPU_R);
        dst2_local.acquire(STARPU_R);
        for(Index i = 0; i < dst.nelems; ++i)
        {
            TEST_ASSERT(dst_local[i] == dst2_local[i]);
        }
        dst_local.release();
        dst2_local.release();
    }
    // Check axis=1
    {
        starpu::normalize::submit<T>(3, 5, 4, 4, one, gamma_beta, sumnorm[1],
                dst);
        normalize<T>(gamma_beta, sumnorm[1], dst2, 4, one, 1);
        dst_local.acquire(STARPU_R);
        dst2_local.acquire(STARPU_R);
        for(Index i = 0; i < dst.nelems; ++i)
        {
            TEST_ASSERT(dst_local[i] == dst2_local[i]);
        }
        dst_local.release();
        dst2_local.release();
    }
    // Check axis=2
    {
        starpu::normalize::submit<T>(12, 1, 5, 5, one, gamma_beta, sumnorm[2],
                dst);
        normalize<T>(gamma_beta, sumnorm[2], dst2, 5, one, 2);
        dst_local.acquire(STARPU_R);
        dst2_local.acquire(STARPU_R);
        for(Index i = 0; i < dst.nelems; ++i)
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
    Tile<T> empty({});
    Tile<T> gamma_beta({2}), dst({3, 4, 5}), dst2({3, 4, 5});
    Tile<T> sumnorm[3] = {Tile<T>({2, 4, 5}), Tile<T>({2, 3, 5}),
        Tile<T>({2, 3, 4})};
    T one = 1, zero = 0, eps=1e-16;
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
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

