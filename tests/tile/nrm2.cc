/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/nrm2.cc
 * NRM2 operation on Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
 * */

#include "nntile/tile/nrm2.hh"
#include "nntile/starpu/nrm2.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/scal.hh"
#include "nntile/starpu/hypot.hh"
#include "../testing.hh"
#include <cmath>
#include <limits>

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check()
{
    // Init data for checking
    constexpr T eps = std::numeric_limits<T>::epsilon();
    Tile<T> src({3, 4, 5}), norm({}), norm2({}), tmp({});
    T norm2_init = 2.0;
    T alpha = -0.5, beta = 2.0;
    auto src_local = src.acquire(STARPU_W);
    for(Index i = 0; i < src.nelems; ++i)
    {
        src_local[i] = T(i);
    }
    src_local.release();
    auto norm2_local = norm2.acquire(STARPU_W);
    norm2_local[0] = norm2_init;
    norm2_local.release();
    starpu::nrm2::submit<T>(src.nelems, src, norm);
    nrm2<T>(alpha, src, beta, norm2, tmp);
    norm2_local.acquire(STARPU_R);
    auto norm_local = norm.acquire(STARPU_R);
    T val1 = norm2_local[0]*norm2_local[0];
    T val2 = alpha*alpha*norm_local[0]*norm_local[0]
        + beta*beta*norm2_init*norm2_init;
    norm2_local.release();
    norm_local.release();
    TEST_ASSERT(std::abs(val1/val2-1.0) <= 10*eps);
}

template<typename T>
void validate()
{
    // Check normal execution
    check<T>();
    // Check throwing exceptions
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::nrm2::init();
    starpu::clear::init();
    starpu::scal::init();
    starpu::hypot::init();
    starpu::nrm2::restrict_where(STARPU_CPU);
    starpu::clear::restrict_where(STARPU_CPU);
    starpu::scal::restrict_where(STARPU_CPU);
    starpu::hypot::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

