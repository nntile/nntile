/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
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
 * @date 2022-12-02
 * */

#include "nntile/tile/nrm2.hh"
#include "nntile/starpu/nrm2.hh"
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
    Tile<T> src({3, 4, 5}), norm({}), norm2({});
    auto src_local = src.acquire(STARPU_W);
    for(Index i = 0; i < src.nelems; ++i)
    {
        src_local[i] = T(i);
    }
    src_local.release();
    starpu::nrm2::submit<T>(src.nelems, src, norm);
    nrm2<T>(src, norm2);
    auto norm_local = norm.acquire(STARPU_R);
    auto norm2_local = norm.acquire(STARPU_R);
    TEST_ASSERT(norm_local[0] == norm2_local[0]);
    norm_local.release();
    norm2_local.release();
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
    starpu::nrm2::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

