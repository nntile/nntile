/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/scal.cc
 * SCAL operation on Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-03-29
 * */

#include "nntile/tile/scal.hh"
#include "nntile/starpu/scal.hh"
#include "../testing.hh"
#include <cmath>
#include <limits>

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check(T alpha)
{
    // Init data for checking
    Tile<T> data({3, 4, 5}), data2({3, 4, 5});
    auto data_local = data.acquire(STARPU_W);
    auto data2_local = data2.acquire(STARPU_W);
    for(Index i = 0; i < data.nelems; ++i)
    {
        data_local[i] = T(i);
        data2_local[i] = T(i);
    }
    data_local.release();
    data2_local.release();
    starpu::scal::submit<T>(alpha, data.nelems, data);
    scal<T>(alpha, data2);
    data_local.acquire(STARPU_R);
    data2_local.acquire(STARPU_R);
    for(Index i = 0; i < data.nelems; ++i)
    {
        TEST_ASSERT(data_local[i] == data2_local[i]);
    }
    data_local.release();
    data2_local.release();
}

template<typename T>
void validate()
{
    // Check normal execution
    check<T>(-1.0);
    check<T>(2.0);
    // Check throwing exceptions
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::scal::init();
    starpu::scal::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

