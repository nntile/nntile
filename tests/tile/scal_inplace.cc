/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/scal_inplace.cc
 * Inplace scal operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/scal_inplace.hh"
#include "nntile/starpu/scal_inplace.hh"
#include "../testing.hh"
#include <cmath>
#include <limits>

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check(Scalar alpha)
{
    using Y = typename T::repr_t;
    // Init data for checking
    Tile<T> data({3, 4, 5}), data2({3, 4, 5});
    auto data_local = data.acquire(STARPU_W);
    auto data2_local = data2.acquire(STARPU_W);
    for(Index i = 0; i < data.nelems; ++i)
    {
        data_local[i] = Y(i);
        data2_local[i] = Y(i);
    }
    data_local.release();
    data2_local.release();
    starpu::scal_inplace::submit<T>(data.nelems, alpha, data);
    scal_inplace<T>(alpha, data2);
    data_local.acquire(STARPU_R);
    data2_local.acquire(STARPU_R);
    for(Index i = 0; i < data.nelems; ++i)
    {
        TEST_ASSERT(Y(data_local[i]) == Y(data2_local[i]));
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
    starpu::scal_inplace::init();
    starpu::scal_inplace::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
