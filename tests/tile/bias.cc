/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/bias.cc
 * Bias operation on Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-03
 * */

#include "nntile/tile/bias.hh"
#include "nntile/starpu/bias.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    std::vector<T> dst2_data(dst.nelems);
    auto dst_local = dst.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        dst_local[i] = dst2_data[i];
    }
    dst_local.release();
    Tile<T> dst2(dst, &dst2_data[0], dst.nelems);
    bias<T>(src, dst, axis);
    Index m = 1;
    for(Index i = 0; i < axis; ++i)
    {
        m *= dst.shape[i];
    }
    Index n = 1;
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        n *= dst.shape[i];
    }
    Index k = dst.shape[axis];
    starpu::bias::submit<T>(m, n, k, src, dst2);
    starpu_task_wait_for_all();
    auto dst2_local = dst.acquire(STARPU_R);
    dst_local.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        TEST_ASSERT(dst_local[i] == dst2_local[i]);
    }
}

template<typename T>
void validate()
{
    std::vector<Index> A_shape{3, 4, 5, 6}, b0_shape{4, 5, 6},
        b1_shape{3, 5, 6}, b2_shape{3, 4, 6}, b3_shape{3, 4, 5};
    TileTraits A_traits(A_shape), b0_traits(b0_shape), b1_traits(b1_shape),
              b2_traits(b2_shape), b3_traits(b3_shape);
    std::vector<T> A_data(A_traits.nelems), b0_data(b0_traits.nelems),
        b1_data(b1_traits.nelems), b2_data(b2_traits.nelems),
        b3_data(b3_traits.nelems);
    for(Index i = 0; i < A_traits.nelems; ++i)
    {
        A_data[i] = T(i+1);
    }
    for(Index i = 0; i < b0_traits.nelems; ++i)
    {
        b0_data[i] = T(2*i+1);
    }
    for(Index i = 0; i < b1_traits.nelems; ++i)
    {
        b1_data[i] = T(3*i+1);
    }
    for(Index i = 0; i < b2_traits.nelems; ++i)
    {
        b2_data[i] = T(4*i+1);
    }
    for(Index i = 0; i < b3_traits.nelems; ++i)
    {
        b3_data[i] = T(5*i+1);
    }
    Tile<T> A(A_traits, &A_data[0], A_traits.nelems),
        b0(b0_traits, &b0_data[0], b0_traits.nelems),
        b1(b1_traits, &b1_data[0], b1_traits.nelems),
        b2(b2_traits, &b2_data[0], b2_traits.nelems),
        b3(b3_traits, &b3_data[0], b3_traits.nelems);
    // Compare results of tile::bias and starpu::bias::submit
    check<T>(b0, A, 0);
    check<T>(b1, A, 1);
    check<T>(b2, A, 2);
    check<T>(b3, A, 3);
    // Checking throwing exceptions
    TEST_THROW(bias(A, A, 0));
    TEST_THROW(bias(b0, A, -1));
    TEST_THROW(bias(b0, A, 1));
    TEST_THROW(bias(b3, A, 2));
    TEST_THROW(bias(b3, A, 4));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::bias::init();
    starpu::bias::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

