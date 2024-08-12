/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/add_slice.cc
 * Tile wrappers for addition of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

#include "nntile/tile/add_slice.hh"
#include "nntile/starpu/add_slice.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check(Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst,
        Index axis)
{
    using Y = typename T::repr_t;
    std::vector<T> dst2_data(dst.nelems);
    auto dst_local = dst.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        dst_local[i] = dst2_data[i];
    }
    dst_local.release();
    Tile<T> dst2(dst, &dst2_data[0], dst.nelems);
    add_slice<T>(alpha, src, beta, dst, axis);
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
    starpu::add_slice::submit<T>(m, n, k, alpha, src, beta, dst2);
    starpu_task_wait_for_all();
    auto dst2_local = dst.acquire(STARPU_R);
    dst_local.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        TEST_ASSERT(Y(dst_local[i]) == Y(dst2_local[i]));
    }
}

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    std::vector<Index> A_shape{3, 4, 5, 6}, b0_shape{4, 5, 6},
        b1_shape{3, 5, 6}, b2_shape{3, 4, 6}, b3_shape{3, 4, 5};
    TileTraits A_traits(A_shape), b0_traits(b0_shape), b1_traits(b1_shape),
              b2_traits(b2_shape), b3_traits(b3_shape);
    std::vector<T> A_data(A_traits.nelems), b0_data(b0_traits.nelems),
        b1_data(b1_traits.nelems), b2_data(b2_traits.nelems),
        b3_data(b3_traits.nelems);
    for(Index i = 0; i < A_traits.nelems; ++i)
    {
        A_data[i] = Y(i+1);
    }
    for(Index i = 0; i < b0_traits.nelems; ++i)
    {
        b0_data[i] = Y(2*i+1);
    }
    for(Index i = 0; i < b1_traits.nelems; ++i)
    {
        b1_data[i] = Y(3*i+1);
    }
    for(Index i = 0; i < b2_traits.nelems; ++i)
    {
        b2_data[i] = Y(4*i+1);
    }
    for(Index i = 0; i < b3_traits.nelems; ++i)
    {
        b3_data[i] = Y(5*i+1);
    }
    Tile<T> A(A_traits, &A_data[0], A_traits.nelems),
        b0(b0_traits, &b0_data[0], b0_traits.nelems),
        b1(b1_traits, &b1_data[0], b1_traits.nelems),
        b2(b2_traits, &b2_data[0], b2_traits.nelems),
        b3(b3_traits, &b3_data[0], b3_traits.nelems);
    // Compare results of tile::add_slice and starpu::add_slice::submit
    check<T>(-1.0, b0, 1.0, A, 0);
    check<T>(1.0, b1, -1.0, A, 1);
    check<T>(2.0, b2, 0.5, A, 2);
    check<T>(-2.0, b3, 0.0, A, 3);
    // Checking throwing exceptions
    TEST_THROW(add_slice<T>(1.0, A, 0.0, A, 0));
    TEST_THROW(add_slice<T>(1.0, b0, 0.0, A, -1));
    TEST_THROW(add_slice<T>(1.0, b0, 0.0, A, 1));
    TEST_THROW(add_slice<T>(1.0, b3, 0.0, A, 2));
    TEST_THROW(add_slice<T>(1.0, b3, 0.0, A, 4));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::add_slice::init();
    starpu::add_slice::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
