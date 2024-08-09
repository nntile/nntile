/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/randn.cc
 * Randn operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/randn.hh"
#include "nntile/starpu/randn.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    // Only CPU implementation is check, as there is no other backend yet
    Tile<T> dst({3, 4, 5}), dst2(dst.shape);
    std::vector<Index> start{1, 1, 1}, underlying_shape{5, 6, 7};
    unsigned long long seed = -1;
    Scalar mean = 1, stddev = 2;
    // Check some valid parameters
    starpu::VariableHandle tmp_index(sizeof(nntile::int64_t)*2*3, STARPU_R);
    starpu::randn::submit<T>(3, dst.nelems, seed, mean, stddev, start,
        dst.shape, dst.stride, underlying_shape, dst, tmp_index);
    randn(dst2, start, underlying_shape, seed, mean, stddev);
    auto dst_local = dst.acquire(STARPU_R);
    auto dst2_local = dst.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        TEST_ASSERT(Y(dst_local[i]) == Y(dst2_local[i]));
    }
    dst_local.release();
    dst2_local.release();
    // Check scalar tile
    Tile<T> scalar({}), scalar2({});
    starpu::Handle null_handle;
    starpu::randn::submit<T>(0, 1, seed, mean, stddev, scalar.shape,
            scalar.shape, scalar.shape, scalar.shape, scalar, null_handle);
    randn(scalar2, scalar.shape, scalar.shape, seed, mean, stddev);
    auto scalar_local = scalar.acquire(STARPU_R);
    auto scalar2_local = scalar2.acquire(STARPU_R);
    TEST_ASSERT(Y(scalar_local[0]) == Y(scalar2_local[0]));
    scalar_local.release();
    scalar2_local.release();
    // Check throwing exceptions
    std::vector<Index> ind2(2);
    TEST_THROW(randn<T>(dst, ind2, underlying_shape, seed, mean, stddev));
    TEST_THROW(randn<T>(dst, start, ind2, seed, mean, stddev));
    start[0] = -1;
    TEST_THROW(randn<T>(dst, start, underlying_shape, seed, mean, stddev));
    start[0] = 3;
    TEST_THROW(randn<T>(dst, start, underlying_shape, seed, mean, stddev));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::randn::init();
    starpu::randn::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
