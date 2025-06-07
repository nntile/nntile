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

#include "nntile/context.hh"
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
    starpu::VariableHandle tmp_index(sizeof(nntile::int64_t)*2*3);
    starpu::randn.submit<std::tuple<T>>(3, dst.nelems, seed, mean, stddev, start,
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
    // Initialize StarPU
    int ncpu=1, ncuda=0, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
