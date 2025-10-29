/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/sum.cc
 * Sum all elements of a Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/sum.hh"
#include "nntile/starpu/sum.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check()
{
    using Y = typename T::repr_t;
    // Init data for checking
    Tile<T> src({3, 4, 5});
    Tile<T> dst({});
    Tile<T> dst2({});
    auto src_local = src.acquire(STARPU_W);
    Scalar alpha = -1.0, beta = 0.5;
    for(Index i = 0; i < src.nelems; ++i)
    {
        src_local[i] = Y(i+1);
    }
    src_local.release();
    Y zero = 0;
    auto dst_local = dst.acquire(STARPU_W);
    auto dst2_local = dst2.acquire(STARPU_W);
    dst_local[0] = zero;
    dst2_local[0] = zero;
    dst_local.release();
    dst2_local.release();
    // Check starpu and tile versions
    {
        starpu::sum.submit<std::tuple<T>>(src.nelems, alpha, src, beta, dst);
        sum<T>(alpha, src, beta, dst2);
        auto dst_local = dst.acquire(STARPU_R);
        auto dst2_local = dst2.acquire(STARPU_R);
        TEST_ASSERT(Y(dst_local[0]) == Y(dst2_local[0]));
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
    Tile<T> src({3, 4, 5});
    Tile<T> dst({});
    Tile<T> dst_bad({1});
    Tile<T> empty({});
    TEST_THROW(sum<T>(1.0, src, 1.0, dst_bad));
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
