/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/transpose.cc
 * Transpose operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/transpose.hh"
#include "nntile/starpu/transpose.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> src({3, 5}), dst({5, 3}), dst_ref({5, 3});
    auto src_local = src.acquire(STARPU_W);
    auto dst_local = dst.acquire(STARPU_W);
    auto dst_ref_local = dst_ref.acquire(STARPU_W);
    for(Index i = 0; i < src.nelems; ++i)
    {
        src_local[i] = Y(i+1);
    }
    for(Index i = 0; i < dst.nelems; ++i)
    {
        dst_local[i] = Y(-i-1);
        dst_ref_local[i] = dst_local[i];
    }
    src_local.release();
    dst_local.release();
    dst_ref_local.release();

    Scalar alpha = 0.5;
    starpu::transpose.submit<std::tuple<T>>(3, 5, alpha, src, dst);
    transpose<T>(alpha, src, dst_ref, 1);

    dst_local.acquire(STARPU_R);
    dst_ref_local.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        TEST_ASSERT(Y(dst_local[i]) == Y(dst_ref_local[i]));
    }
    dst_local.release();
    dst_ref_local.release();

    TEST_THROW(transpose<T>(alpha, src, dst_ref, 0));
}

int main(int argc, char **argv)
{
    int ncpu=1, ncuda=0, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
