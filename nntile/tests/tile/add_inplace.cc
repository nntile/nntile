/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/add_inplace.cc
 * Add inplace operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/add_inplace.hh"
#include "nntile/starpu/add_inplace.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> src({2, 3, 4}), dst({2, 3, 4}), dst_ref({2, 3, 4});
    auto src_local = src.acquire(STARPU_W);
    auto dst_local = dst.acquire(STARPU_W);
    auto dst_ref_local = dst_ref.acquire(STARPU_W);
    for(Index i = 0; i < src.nelems; ++i)
    {
        src_local[i] = Y(0.1 * (i+1));
        dst_local[i] = Y(0.2 * (i+1));
        dst_ref_local[i] = dst_local[i];
    }
    src_local.release();
    dst_local.release();
    dst_ref_local.release();

    Scalar alpha = -0.5, beta = 0.3;
    starpu::add_inplace.submit<std::tuple<T>>(src.nelems, alpha, src, beta, dst);
    add_inplace<T>(alpha, src, beta, dst_ref);

    dst_local.acquire(STARPU_R);
    dst_ref_local.acquire(STARPU_R);
    for(Index i = 0; i < src.nelems; ++i)
    {
        TEST_ASSERT(Y(dst_local[i]) == Y(dst_ref_local[i]));
    }
    dst_local.release();
    dst_ref_local.release();

    TEST_THROW(add_inplace<T>(alpha, Tile<T>({2}), beta, Tile<T>({3})));
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
