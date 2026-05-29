/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/add_fiber_inplace.cc
 * Tile wrappers for addition of a tensor and a broadcasted fiber (in-place)
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/add_fiber_inplace.hh"
#include "nntile/starpu/add_fiber_inplace.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check(Scalar alpha, Scalar beta, Index axis)
{
    using Y = typename T::repr_t;
    Tile<T> dst({3, 4, 5}), dst2({3, 4, 5});
    Tile<T> src({dst.shape[axis]});
    auto src_local = src.acquire(STARPU_W);
    auto dst_local = dst.acquire(STARPU_W);
    auto dst2_local = dst2.acquire(STARPU_W);
    for(Index i = 0; i < src.nelems; ++i)
    {
        src_local[i] = Y(i + 1);
    }
    for(Index i = 0; i < dst.nelems; ++i)
    {
        dst_local[i] = Y(0.25 * (i + 1));
        dst2_local[i] = dst_local[i];
    }
    src_local.release();
    dst_local.release();
    dst2_local.release();

    Index m = dst.stride[axis];
    Index batch = src.matrix_shape[1][1];
    Index n = dst.matrix_shape[axis+1][1] / batch;
    Index k = dst.shape[axis];
    starpu::add_fiber_inplace.submit<std::tuple<T>>(m, n, k, batch, alpha, src,
            beta, dst);
    add_fiber_inplace<T>(alpha, src, beta, dst2, axis, 0);

    dst_local.acquire(STARPU_R);
    dst2_local.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        TEST_ASSERT(Y(dst_local[i]) == Y(dst2_local[i]));
    }
    dst_local.release();
    dst2_local.release();
}

template<typename T>
void validate()
{
    check<T>(1.0, 0.5, 2);
    check<T>(-1.0, 1.0, 0);
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
