/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/scale_fiber.cc
 * scale_fiber on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/scale_fiber.hh"
#include "nntile/starpu/scale_fiber.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check(Scalar alpha, Index axis)
{
    using Y = typename T::repr_t;
    Tile<T> dst({3, 4, 5}), dst_ref({3, 4, 5});
    Tile<T> src({dst.shape[axis]});
    auto sl = src.acquire(STARPU_W);
    auto dl = dst.acquire(STARPU_W);
    auto drl = dst_ref.acquire(STARPU_W);
    for(Index i = 0; i < src.nelems; ++i)
    {
        sl[i] = Y(0.5 + i);
    }
    for(Index i = 0; i < dst.nelems; ++i)
    {
        dl[i] = Y(0);
        drl[i] = Y(0);
    }
    sl.release();
    dl.release();
    drl.release();

    Index m = dst.stride[axis];
    Index batch = src.matrix_shape[1][1];
    Index n = dst.matrix_shape[axis+1][1] / batch;
    Index k = dst.shape[axis];
    starpu::scale_fiber.submit<std::tuple<T>>(m, n, k, batch, alpha, src, dst);
    scale_fiber<T>(alpha, src, dst_ref, axis, 0);

    dl.acquire(STARPU_R);
    drl.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        TEST_ASSERT(Y(dl[i]) == Y(drl[i]));
    }
    dl.release();
    drl.release();
}

template<typename T>
void validate()
{
    check<T>(2.0, 0);
    check<T>(-0.5, 1);
    check<T>(1.25, 2);
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
