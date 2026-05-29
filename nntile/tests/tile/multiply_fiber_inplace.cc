/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/multiply_fiber_inplace.cc
 * In-place per-element product of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/multiply_fiber_inplace.hh"
#include "nntile/starpu/multiply_fiber_inplace.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check(Scalar alpha, Index axis)
{
    using Y = typename T::repr_t;
    Tile<T> dst({3, 4, 5}), dst2({3, 4, 5});
    Tile<T> src({dst.shape[axis]});
    auto sl = src.acquire(STARPU_W);
    auto dl = dst.acquire(STARPU_W);
    auto d2l = dst2.acquire(STARPU_W);
    for(Index i = 0; i < src.nelems; ++i)
    {
        sl[i] = Y(i + 1);
    }
    for(Index i = 0; i < dst.nelems; ++i)
    {
        dl[i] = Y(0.3 * (i + 1));
        d2l[i] = dl[i];
    }
    sl.release();
    dl.release();
    d2l.release();

    Index m = dst.stride[axis];
    Index n = dst.matrix_shape[axis+1][1];
    Index k = dst.shape[axis];
    starpu::multiply_fiber_inplace.submit<std::tuple<T>>(m, n, k, alpha, src,
            dst);
    multiply_fiber_inplace<T>(alpha, src, dst2, axis);

    dl.acquire(STARPU_R);
    d2l.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        TEST_ASSERT(Y(dl[i]) == Y(d2l[i]));
    }
    dl.release();
    d2l.release();
}

template<typename T>
void validate()
{
    check<T>(0.5, 1);
    check<T>(-1.0, 2);
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
