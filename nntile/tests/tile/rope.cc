/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/rope.cc
 * Rotary positional embedding on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/rope.hh"
#include "nntile/starpu/rope.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> sin({2}), cos({2}), src({4, 5}), dst({4, 5}), dst_ref({4, 5});
    auto sl = sin.acquire(STARPU_W);
    auto cl = cos.acquire(STARPU_W);
    auto srcl = src.acquire(STARPU_W);
    auto dstl = dst.acquire(STARPU_W);
    auto drefl = dst_ref.acquire(STARPU_W);
    for(Index i = 0; i < sin.nelems; ++i)
    {
        sl[i] = Y(0.1 * (i + 1));
        cl[i] = Y(0.2 * (i + 1));
    }
    for(Index i = 0; i < src.nelems; ++i)
    {
        srcl[i] = Y(0.03 * (i + 1));
        dstl[i] = Y(0);
        drefl[i] = Y(0);
    }
    sl.release();
    cl.release();
    srcl.release();
    dstl.release();
    drefl.release();

    Index m = sin.nelems;
    Index n = src.matrix_shape[sin.ndim][1];
    starpu::rope.submit<std::tuple<T>>(m, n, sin, cos, src, dst);
    rope<T>(sin, cos, src, dst_ref);

    dstl.acquire(STARPU_R);
    drefl.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        TEST_ASSERT(Y(dstl[i]) == Y(drefl[i]));
    }
    dstl.release();
    drefl.release();
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
