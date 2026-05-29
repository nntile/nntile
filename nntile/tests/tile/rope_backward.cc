/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/rope_backward.cc
 * Backward RoPE operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/rope_backward.hh"
#include "nntile/starpu/rope_backward.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> sin({2}), cos({2}), dy({4, 5}), dx({4, 5}), dx_ref({4, 5});
    auto sin_local = sin.acquire(STARPU_W);
    auto cos_local = cos.acquire(STARPU_W);
    auto dy_local = dy.acquire(STARPU_W);
    auto dx_local = dx.acquire(STARPU_W);
    auto dx_ref_local = dx_ref.acquire(STARPU_W);
    for(Index i = 0; i < sin.nelems; ++i)
    {
        sin_local[i] = Y(0.1 * (i+1));
        cos_local[i] = Y(0.2 * (i+1));
    }
    for(Index i = 0; i < dy.nelems; ++i)
    {
        dy_local[i] = Y(0.05 * (i+1));
        dx_local[i] = Y(-0.1 * (i+1));
        dx_ref_local[i] = dx_local[i];
    }
    sin_local.release();
    cos_local.release();
    dy_local.release();
    dx_local.release();
    dx_ref_local.release();

    starpu::rope_backward.submit<std::tuple<T>>(2, 5, sin, cos, dy, dx);
    rope_backward<T>(sin, cos, dy, dx_ref);

    dx_local.acquire(STARPU_R);
    dx_ref_local.acquire(STARPU_R);
    for(Index i = 0; i < dx.nelems; ++i)
    {
        TEST_ASSERT(Y(dx_local[i]) == Y(dx_ref_local[i]));
    }
    dx_local.release();
    dx_ref_local.release();
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
