/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/sgd_step.cc
 * SGD with momentum step on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/sgd_step.hh"
#include "nntile/starpu/sgd_step.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> grad({2, 3}), vel({2, 3}), p({2, 3});
    Tile<T> velr({2, 3}), pr({2, 3});
    auto gl = grad.acquire(STARPU_W);
    auto vl = vel.acquire(STARPU_W);
    auto pl = p.acquire(STARPU_W);
    auto vlr = velr.acquire(STARPU_W);
    auto plr = pr.acquire(STARPU_W);
    for(Index i = 0; i < grad.nelems; ++i)
    {
        gl[i] = Y(0.02 * (i + 1));
        vl[i] = Y(0.1 * (i + 1));
        pl[i] = Y(1.0 + 0.03 * i);
        vlr[i] = vl[i];
        plr[i] = pl[i];
    }
    gl.release();
    vl.release();
    pl.release();
    vlr.release();
    plr.release();

    Index num_iter = 2;
    Scalar momentum = 0.85, lr = 1e-2, wd = 1e-3, dampening = 0.0;
    bool nesterov = false;
    starpu::sgd_step.submit<std::tuple<T>>(num_iter, grad.nelems, momentum, lr,
            wd, dampening, nesterov, grad, vel, p);
    sgd_step<T>(num_iter, momentum, lr, wd, dampening, nesterov, grad, velr,
            pr);

    vl.acquire(STARPU_R);
    pl.acquire(STARPU_R);
    vlr.acquire(STARPU_R);
    plr.acquire(STARPU_R);
    for(Index i = 0; i < grad.nelems; ++i)
    {
        TEST_ASSERT(Y(vl[i]) == Y(vlr[i]));
        TEST_ASSERT(Y(pl[i]) == Y(plr[i]));
    }
    vl.release();
    pl.release();
    vlr.release();
    plr.release();
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
