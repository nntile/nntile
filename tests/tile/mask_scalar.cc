/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/mask_scalar.cc
 * mask_scalar on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/mask_scalar.hh"
#include "nntile/starpu/mask_scalar.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<bool_t> mask({2, 3});
    Tile<T> data({2, 3}), data_ref({2, 3});
    auto ml = mask.acquire(STARPU_W);
    auto dl = data.acquire(STARPU_W);
    auto drl = data_ref.acquire(STARPU_W);
    for(Index i = 0; i < mask.nelems; ++i)
    {
        ml[i] = bool_t((i % 3) != 0);
    }
    for(Index i = 0; i < data.nelems; ++i)
    {
        dl[i] = Y(i + 1);
        drl[i] = Y(i + 1);
    }
    ml.release();
    dl.release();
    drl.release();

    Scalar val = -9.0;
    starpu::mask_scalar.submit<std::tuple<T>>(
            data.matrix_shape[data.ndim][0],
            data.matrix_shape[data.ndim][1],
            mask, val, data);
    mask_scalar<T>(mask, val, data_ref, 0);

    dl.acquire(STARPU_R);
    drl.acquire(STARPU_R);
    for(Index i = 0; i < data.nelems; ++i)
    {
        TEST_ASSERT(Y(dl[i]) == Y(drl[i]));
    }
    dl.release();
    drl.release();
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
