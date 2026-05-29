/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/hypot_inplace.cc
 * hypot_inplace on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/hypot_inplace.hh"
#include "nntile/starpu/hypot_inplace.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> src({2, 3, 4}), dst({2, 3, 4}), dst_ref({2, 3, 4});
    auto sl = src.acquire(STARPU_W);
    auto dl = dst.acquire(STARPU_W);
    auto drl = dst_ref.acquire(STARPU_W);
    for(Index i = 0; i < src.nelems; ++i)
    {
        sl[i] = Y(0.3 * (i + 1));
        dl[i] = Y(0.4 * (i + 1));
        drl[i] = Y(0.4 * (i + 1));
    }
    sl.release();
    dl.release();
    drl.release();

    Scalar alpha = 0.7, beta = 1.1;
    starpu::hypot_inplace.submit<std::tuple<T>>(src.nelems, alpha, src, beta,
            dst);
    hypot_inplace<T>(alpha, src, beta, dst_ref);

    dl.acquire(STARPU_R);
    drl.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
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
