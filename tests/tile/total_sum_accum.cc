/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/total_sum_accum.cc
 * total_sum_accum on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/total_sum_accum.hh"
#include "nntile/starpu/total_sum_accum.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> logsumexp({2, 2});
    Tile<T> src({3, 2, 2});
    Tile<nntile::int64_t> class_labels({2, 2});
    Tile<fp32_t> val({}), val_ref({});

    auto lel = logsumexp.acquire(STARPU_W);
    auto sl = src.acquire(STARPU_W);
    auto cl = class_labels.acquire(STARPU_W);
    for(Index i = 0; i < logsumexp.nelems; ++i)
    {
        lel[i] = Y(0.1 * (i + 1));
    }
    for(Index i = 0; i < src.nelems; ++i)
    {
        sl[i] = Y(0.05 * (i + 1));
    }
    cl[0] = nntile::int64_t(0);
    cl[1] = nntile::int64_t(1);
    cl[2] = nntile::int64_t(2);
    cl[3] = nntile::int64_t(0);
    lel.release();
    sl.release();
    cl.release();

    auto vl = val.acquire(STARPU_W);
    auto vlr = val_ref.acquire(STARPU_W);
    vl[0] = Y(0);
    vlr[0] = Y(0);
    vl.release();
    vlr.release();

    Scalar alpha = 1.0;
    Index ignore_index = -1;
    starpu::total_sum_accum.submit<std::tuple<T>>(alpha, src.shape[0],
            logsumexp.nelems, ignore_index, logsumexp, src, class_labels, val);
    total_sum_accum<T>(alpha, logsumexp, src, class_labels, val_ref,
            ignore_index);

    using ValY = typename fp32_t::repr_t;
    vl.acquire(STARPU_R);
    vlr.acquire(STARPU_R);
    TEST_ASSERT(ValY(vl[0]) == ValY(vlr[0]));
    vl.release();
    vlr.release();
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
