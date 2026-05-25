/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/subtract_indexed_outputs.cc
 * subtract_indexed_outputs on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/subtract_indexed_outputs.hh"
#include "nntile/starpu/subtract_indexed_outputs.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<nntile::int64_t> labels({2, 2});
    Tile<T> dst({3, 2, 2}), dst_ref({3, 2, 2});
    auto ll = labels.acquire(STARPU_W);
    ll[0] = nntile::int64_t(0);
    ll[1] = nntile::int64_t(1);
    ll[2] = nntile::int64_t(2);
    ll[3] = nntile::int64_t(0);
    ll.release();

    auto dl = dst.acquire(STARPU_W);
    auto drl = dst_ref.acquire(STARPU_W);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        dl[i] = Y(1.0 + 0.1 * i);
        drl[i] = Y(1.0 + 0.1 * i);
    }
    dl.release();
    drl.release();

    Scalar val = 0.5;
    Index ignore_index = -1;
    starpu::subtract_indexed_outputs.submit<std::tuple<T>>(dst.shape[0],
            labels.nelems, ignore_index, val, labels, dst);
    subtract_indexed_outputs<T>(val, labels, dst_ref, ignore_index);

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
