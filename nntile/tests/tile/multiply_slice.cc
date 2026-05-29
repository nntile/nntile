/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/multiply_slice.cc
 * Tile wrappers for multiplication of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/multiply_slice.hh"
#include "nntile/starpu/multiply_slice.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check(Scalar alpha, const Tile<T> &src, Tile<T> &dst, Index axis)
{
    using Y = typename T::repr_t;
    std::vector<T> dst2_data(dst.nelems);
    Tile<T> dst2(dst, &dst2_data[0], dst.nelems);
    auto dst_local = dst.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        dst2_data[i] = dst_local[i];
    }
    dst_local.release();
    multiply_slice<T>(alpha, src, dst, axis);
    Index m = 1;
    for(Index i = 0; i < axis; ++i)
    {
        m *= dst.shape[i];
    }
    Index n = 1;
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        n *= dst.shape[i];
    }
    Index k = dst.shape[axis];
    starpu::multiply_slice.submit<std::tuple<T>>(m, n, k, alpha, src, dst2);
    starpu_task_wait_for_all();
    dst_local.acquire(STARPU_R);
    auto dst2_local = dst2.acquire(STARPU_R);
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
    using Y = typename T::repr_t;
    std::vector<Index> dst_shape{3, 4, 5, 6};
    for(Index axis = 0; axis < 4; ++axis)
    {
        std::vector<Index> src_shape;
        for(Index i = 0; i < static_cast<Index>(dst_shape.size()); ++i)
        {
            if(i != axis)
            {
                src_shape.push_back(dst_shape[i]);
            }
        }
        TileTraits src_traits(src_shape), dst_traits(dst_shape);
        Tile<T> src(src_traits), dst(dst_traits);
        auto sl = src.acquire(STARPU_W);
        auto dl = dst.acquire(STARPU_W);
        for(Index i = 0; i < src.nelems; ++i)
        {
            sl[i] = Y(0.1 * (i + 1));
        }
        for(Index i = 0; i < dst.nelems; ++i)
        {
            dl[i] = Y(0.2 * (i + 1));
        }
        sl.release();
        dl.release();
        check<T>(-0.5, src, dst, axis);
    }
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
