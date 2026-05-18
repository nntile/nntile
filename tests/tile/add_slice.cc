/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/add_slice.cc
 * Tile wrappers for out-of-place addition of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/add_slice.hh"
#include "nntile/starpu/add_slice.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check(Scalar alpha, Scalar beta, Index axis)
{
    using Y = typename T::repr_t;
    std::vector<Index> dst_shape{3, 4, 5, 6};
    std::vector<Index> src1_shape;
    for(Index i = 0; i < static_cast<Index>(dst_shape.size()); ++i)
    {
        if(i != axis)
        {
            src1_shape.push_back(dst_shape[i]);
        }
    }
    TileTraits src1_traits(src1_shape), dst_traits(dst_shape);
    Tile<T> src1(src1_traits), src2(dst_traits), dst(dst_traits),
        dst_ref(dst_traits);
    auto s1 = src1.acquire(STARPU_W);
    auto s2 = src2.acquire(STARPU_W);
    auto d = dst.acquire(STARPU_W);
    auto dr = dst_ref.acquire(STARPU_W);
    for(Index i = 0; i < src1.nelems; ++i)
    {
        s1[i] = Y(2 * i + 1);
    }
    for(Index i = 0; i < src2.nelems; ++i)
    {
        s2[i] = Y(i + 1);
        d[i] = Y(0);
        dr[i] = Y(0);
    }
    s1.release();
    s2.release();
    d.release();
    dr.release();

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
    starpu::add_slice.submit<std::tuple<T>>(m, n, k, alpha, src1, beta, src2,
            dst);
    add_slice<T>(alpha, src1, beta, src2, dst_ref, axis);

    d.acquire(STARPU_R);
    dr.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        TEST_ASSERT(Y(d[i]) == Y(dr[i]));
    }
    d.release();
    dr.release();
}

template<typename T>
void validate()
{
    check<T>(1.0, 1.0, 0);
    check<T>(-0.5, 2.0, 2);
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
