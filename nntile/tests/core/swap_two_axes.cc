/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file tests/core/swap_two_axes.cc
 * swap_two_axes operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/core/swap_two_axes.hh"
#include "nntile/core/swap_two_axes_decompose.hh"
#include "nntile/starpu/swap_two_axes.hh"
#include "../testing.hh"

#include <numeric>
#include <vector>

using namespace nntile;
using namespace nntile::core;

namespace
{

template<typename T>
void reference_swap_5d(
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const std::vector<T> &src,
    std::vector<T> &dst)
{
    for (Index i0 = 0; i0 < d0; ++i0)
    {
        for (Index i1 = 0; i1 < d1; ++i1)
        {
            for (Index i2 = 0; i2 < d2; ++i2)
            {
                for (Index i3 = 0; i3 < d3; ++i3)
                {
                    for (Index i4 = 0; i4 < d4; ++i4)
                    {
                        const Index src_idx =
                            ((((i0 * d1 + i1) * d2 + i2) * d3 + i3) * d4 +
                                i4);
                        const Index dst_idx =
                            ((((i0 * d3 + i3) * d2 + i2) * d1 + i1) * d4 +
                                i4);
                        dst[static_cast<size_t>(dst_idx)] =
                            src[static_cast<size_t>(src_idx)];
                    }
                }
            }
        }
    }
}

} // namespace

template<typename T>
void validate_5d_kernel(Index d0, Index d1, Index d2, Index d3, Index d4)
{
    using Y = typename T::repr_t;
    const Index nelems = d0 * d1 * d2 * d3 * d4;
    Tile<T> src({nelems}), dst({nelems}), dst_ref({nelems});
    std::vector<T> host_src(static_cast<size_t>(nelems));
    std::vector<T> host_ref(static_cast<size_t>(nelems));
    for (Index i = 0; i < nelems; ++i)
    {
        host_src[static_cast<size_t>(i)] = Y(i + 1);
    }
    reference_swap_5d(d0, d1, d2, d3, d4, host_src, host_ref);

    auto src_local = src.acquire(STARPU_W);
    auto dst_local = dst.acquire(STARPU_W);
    auto dst_ref_local = dst_ref.acquire(STARPU_W);
    for (Index i = 0; i < nelems; ++i)
    {
        src_local[i] = host_src[static_cast<size_t>(i)];
        dst_local[i] = Y(-1);
        dst_ref_local[i] = host_ref[static_cast<size_t>(i)];
    }
    src_local.release();
    dst_local.release();
    dst_ref_local.release();

    starpu::swap_two_axes.submit<std::tuple<T>>(
        -1,
        d0,
        d1,
        d2,
        d3,
        d4,
        src,
        dst);
    starpu_task_wait_for_all();

    dst_local.acquire(STARPU_R);
    dst_ref_local.acquire(STARPU_R);
    for (Index i = 0; i < nelems; ++i)
    {
        TEST_ASSERT(Y(dst_local[i]) == Y(dst_ref_local[i]));
    }
    dst_local.release();
    dst_ref_local.release();
}

template<typename T>
void validate_nd_tile(
    const std::vector<Index> &shape,
    Index dim0,
    Index dim1)
{
    using Y = typename T::repr_t;
    const SwapTwoAxesDecomposition decomp =
        decompose_swap_axes(shape, dim0, dim1);
    const auto &out_shape = decomp.output_shape;
    const Index nelems = std::accumulate(
        shape.begin(),
        shape.end(),
        Index(1),
        std::multiplies<>());

    Tile<T> src(shape), dst(out_shape), dst_ref(out_shape);
    std::vector<T> host_src(static_cast<size_t>(nelems));
    std::vector<T> host_ref(static_cast<size_t>(nelems));
    for (Index i = 0; i < nelems; ++i)
    {
        host_src[static_cast<size_t>(i)] = Y(i + 1);
    }
    const auto &d = decomp.sizes_5d;
    reference_swap_5d(
        d[0],
        d[1],
        d[2],
        d[3],
        d[4],
        host_src,
        host_ref);

    auto src_local = src.acquire(STARPU_W);
    auto dst_ref_local = dst_ref.acquire(STARPU_W);
    for (Index i = 0; i < nelems; ++i)
    {
        src_local[i] = host_src[static_cast<size_t>(i)];
        dst_ref_local[i] = host_ref[static_cast<size_t>(i)];
    }
    src_local.release();
    dst_ref_local.release();

    swap_two_axes<T>(-1, src, dst, dim0, dim1);

    auto dst_local = dst.acquire(STARPU_R);
    dst_ref_local.acquire(STARPU_R);
    for (Index i = 0; i < nelems; ++i)
    {
        TEST_ASSERT(Y(dst_local[i]) == Y(dst_ref_local[i]));
    }
    dst_local.release();
    dst_ref_local.release();
}

int main(int argc, char **argv)
{
    int ncpu = 1, ncuda = 0, ooc = 0, verbose = 0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    validate_5d_kernel<fp32_t>(2, 3, 1, 4, 2);
    validate_5d_kernel<fp32_t>(1, 5, 1, 3, 1);
    validate_5d_kernel<fp64_t>(2, 2, 2, 2, 2);

    validate_nd_tile<fp32_t>({5, 3}, 0, 1);
    validate_nd_tile<fp32_t>({2, 8, 4, 16}, 1, 2);
    validate_nd_tile<fp32_t>({2, 4, 6, 8}, 2, 3);
    return 0;
}
