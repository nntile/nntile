/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: a virtual planner
 * builds one plan per batch from a drained snapshot, and a literal
 * executor issues exactly the planned operations.
 *
 * @file nntile/tests/nnhaul/swap_two_axes.cc
 * swap_two_axes (5-D axes 1/3 swap): direct kernel versus the NNHaul
 * wrapper on the same inputs, including the class-boundary shape and
 * the d4 == 1 matrix-transpose regime.
 *
 * @version 1.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/swap_two_axes.hh>
#include <nntile/nnhaul/ops/swap_two_axes.hh>

using nntile::Index;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index d0, Index d1, Index d2, Index d3, Index d4)
{
    std::size_t const nelems = static_cast<std::size_t>(d0) * d1 * d2 * d3
        * d4;
    using Y = typename T::repr_t;
    std::vector<T> src(nelems);
    std::vector<T> dst(nelems);
    for (std::size_t i = 0; i < nelems; ++i)
    {
        src[i] = Y(static_cast<double>(i % 1021) * 0.5 - 11.0);
        dst[i] = Y(-1.0);
    }
    auto cpu_dst = dst;
    std::cout << "Run kernel::swap_two_axes::cpu\n";
    nntile::kernel::swap_two_axes::cpu<T>(d0, d1, d2, d3, d4, src.data(),
        cpu_dst.data());
    for(std::size_t chunk : wrap::chunk_variants(nelems * sizeof(T)))
    {
        for(int cuda = 0; cuda < 2; ++cuda)
        {
            wrap::Chunked src_h(nelems * sizeof(T), chunk);
            wrap::Chunked dst_h(nelems * sizeof(T), chunk);
            wrap::store(src_h.handle, src);
            wrap::store(dst_h.handle, dst);
            nntile::nnhaul::swap_two_axes.submit<std::tuple<T>>(
                cuda ? wrap::cuda_worker : wrap::cpu_worker, d0, d1, d2, d3,
                d4, src_h.handle, dst_h.handle);
            wrap::expect_equal(cuda ? "swap_two_axes cuda" :
                "swap_two_axes cpu", cpu_dst,
                wrap::load<T>(dst_h.handle, dst.size()));
        }
    }
}

int main()
{
    wrap::Session session;
    (void)session;
    // Small, d4 == 1 (a d1 x d3 transpose per (i0, i2)), and the
    // class-boundary shape. The one-class rule has no unchunked
    // scratch, so this op's universal staging fallback (gather, classic
    // full-view kernel, scatter) only serves tiles one whole class
    // covers: a multi-chunk scratch handed to the full-view kernel is
    // the engine's loud get_ptr caller defect, and the panel-style
    // kernel variant that would unlock chunk-local sub-calls is future
    // work (see the decomposition note in the nntile wrapper). The
    // last shape is exactly chunk_class_bytes() of fp32 - the largest
    // tile the fallback laws.
    check<fp32_t>(2, 3, 4, 5, 6);
    check<fp32_t>(1, 17, 1, 13, 1);
    check<fp32_t>(3, 5, 2, 7, 1);
    check<fp64_t>(3, 5, 2, 7, 1);
    check<fp32_t>(4, 64, 32, 8, 16);
    return 0;
}
