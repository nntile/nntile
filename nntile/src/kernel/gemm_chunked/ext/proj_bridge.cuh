/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm_chunked/ext/proj_bridge.cuh
 * Piece `proj` on top of the chunked bridge (gemm_chunked_bridge.cuh): a
 * schedule whose last wave leaves no SM idle.
 *
 * A grid of N output tiles on U SMs (one 256-thread threadblock per SM)
 * runs in ceil(N/U) waves; when U does not divide N the last wave fills
 * only N mod U SMs. The schedule is three launches on the caller's
 * stream:
 *
 *   1. kernel::GemmWaveFull: the first (ceil(N/U) - 1) * U tiles exactly
 *      as the plain bridge computes them (one threadblock each, whole
 *      k, straight to C). It is the bridge's kernel text with one more
 *      early exit; the mainloop and the epilogue are untouched, because
 *      the compiler's schedule of that loop is worth a few percent and
 *      moves with any edit around it.
 *   2. kernel::GemmWaveTail: the remaining R tiles (R < U) on all U
 *      threadblocks. Each tile is cut once, at k panel `split`: its head
 *      goes to a threadblock of its own, its tail to one of the E = U - R
 *      spare threadblocks, which walk the tails round-robin (slot j of
 *      extra e is tail tile e + j E) and so stay in lockstep with each
 *      other, as the heads are: every threadblock of a tier reads the
 *      same k panel at the same time, and the operands stay shared
 *      through L2 as in a full wave (a plain stream of the tiles over U
 *      threadblocks makes every threadblock read a different panel: four
 *      times the DRAM traffic of a wave, and the wave runs memory-bound).
 *   3. kernel::GemmWaveFixup: per tail tile, head + tail in ascending k,
 *      alpha and beta, the store to C through its chunk table.
 *
 * The cut is shape arithmetic: with W = R * kit / U panels per
 * threadblock and J the larger number of tails an extra threadblock
 * gets, the tail is floor(W / J) panels and the head the rest of the
 * tile. A tail under a quarter wave (fewer tails than spare
 * threadblocks) keeps the plain grid. Fixed cut, ascending k in both
 * parts, fixed-order reduction, no atomics: the family's split-K law.
 *
 * @version 1.0.0
 * */

#pragma once

// The chunk-table-aware SIMT mainloop and epilogue iterators this builds on
#include "../gemm_chunked_bridge.cuh"

namespace cutlass
{
namespace gemm
{
namespace kernel
{

//! The bridge's kernel (cutlass::gemm::kernel::GemmChunked, single
//! k-slice) with one more early exit: the threadblocks of the tail tiles
//! (the block params.tail_m x (grid n - params.tail_n0) at the far side
//! of n) do nothing here; the tail kernel owns them.
template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_>
struct GemmWaveFull
{
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using OutputOp = typename Epilogue::OutputOp;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    struct Params
    {
        GemmCoord problem_size;
        GemmCoord grid_tiled_shape;
        int swizzle_log_tile;
        typename Mma::IteratorA::Params params_A;
        typename Mma::IteratorA::TensorRef ref_A;
        typename Mma::IteratorB::Params params_B;
        typename Mma::IteratorB::TensorRef ref_B;
        typename Epilogue::OutputTileIterator::Params params_C;
        typename Epilogue::OutputTileIterator::TensorRef ref_C;
        typename Epilogue::OutputTileIterator::Params params_D;
        typename Epilogue::OutputTileIterator::TensorRef ref_D;
        typename OutputOp::Params output_op;
        int gemm_k_size;
        //! The tail tiles (the tail kernel owns them): output tiles
        //! with m index below tail_m and n index from tail_n0 on. tail_m
        //! == 0: no tail.
        int tail_m;
        int tail_n0;
    };

    union SharedStorage
    {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage &shared_storage) const
    {
        // Compute threadblock location
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord threadblock_tile_offset =
            threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        // Early exit if CTA is out of range
        if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
            params.grid_tiled_shape.n() <= threadblock_tile_offset.n())
        {
            return;
        }

        // Early exit if the CTA is a tail tile: the tail kernel owns it
        if (threadblock_tile_offset.m() < params.tail_m &&
            threadblock_tile_offset.n() >= params.tail_n0)
        {
            return;
        }

        // Compute initial location in logical coordinates
        cutlass::MatrixCoord tb_offset_A{
            threadblock_tile_offset.m() * Mma::Shape::kM,
            threadblock_tile_offset.k() * params.gemm_k_size,
        };

        cutlass::MatrixCoord tb_offset_B{
            threadblock_tile_offset.k() * params.gemm_k_size,
            threadblock_tile_offset.n() * Mma::Shape::kN};

        // Problem size is a function of threadblock index in the K dimension
        int problem_size_k = min(params.problem_size.k(),
            (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

        // Compute threadblock-scoped matrix multiply-add
        int gemm_k_iterations =
            (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) /
            Mma::Shape::kK;

        // Compute position within threadblock
        int thread_idx = threadIdx.x;

        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A(params.params_A,
            params.ref_A.data(),
            {params.problem_size.m(), problem_size_k},
            thread_idx,
            tb_offset_A,
            nullptr);

        typename Mma::IteratorB iterator_B(params.params_B,
            params.ref_B.data(),
            {problem_size_k, params.problem_size.n()},
            thread_idx,
            tb_offset_B,
            nullptr);

        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
        int lane_idx = threadIdx.x % 32;

        // Construct thread-scoped matrix multiply
        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

        typename Mma::FragmentC accumulators;

        accumulators.clear();

        // Compute threadblock-scoped matrix multiply-add
        mma(gemm_k_iterations,
            accumulators,
            iterator_A,
            iterator_B,
            accumulators);

        OutputOp output_op(params.output_op);

        // Masked tile iterators constructed from members

        threadblock_tile_offset =
            threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        MatrixCoord threadblock_offset(
            threadblock_tile_offset.m() * Mma::Shape::kM,
            threadblock_tile_offset.n() * Mma::Shape::kN);

        // Tile iterator loading from source tensor.
        typename Epilogue::OutputTileIterator iterator_C(params.params_C,
            params.ref_C.data(),
            params.problem_size.mn(),
            thread_idx,
            threadblock_offset,
            nullptr);

        // Tile iterator writing to destination tensor.
        typename Epilogue::OutputTileIterator iterator_D(params.params_D,
            params.ref_D.data(),
            params.problem_size.mn(),
            thread_idx,
            threadblock_offset,
            nullptr);

        Epilogue epilogue(
            shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

        // Execute the epilogue operator to update the destination tensor.
        epilogue(output_op, iterator_D, accumulators, iterator_C);
    }
};

//! The last wave: R tail tiles on U = R + E threadblocks. The tail tiles
//! are a compact block of the output, tail_m tiles along m (the first
//! ones) x (grid n - tail_n0) along n (the last ones), so that the
//! threadblocks of the wave share operand panels through L2 as in a full
//! wave; tail tile t is at (t % tail_m, tail_n0 + t / tail_m). Block
//! t < R computes the head [0, split) of tail tile t; block R + e
//! computes the tails [split, kit) of the tail tiles e, e + E, .. (slot
//! j of extra e is tail tile e + j E). Every part lands as a raw partial in a
//! slot of the flat workspace (tile-packed kM x kN): the head of tail tile t
//! in slot t, slot j of extra e in slot R + e * slot_stride + j.
template <typename Mma_, typename Epilogue_> struct GemmWaveTail
{
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using OutputOp = typename Epilogue::OutputOp;
    using FragmentC = typename Mma::FragmentC;

    static int const kM = Mma::Shape::kM;
    static int const kN = Mma::Shape::kN;
    static int const kK = Mma::Shape::kK;
    static int const kThreadCount = 32 * Mma::WarpCount::kCount;

    using TileIterator = typename Epilogue::OutputTileIterator;

    struct Params
    {
        GemmCoord problem_size;
        typename Mma::IteratorA::Params params_A;
        typename Mma::IteratorA::TensorRef ref_A;
        typename Mma::IteratorB::Params params_B;
        typename Mma::IteratorB::TensorRef ref_B;
        typename TileIterator::Params params_S; //!< the flat partial slots
        float *partials;
        int tail_m;      //!< the tail block's tiles along m
        int tail_n0;     //!< the tail block's first tile along n
        int tiles;       //!< R: the tail tiles
        int extras;      //!< E: the threadblocks beyond the heads
        int slot_stride; //!< slots per extra threadblock
        int split;       //!< k panels of a head; the tail is [split, kit)
        int kit;         //!< k panels of a whole tile
    };

    union SharedStorage
    {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

  private:
    //! accum (cleared by the caller) += A * B over `iters` k panels from
    //! panel k0 of the problem's output tile `tile`
    CUTLASS_DEVICE void mainloop_(Params const &params,
        SharedStorage &shared,
        int thread_idx,
        int warp_idx,
        int lane_idx,
        int tile,
        int k0,
        int iters,
        FragmentC &accum) const
    {
        typename Mma::IteratorA iterator_A(params.params_A,
            params.ref_A.data(),
            {params.problem_size.m(), params.problem_size.k()},
            thread_idx,
            {(tile % params.tail_m) * kM, k0 * kK},
            nullptr);
        typename Mma::IteratorB iterator_B(params.params_B,
            params.ref_B.data(),
            {params.problem_size.k(), params.problem_size.n()},
            thread_idx,
            {k0 * kK, (params.tail_n0 + tile / params.tail_m) * kN},
            nullptr);
        Mma mma(shared.main_loop, thread_idx, warp_idx, lane_idx);
        mma(iters, accum, iterator_A, iterator_B, accum);
    }

    //! slot = accum (raw partial, identity output op, flat store)
    CUTLASS_DEVICE void store_s_(Params const &params,
        SharedStorage &shared,
        int thread_idx,
        int warp_idx,
        int lane_idx,
        long long slot,
        FragmentC const &accum) const
    {
        TileIterator iterator_S(params.params_S,
            params.partials + slot * (static_cast<long long>(kM) * kN),
            {kM, kN},
            thread_idx,
            MatrixCoord(0, 0));
        Epilogue epilogue(shared.epilogue, thread_idx, warp_idx, lane_idx);
        epilogue(
            OutputOp(typename OutputOp::Params(1.f, 0.f)), iterator_S, accum);
    }

  public:
    CUTLASS_DEVICE void operator()(
        Params const &params, SharedStorage &shared_storage) const
    {
        int const b = blockIdx.x;
        int const thread_idx = threadIdx.x;
        int const warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
        int const lane_idx = threadIdx.x % 32;

        if (b < params.tiles)
        {
            FragmentC accum;
            accum.clear();
            mainloop_(params,
                shared_storage,
                thread_idx,
                warp_idx,
                lane_idx,
                b,
                0,
                params.split,
                accum);
            store_s_(params,
                shared_storage,
                thread_idx,
                warp_idx,
                lane_idx,
                b,
                accum);
            return;
        }

        int const e = b - params.tiles;
        int j = 0;
        for (int t = e; t < params.tiles; t += params.extras, ++j)
        {
            FragmentC accum;
            accum.clear();
            if (j > 0)
            {
                // The epilogue reads the shared storage the mainloop writes
                __syncthreads();
            }
            mainloop_(params,
                shared_storage,
                thread_idx,
                warp_idx,
                lane_idx,
                t,
                params.split,
                params.kit - params.split,
                accum);
            store_s_(params,
                shared_storage,
                thread_idx,
                warp_idx,
                lane_idx,
                static_cast<long long>(params.tiles) + e * params.slot_stride +
                    j,
                accum);
        }
    }
};

//! The tail's fixed-order reduction: block t sums the head and the tail
//! of tail tile t in ascending k, applies alpha and beta, and stores C
//! through its chunk table. float4: kM and kN are 4-multiples and so is
//! every chunk class, so no vector straddles a chunk.
template <int kM, int kN, int THREADS>
__global__ void __launch_bounds__(THREADS)
    GemmWaveFixup(const float *__restrict__ partials,
        int tiles,
        int extras,
        int slot_stride,
        int tail_m,
        int tail_n0,
        int n,
        void *__restrict__ c,
        int c_shift,
        float alpha,
        float beta)
{
    int const t = blockIdx.x;
    int const tail_slot = tiles + (t % extras) * slot_stride + t / extras;
    long long const tile_elems = static_cast<long long>(kM) * kN;
    float4 const *const head =
        reinterpret_cast<float4 const *>(partials + tile_elems * t);
    float4 const *const tail =
        reinterpret_cast<float4 const *>(partials + tile_elems * tail_slot);
    constexpr int kVecs = kM * kN / 4 / THREADS;

    long long const row0 = static_cast<long long>(t % tail_m) * kM;
    int const col0 = (tail_n0 + t / tail_m) * kN;
    int const shift = c_shift >= 62 ? 62 : c_shift;
    long long const mask = (static_cast<long long>(1) << shift) - 1;
    float *const *const table = reinterpret_cast<float *const *>(c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVecs; ++i)
    {
        int const v = i * THREADS + threadIdx.x;
        float4 const h = head[v];
        float4 const w = tail[v];
        float out[4] = {alpha * (h.x + w.x),
            alpha * (h.y + w.y),
            alpha * (h.z + w.z),
            alpha * (h.w + w.w)};
        int const r = v / (kN / 4);
        int const c4 = (v % (kN / 4)) * 4;
        long long const elem = (row0 + r) * n + col0 + c4;
        float *const pc = c_shift == 0 ? static_cast<float *>(c) + elem
                                       : table[elem >> shift] + (elem & mask);
        if (beta != 0.f)
        {
            CUTLASS_PRAGMA_UNROLL
            for (int q = 0; q < 4; ++q)
            {
                out[q] += beta * pc[q];
            }
        }
        *reinterpret_cast<float4 *>(pc) =
            make_float4(out[0], out[1], out[2], out[3]);
    }
}

} // namespace kernel
} // namespace gemm
} // namespace cutlass
