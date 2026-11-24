/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm_chunked/ext/wgrad_bridge.cuh
 * The wgrad piece's own kernel stack: a TN-only (A k x m column-major,
 * B k x n row-major, both quad-contiguous along m / n) SIMT fp32 GEMM
 * whose threadblocks are BALANCED WORKERS instead of output tiles, and
 * the fixed-order reduction that finishes the tiles several workers
 * share. A private copy of what the piece needs; cuda.cu and
 * gemm_chunked_bridge.cuh stay untouched (only the table-aware epilogue
 * iterator is reused from the bridge).
 *
 * Schedule (one CTA per SM, `WORKERS` CTAs in all; T = tiles,
 * `kp` = k-panels per tile; tiles are numbered by RANK over a tile
 * order, see WgradBalanced::rank_to_tile):
 *   q = T / WORKERS, r = T - q * WORKERS (the leftover tiles);
 *   worker w computes q whole tiles (ranks w, w + WORKERS, ...; written
 *   straight to C), then the iteration range
 *   [w r kp / WORKERS, (w + 1) r kp / WORKERS) of the leftover tiles'
 *   flattened k space (tile-major), in ascending k. A share lies inside
 *   one tile or straddles two; the pieces go to flat workspace slots
 *   (2 per worker) and `wgrad_reduce` sums each shared tile's pieces in
 *   ascending-k (ascending worker) order.
 *   Every worker gets the same number of k-iterations (plus or minus
 *   one), so nothing is lost to the wave tail. No atomics anywhere:
 *   the split is a pure function of the shape and the reduction runs in
 *   a fixed ascending-k order.
 *
 * @version 1.0.0
 * */

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"

// Standard libraries
#include <cstddef>

// The chunk-table-aware epilogue tile iterator
#include "../gemm_chunked_bridge.cuh"

namespace nntile::kernel::gemm_chunked::ext
{
namespace wgradbt
{

using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
using WgradOutputOp =
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;

//! Resolve element `elem` of a C / D operand (the family's three
//! addressing modes: flat base, chunk table, one-shot sentinel).
CUTLASS_DEVICE float *chunk_at_mut(
    void *tab_or_flat, int shift, std::int64_t elem)
{
    if (shift == 0)
    {
        return reinterpret_cast<float *>(tab_or_flat) + elem;
    }
    float *const *tab = reinterpret_cast<float *const *>(tab_or_flat);
    if (shift >= 62)
    {
        return tab[0] + elem;
    }
    return tab[elem >> shift] + (elem & ((std::int64_t(1) << shift) - 1));
}

//! Global iterator over one operand stage (tile rows BK along k, BC
//! along the contiguous direction m or n), one or more 4-float runs per
//! thread: thread t owns the quads t, t + THREADS, ... of the stage's
//! quad list (row-major: `BC / 4` quads per k-row). Chunk-table
//! addressing at run granularity and unconditional (a runtime condition
//! on it in this loop cost 7 %): one table entry per quad load (the
//! 16B intra-chunk law forbids a quad straddling chunks). Interface:
//! the MmaPipelined contract.
template <int BK, int BC, int THREADS> class QuadContig
{
  public:
    using Element = float;
    using Index = int32_t;
    using LongIndex = std::int64_t;
    using Fragment = cutlass::Array<Element, 4 * (BK * BC / 4 / THREADS)>;
    static int const kQuads = BK * BC / 4 / THREADS;

    struct Params
    {
        int shift = 0;    //!< log2(chunk floats); >= 62: one-shot sentinel
        LongIndex ld = 0; //!< m (A) or n (B): the contiguous extent
    };

    //! The quads of the stage that starts at k-row `k0` of the tile
    //! column `c0`.
    CUTLASS_DEVICE QuadContig(
        Params const &p, Element const *ptr, int k0, int c0, int thread_id) :
        ptr_(ptr),
        mask_((std::int64_t(1) << (p.shift >= 62 ? 62 : p.shift)) - 1),
        se_(p.shift >= 62 ? 62 : p.shift)
    {
        int const cq = BC / 4;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQuads; ++j)
        {
            int const q = thread_id + j * THREADS;
            e_[j] = LongIndex(k0 + q / cq) * p.ld + c0 + (q % cq) * 4;
        }
        d_ = LongIndex(BK) * p.ld;
    }

    CUTLASS_DEVICE void load(Fragment &frag) const
    {
        if (!valid_)
        {
            return; //!< phantom last stage: the caller pre-clears
        }
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQuads; ++j)
        {
            Element const *base =
                reinterpret_cast<Element const *const *>(ptr_)[e_[j] >> se_];
            *reinterpret_cast<float4 *>(&frag[4 * j]) =
                *reinterpret_cast<const float4 *>(base + (e_[j] & mask_));
        }
    }

    CUTLASS_DEVICE QuadContig &operator++()
    {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQuads; ++j)
        {
            e_[j] += d_;
        }
        return *this;
    }

    CUTLASS_DEVICE void clear_mask(bool b) { valid_ = !b && valid_; }

  private:
    Element const *ptr_;
    LongIndex e_[kQuads > 0 ? kQuads : 1];
    LongIndex d_, mask_;
    int se_;
    bool valid_ = true;
};

//! Shared-memory store matching QuadContig: the stage tile is stored
//! with its contiguous direction at stride `BC` (A column-major m x k,
//! B row-major k x n -- one formula), one 16B store per quad.
//! `IS_A` selects which MatrixCoord component advances a stage.
template <int BK, int BC, int THREADS, typename Layout, bool IS_A>
class QuadStore
{
  public:
    using Element = float;
    using Index = int32_t;
    using LongIndex = std::int64_t;
    using TensorRef = cutlass::TensorRef<Element, Layout>;
    using TensorCoord = cutlass::MatrixCoord;
    using Fragment = cutlass::Array<Element, 4 * (BK * BC / 4 / THREADS)>;
    static int const kQuads = BK * BC / 4 / THREADS;

    CUTLASS_DEVICE QuadStore(TensorRef const &ref, int thread_idx) :
        base_(ref.data())
    {
        ld_ = int(ref.stride(0));
        stage_ = LongIndex(BK) * ld_;
        t_ = thread_idx;
    }

    CUTLASS_DEVICE void store(Fragment const &frag)
    {
        int const cq = BC / 4;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQuads; ++j)
        {
            int const q = t_ + j * THREADS;
            *reinterpret_cast<float4 *>(
                base_ + LongIndex(q / cq) * ld_ + (q % cq) * 4) =
                *reinterpret_cast<const float4 *>(&frag[4 * j]);
        }
    }

    CUTLASS_DEVICE QuadStore &operator++()
    {
        base_ += stage_;
        return *this;
    }

    CUTLASS_DEVICE void add_tile_offset(TensorCoord const &c)
    {
        base_ += (IS_A ? c.column() : c.row()) * stage_;
    }

  private:
    Element *base_;
    LongIndex stage_;
    int ld_, t_;
};

//! The balanced-worker GEMM (see the file comment for the schedule).
//! Threadblock tile BM x BN x BK on CUTLASS's SIMT MmaPipelined
//! mainloop and SIMT epilogue; both operands resolved through their
//! chunk tables; a segment that covers a whole tile writes
//! alpha * acc + beta * C straight into C, any other segment writes
//! its raw partial into its workspace slot.
template <typename Mma_, typename Epilogue_, int BM_, int BN_, int BK_>
struct WgradBalanced
{
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using OutputOp = typename Epilogue::OutputOp;
    using IteratorA = typename Mma::IteratorA;
    using IteratorB = typename Mma::IteratorB;
    using OutputTileIterator = typename Epilogue::OutputTileIterator;
    static int const kThreadCount = 32 * Mma::WarpCount::kCount;
    static int const kBM = BM_, kBN = BN_, kBK = BK_;

    struct Params
    {
        int m, n, k;
        int ba, bb, bbn; //!< tile block ba x bb; bbn blocks per block row
        int kp;          //!< k-panels per tile
        int workers, qfull, r;
        std::int64_t leftover; //!< r * kp
        typename IteratorA::Params params_A;
        float const *ptr_A;
        typename IteratorB::Params params_B;
        float const *ptr_B;
        typename OutputTileIterator::Params params_C;
        float *ptr_C;
        typename OutputTileIterator::Params params_W;
        float *ws;
        float alpha, beta;
    };

    union SharedStorage
    {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

    //! Tile rank -> tile coordinates. The ranks run over blocks of
    //! ba x bb tiles (block-row major, n fastest inside a block); with
    //! ba = tm, bb = 1 this is the plain m-fastest strip order. With
    //! blocks, the workers' concurrently active ranks (w, w + WORKERS,
    //! ...) cover a compact part of the output, which keeps the operand
    //! panels shared in L2 on tall grids (tm tile-rows for few
    //! tile-columns).
    CUTLASS_DEVICE void rank_to_tile(
        Params const &p, int rank, int &ti_m, int &ti_n) const
    {
        int const bp = p.ba * p.bb;
        int const blk = rank / bp, inner = rank - blk * bp;
        ti_n = (blk % p.bbn) * p.bb + inner % p.bb;
        ti_m = (blk / p.bbn) * p.ba + inner / p.bb;
    }

    CUTLASS_DEVICE OutputTileIterator c_tile(
        Params const &p, int rank, int thread_idx) const
    {
        int ti_m, ti_n;
        rank_to_tile(p, rank, ti_m, ti_n);
        return OutputTileIterator(p.params_C,
            p.ptr_C,
            {p.m, p.n},
            thread_idx,
            {ti_m * kBM, ti_n * kBN});
    }

    CUTLASS_DEVICE OutputTileIterator ws_tile(
        Params const &p, int slot, int thread_idx) const
    {
        return OutputTileIterator(p.params_W,
            p.ws + (std::int64_t(slot) * kBM) * kBN,
            {kBM, kBN},
            thread_idx,
            {0, 0});
    }

    //! One k-range [k_lo, k_hi) (panels) of tile (ti_m, ti_n) of the
    //! GEMM, finished by the epilogue into `tile_iter`.
    CUTLASS_DEVICE void segment(Params const &p,
        SharedStorage &shared,
        int ti_m,
        int ti_n,
        int k_lo,
        int k_hi,
        OutputTileIterator tile_iter,
        OutputOp const &op) const
    {
        int const thread_idx = threadIdx.x;
        int const warp_idx = __shfl_sync(0xffffffff, thread_idx / 32, 0);
        int const lane_idx = thread_idx % 32;
        IteratorA iterator_A(
            p.params_A, p.ptr_A, k_lo * kBK, ti_m * kBM, thread_idx);
        IteratorB iterator_B(
            p.params_B, p.ptr_B, k_lo * kBK, ti_n * kBN, thread_idx);
        Mma mma(shared.main_loop, thread_idx, warp_idx, lane_idx);
        typename Mma::FragmentC accumulators;
        accumulators.clear();
        mma(k_hi - k_lo, accumulators, iterator_A, iterator_B, accumulators);
        Epilogue epilogue(shared.epilogue, thread_idx, warp_idx, lane_idx);
        epilogue(op, tile_iter, accumulators, tile_iter);
        __syncthreads();
    }

    CUTLASS_DEVICE void operator()(
        Params const &p, SharedStorage &shared_storage) const
    {
        int const thread_idx = threadIdx.x;
        int const w = blockIdx.x;
        OutputOp const direct_op(typename OutputOp::Params(p.alpha, p.beta));
        OutputOp const raw_op(typename OutputOp::Params(1.f, 0.f));
        // The q whole tiles of this worker.
        for (int i = 0; i < p.qfull; ++i)
        {
            int const t = w + i * p.workers;
            int ti_m, ti_n;
            rank_to_tile(p, t, ti_m, ti_n);
            segment(p,
                shared_storage,
                ti_m,
                ti_n,
                0,
                p.kp,
                c_tile(p, t, thread_idx),
                direct_op);
        }
        // This worker's share of the leftover tiles' k space.
        std::int64_t const s = std::int64_t(w) * p.leftover / p.workers;
        std::int64_t const e = std::int64_t(w + 1) * p.leftover / p.workers;
        if (s >= e)
        {
            return;
        }
        int const j0 = int(s / p.kp), j1 = int((e - 1) / p.kp);
        std::int64_t const tile0 = p.qfull * std::int64_t(p.workers);
        if (j0 == j1 && s == std::int64_t(j0) * p.kp &&
            e == std::int64_t(j0 + 1) * p.kp)
        {
            // The whole tile in one piece: straight to C.
            int const t = int(tile0) + j0;
            int ti_m, ti_n;
            rank_to_tile(p, t, ti_m, ti_n);
            segment(p,
                shared_storage,
                ti_m,
                ti_n,
                0,
                p.kp,
                c_tile(p, t, thread_idx),
                direct_op);
            return;
        }
        // Pieces of at most two tiles: raw partials to the slots 2w,
        // 2w+1 (ascending k); the k-ranges are tile-relative panels.
        int const npieces = j0 == j1 ? 1 : 2;
        for (int j = 0; j < npieces; ++j)
        {
            int const jj = j0 + j;
            int const t = int(tile0) + jj;
            std::int64_t const lo = j == 0 ? s : std::int64_t(jj) * p.kp;
            std::int64_t const hi =
                j == npieces - 1 ? e : std::int64_t(jj + 1) * p.kp;
            int ti_m, ti_n;
            rank_to_tile(p, t, ti_m, ti_n);
            segment(p,
                shared_storage,
                ti_m,
                ti_n,
                int(lo - std::int64_t(jj) * p.kp),
                int(hi - std::int64_t(jj) * p.kp),
                ws_tile(p, 2 * w + j, thread_idx),
                raw_op);
        }
    }
};

//! Finish the tiles several workers share: C = alpha * sum(pieces,
//! ascending k) + beta * C. One block per BM-row group of one leftover
//! tile; the piece list is recomputed from the same shape formulas the
//! GEMM used (worker w's share [w R / WORKERS, (w + 1) R / WORKERS)).
template <int BM, int BN, int WORKERS>
__global__ void wgrad_reduce(const float *__restrict__ ws,
    void *c,
    int c_shift,
    int n,
    int ba,
    int bb,
    int bbn,
    int kp,
    long long R,
    int tile0,
    float alpha,
    float beta)
{
    int const tpr = BN / 4;    //!< threads per tile row
    int const rpb = 256 / tpr; //!< tile rows per block
    int const bpg = BM / rpb;  //!< blocks per tile
    int const j = blockIdx.x / bpg, g = blockIdx.x % bpg;
    int const row = g * rpb + threadIdx.x / tpr;
    int const col = (threadIdx.x % tpr) * 4;
    int const tile = tile0 + j;
    int const bp = ba * bb, blk = tile / bp, inner = tile - blk * bp;
    int const ti_m = ((blk / bbn) * ba + inner / bb) * BM;
    int const ti_n = ((blk % bbn) * bb + inner % bb) * BN;
    std::int64_t const tile_lo = std::int64_t(j) * kp;
    std::int64_t const tile_hi = tile_lo + kp;
    float4 acc = {0.f, 0.f, 0.f, 0.f};
    // Workers overlapping this tile: w R / WORKERS < tile_hi and
    // (w + 1) R / WORKERS > tile_lo; the bounds carry one spare
    // worker on each side for the floor.
    int const w_lo = max(0, int(tile_lo * WORKERS / R) - 1);
    int const w_hi = min(WORKERS, int((tile_hi * WORKERS - 1) / R) + 2);
    for (int w = w_lo; w < w_hi; ++w)
    {
        std::int64_t const s = std::int64_t(w) * R / WORKERS;
        std::int64_t const e = std::int64_t(w + 1) * R / WORKERS;
        if (s < tile_hi && e > tile_lo)
        {
            int const slot = 2 * w + (s < tile_lo ? 1 : 0);
            float4 const v = *reinterpret_cast<const float4 *>(
                ws + (std::int64_t(slot) * BM + row) * BN + col);
            acc.x += v.x;
            acc.y += v.y;
            acc.z += v.z;
            acc.w += v.w;
        }
    }
    float out[4] = {
        alpha * acc.x, alpha * acc.y, alpha * acc.z, alpha * acc.w};
    std::int64_t const elem = std::int64_t(ti_m + row) * n + ti_n + col;
    float *dst = chunk_at_mut(c, c_shift, elem);
    if (beta != 0.f)
    {
        float4 const cv = *reinterpret_cast<const float4 *>(dst);
        out[0] += beta * cv.x;
        out[1] += beta * cv.y;
        out[2] += beta * cv.z;
        out[3] += beta * cv.w;
    }
    *reinterpret_cast<float4 *>(dst) =
        make_float4(out[0], out[1], out[2], out[3]);
}

//! Assemble and launch one configuration: threadblock tile BM x BN x BK,
//! warp tile WM x WN x BK, `WORKERS` balanced CTAs. TN only (A k x m
//! column-major, B k x n row-major), both operands through chunk
//! tables, C in any addressing mode. `BLOCKED` numbers the tiles over
//! blocks instead of the m-fastest strip. Declines (false, nothing
//! launched) what the tile cannot tile.
template <int BM, int BN, int BK, int WM, int WN, int WORKERS, bool BLOCKED>
bool run(const Call &call, cudaStream_t stream) noexcept
{
    if (call.batch != 1 || call.ta != Op::Trans || call.tb != Op::NoTrans ||
        call.m < BM || call.n < BN || call.k < 2 * BK || call.m % BM != 0 ||
        call.n % BN != 0 || call.k % BK != 0 || call.a_shift == 0 ||
        call.b_shift == 0)
    {
        return false;
    }
    static int const kThreads = 32 * (BM / WM) * (BN / WN);
    using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<BM, BN, BK>,
        cutlass::gemm::GemmShape<WM, WN, BK>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        float,
        CM,
        float,
        RM,
        float,
        RM,
        cutlass::arch::OpClassSimt,
        2,
        cutlass::arch::OpClassSimt>;
    using IteratorA = QuadContig<BK, BM, kThreads>;
    using IteratorB = QuadContig<BK, BN, kThreads>;
    using Mma =
        cutlass::gemm::threadblock::MmaPipelined<typename MmaCore::Shape,
            IteratorA,
            QuadStore<BK, BM, kThreads, CM, true>,
            IteratorB,
            QuadStore<BK, BN, kThreads, RM, false>,
            float,
            RM,
            typename MmaCore::MmaPolicy>;
    using EpiParts = cutlass::epilogue::threadblock::DefaultEpilogueSimt<
        cutlass::gemm::GemmShape<BM, BN, BK>,
        typename Mma::Operator,
        WgradOutputOp,
        1>;
    using OutputTileIterator = cutlass::epilogue::threadblock::
        ChunkedEpilogueIterator<typename EpiParts::OutputTileThreadMap, float>;
    using Epilogue = cutlass::epilogue::threadblock::Epilogue<
        cutlass::gemm::GemmShape<BM, BN, BK>,
        typename Mma::Operator,
        EpiParts::kPartitionsK,
        OutputTileIterator,
        typename EpiParts::AccumulatorFragmentIterator,
        typename EpiParts::WarpTileIterator,
        typename EpiParts::SharedLoadIterator,
        WgradOutputOp,
        typename EpiParts::Padding>;
    using Kernel = WgradBalanced<Mma, Epilogue, BM, BN, BK>;

    int const m = call.m, n = call.n, k = call.k;
    int const tm = m / BM, tn = n / BN, tiles = tm * tn, kp = k / BK;
    int const qfull = tiles / WORKERS, r = tiles - qfull * WORKERS;
    // The tile block: a strip along m (the bridge's order), or -- so
    // that one layer of workers sits in one block -- ba = the largest
    // divisor of tm up to 16, bb = the smallest divisor of tn with
    // ba * bb >= WORKERS (tn when there is none).
    int ba = tm, bb = 1;
    if constexpr (BLOCKED)
    {
        ba = tm;
        while (ba > 16 || tm % ba != 0)
        {
            --ba;
        }
        bb = tn;
        for (int d = 1; d <= tn; ++d)
        {
            if (tn % d == 0 && ba * d >= WORKERS)
            {
                bb = d;
                break;
            }
        }
    }
    float *ws = nullptr;
    if (r > 0)
    {
        ws = static_cast<float *>(scratch(
            std::size_t(2 * WORKERS) * std::size_t(BM * BN) * sizeof(float)));
        if (ws == nullptr)
        {
            return false;
        }
    }
    typename Kernel::Params p{};
    p.m = m;
    p.n = n;
    p.k = k;
    p.ba = ba;
    p.bb = bb;
    p.bbn = (tn + bb - 1) / bb;
    p.kp = kp;
    p.workers = WORKERS;
    p.qfull = qfull;
    p.r = r;
    p.leftover = std::int64_t(r) * kp;
    p.params_A.shift = call.a_shift;
    p.params_A.ld = m;
    p.ptr_A = static_cast<float const *>(call.a);
    p.params_B.shift = call.b_shift;
    p.params_B.ld = n;
    p.ptr_B = static_cast<float const *>(call.b);
    p.params_C =
        typename OutputTileIterator::Params(RM::packed({m, n}), call.c_shift);
    p.ptr_C = static_cast<float *>(call.c);
    p.params_W = typename OutputTileIterator::Params(RM::packed({BM, BN}), 0);
    p.ws = ws;
    p.alpha = call.alpha;
    p.beta = call.beta;
    int const smem = int(sizeof(typename Kernel::SharedStorage));
    if (smem >= (48 << 10) && cudaFuncSetAttribute(cutlass::Kernel<Kernel>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  smem) != cudaSuccess)
    {
        return false;
    }
    cutlass::Kernel<Kernel>
        <<<dim3(WORKERS), dim3(kThreads), smem, stream>>>(p);
    if (cudaGetLastError() != cudaSuccess)
    {
        return false;
    }
    if (r > 0)
    {
        static int const kTpr = BN / 4;
        wgrad_reduce<BM, BN, WORKERS>
            <<<dim3(r * (BM / (256 / kTpr))), dim3(256), 0, stream>>>(ws,
                call.c,
                call.c_shift,
                n,
                ba,
                bb,
                p.bbn,
                kp,
                p.leftover,
                qfull * WORKERS,
                call.alpha,
                call.beta);
        return cudaGetLastError() == cudaSuccess;
    }
    return true;
}

} // namespace wgradbt
} // namespace nntile::kernel::gemm_chunked::ext
