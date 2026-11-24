/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm_chunked/ext/mixed_bridge.cuh
 * The kernel parts of the `mixed` extension piece: the CUTLASS 2.11 SIMT
 * fp32 mainloop and epilogue assembled for MIXED layouts (some operands
 * single-chunk flat, the rest chunk tables), on several threadblock
 * tiles, with an optional deterministic wave-balanced K split.
 *
 * Nothing here edits the family's core. gemm_chunked_bridge.cuh is
 * included for its epilogue iterator; the loaders below generalize the
 * bridge's quad-contiguous loaders to any threadblock tile whose k panel
 * is 8 (the bridge's own are written for 128x128x8 and 256 threads), and
 * the kernels carry the chunk shifts the way kernel::GemmChunked does.
 *
 * @version 1.0.0
 * */

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

// The chunk-table-aware epilogue iterator (and the CUTLASS headers the
// bridge header relies on; ext/mixed.cu includes the two device headers
// it needs first).
#include "../gemm_chunked_bridge.cuh"

namespace nntile::kernel::gemm_chunked::ext::mixed_detail
{

using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
using EpiOp =
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
using cutlass::MatrixCoord;

//! cutlass::platform::conditional<B, T, F>::type
template <bool B, typename T, typename F>
using cond_t = typename cutlass::platform::conditional<B, T, F>::type;

// =====================================================================
// Quad-contiguous global loaders (the bridge's BridgeQuadA / BridgeQuadB
// for any threadblock tile whose k panel is 8). One panel of the A tile
// is BM x 8 (B: 8 x BN): 2 * ROWS quads of 4 floats for T threads, each
// thread owning quads t, t + T, ...; a quad is 4 floats along the
// operand's storage-contiguous direction:
//   along k (A row-major, B column-major): quad q -> row q >> 1, k 4 * (q & 1)
//   along m/n (A column-major, B row-major): quad q -> k-row q / (ROWS / 4),
//                                              m/n 4 * (q % (ROWS / 4))
// One table resolve and one 128-bit load per quad: table traffic is
// O(rows per panel). AC selects flat (0) and table (1) addressing at
// compile time; the 16B intra-chunk law (classes >= 2^18 floats,
// class-aligned bases, 4-aligned quad starts) forbids a quad straddling
// a chunk boundary.
// =====================================================================

//! A-side: A tile BM x 8. GL = RowMajor (A is m x k, k-contiguous) or
//! ColumnMajor (A is k x m, m-contiguous).
template <typename GL, int AC, int BM, int T> class QuadLoadA
{
  public:
    using Element = float;
    using Layout = GL;
    using Index = int32_t;
    using LongIndex = int64_t;
    using TensorRef = cutlass::TensorRef<Element const, Layout>;
    using TensorCoord = cutlass::MatrixCoord;

    static bool const kIsRowMajor = cutlass::platform::is_same<GL, RM>::value;
    static int const kQ = BM * 2 / T; //!< quads per thread per panel
    static_assert(kQ >= 1 && kQ * T == BM * 2, "BM x 8 tile over T threads");
    static_assert(BM % 4 == 0, "");

    using Fragment = cutlass::Array<Element, 4 * kQ>;

    struct Params
    {
        int shift = 0;
        //! The operand's TRUE leading dimension (elements; the extent is
        //! slice-capped under a K split, the layout stride is not).
        LongIndex ld = 0;
        CUTLASS_HOST_DEVICE Params() = default;
        CUTLASS_HOST_DEVICE Params(Layout const &layout, int chunk_shift) :
            shift(chunk_shift), ld(layout.stride(0))
        {
        }
    };

    CUTLASS_DEVICE
    QuadLoadA(Params const &params,
        Element const *ptr,
        MatrixCoord extent,
        int thread_id,
        MatrixCoord tile_offset,
        int const * = nullptr) :
        ptr_(ptr)
    {
        (void) extent;
        LongIndex const ld = params.ld;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQ; ++j)
        {
            int const q = thread_id + j * T;
            if constexpr (kIsRowMajor)
            {
                e_[j] = LongIndex(tile_offset.row() + (q >> 1)) * ld +
                        tile_offset.column() + (q & 1) * 4;
            }
            else
            {
                e_[j] = LongIndex(tile_offset.column() + q / (BM / 4)) * ld +
                        tile_offset.row() + (q % (BM / 4)) * 4;
            }
        }
        d_ = kIsRowMajor ? LongIndex(8) : LongIndex(8) * ld;
        sh_ = params.shift >= 62 ? 62 : params.shift;
        valid_ = true;
    }

    CUTLASS_DEVICE
    void load(Fragment &frag) const
    {
        if (!valid_)
        {
            return; //!< masked (phantom last panel): caller pre-clears
        }
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQ; ++j)
        {
            Element const *q;
            if constexpr (AC == 0)
            {
                q = ptr_ + e_[j];
            }
            else
            {
                int64_t const mask = (int64_t(1) << sh_) - 1;
                q = reinterpret_cast<Element const *const *>(
                        ptr_)[e_[j] >> sh_] +
                    (e_[j] & mask);
            }
            *reinterpret_cast<float4 *>(&frag[4 * j]) =
                *reinterpret_cast<const float4 *>(q);
        }
    }

    CUTLASS_DEVICE
    QuadLoadA &operator++()
    {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQ; ++j)
        {
            e_[j] += d_;
        }
        return *this;
    }

    CUTLASS_DEVICE void clear_mask(bool b) { valid_ = !b && valid_; }

  private:
    Element const *ptr_;
    LongIndex e_[kQ];
    LongIndex d_;
    int sh_;
    bool valid_;
};

//! B-side (mirror of QuadLoadA; B tile 8 x BN). GL = RowMajor (B is
//! k x n, n-contiguous) or ColumnMajor (B is n x k, k-contiguous).
template <typename GL, int AC, int BN, int T> class QuadLoadB
{
  public:
    using Element = float;
    using Layout = GL;
    using Index = int32_t;
    using LongIndex = int64_t;
    using TensorRef = cutlass::TensorRef<Element const, Layout>;
    using TensorCoord = cutlass::MatrixCoord;

    static bool const kIsRowMajor = cutlass::platform::is_same<GL, RM>::value;
    static int const kQ = BN * 2 / T; //!< quads per thread per panel
    static_assert(kQ >= 1 && kQ * T == BN * 2, "8 x BN tile over T threads");
    static_assert(BN % 4 == 0, "");

    using Fragment = cutlass::Array<Element, 4 * kQ>;

    struct Params
    {
        int shift = 0;
        LongIndex ld = 0;
        CUTLASS_HOST_DEVICE Params() = default;
        CUTLASS_HOST_DEVICE Params(Layout const &layout, int chunk_shift) :
            shift(chunk_shift), ld(layout.stride(0))
        {
        }
    };

    CUTLASS_DEVICE
    QuadLoadB(Params const &params,
        Element const *ptr,
        MatrixCoord extent,
        int thread_id,
        MatrixCoord tile_offset,
        int const * = nullptr) :
        ptr_(ptr)
    {
        (void) extent;
        LongIndex const ld = params.ld;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQ; ++j)
        {
            int const q = thread_id + j * T;
            if constexpr (kIsRowMajor)
            {
                e_[j] = LongIndex(tile_offset.row() + q / (BN / 4)) * ld +
                        tile_offset.column() + (q % (BN / 4)) * 4;
            }
            else
            {
                e_[j] = LongIndex(tile_offset.column() + (q >> 1)) * ld +
                        tile_offset.row() + (q & 1) * 4;
            }
        }
        d_ = kIsRowMajor ? LongIndex(8) * ld : LongIndex(8);
        sh_ = params.shift >= 62 ? 62 : params.shift;
        valid_ = true;
    }

    CUTLASS_DEVICE
    void load(Fragment &frag) const
    {
        if (!valid_)
        {
            return; //!< masked (phantom last panel): caller pre-clears
        }
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQ; ++j)
        {
            Element const *q;
            if constexpr (AC == 0)
            {
                q = ptr_ + e_[j];
            }
            else
            {
                int64_t const mask = (int64_t(1) << sh_) - 1;
                q = reinterpret_cast<Element const *const *>(
                        ptr_)[e_[j] >> sh_] +
                    (e_[j] & mask);
            }
            *reinterpret_cast<float4 *>(&frag[4 * j]) =
                *reinterpret_cast<const float4 *>(q);
        }
    }

    CUTLASS_DEVICE
    QuadLoadB &operator++()
    {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQ; ++j)
        {
            e_[j] += d_;
        }
        return *this;
    }

    CUTLASS_DEVICE void clear_mask(bool b) { valid_ = !b && valid_; }

  private:
    Element const *ptr_;
    LongIndex e_[kQ];
    LongIndex d_;
    int sh_;
    bool valid_;
};

// =====================================================================
// The matching shared-memory stores. The policy keeps A m-contiguous
// (ColumnMajor stage, ld = stride per k) and B n-contiguous (RowMajor
// stage), so a quad along k is stored as 4 scalars at stride ld (the
// transposed store, CUTLASS's own pattern for these sides) and a quad
// along m/n as one float4.
// =====================================================================

//! A-side store: quad index q as in QuadLoadA.
template <bool kQuadAlongK, int BM, int T> class QuadStoreA
{
  public:
    using Element = float;
    using Layout = CM;
    using Index = int32_t;
    using LongIndex = int64_t;
    using TensorRef = cutlass::TensorRef<Element, Layout>;
    using TensorCoord = cutlass::MatrixCoord;

    static int const kQ = BM * 2 / T;
    using Fragment = cutlass::Array<Element, 4 * kQ>;

    CUTLASS_DEVICE
    QuadStoreA(TensorRef const &ref, int thread_idx) : base_(ref.data())
    {
        ld_ = int(ref.stride(0));
        stage_ = LongIndex(8) * ld_;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQ; ++j)
        {
            int const q = thread_idx + j * T;
            if constexpr (kQuadAlongK)
            {
                off_[j] = LongIndex(q >> 1) + LongIndex((q & 1) * 4) * ld_;
            }
            else
            {
                off_[j] = LongIndex(q % (BM / 4)) * 4 +
                          LongIndex(q / (BM / 4)) * ld_;
            }
        }
    }

    CUTLASS_DEVICE
    void store(Fragment const &frag)
    {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQ; ++j)
        {
            if constexpr (kQuadAlongK)
            {
                float4 const f =
                    *reinterpret_cast<const float4 *>(&frag[4 * j]);
                Element *p = base_ + off_[j];
                p[0] = f.x;
                p[ld_] = f.y;
                p[LongIndex(2) * ld_] = f.z;
                p[LongIndex(3) * ld_] = f.w;
            }
            else
            {
                *reinterpret_cast<float4 *>(base_ + off_[j]) =
                    *reinterpret_cast<const float4 *>(&frag[4 * j]);
            }
        }
    }

    CUTLASS_DEVICE
    QuadStoreA &operator++()
    {
        base_ += stage_;
        return *this;
    }

    CUTLASS_DEVICE
    void add_tile_offset(MatrixCoord const &c)
    {
        base_ += LongIndex(c.column()) * stage_;
    }

  private:
    Element *base_;
    LongIndex off_[kQ];
    LongIndex stage_;
    int ld_;
};

//! B-side store: quad index q as in QuadLoadB.
template <bool kQuadAlongN, int BN, int T> class QuadStoreB
{
  public:
    using Element = float;
    using Layout = RM;
    using Index = int32_t;
    using LongIndex = int64_t;
    using TensorRef = cutlass::TensorRef<Element, Layout>;
    using TensorCoord = cutlass::MatrixCoord;

    static int const kQ = BN * 2 / T;
    using Fragment = cutlass::Array<Element, 4 * kQ>;

    CUTLASS_DEVICE
    QuadStoreB(TensorRef const &ref, int thread_idx) : base_(ref.data())
    {
        ld_ = int(ref.stride(0));
        stage_ = LongIndex(8) * ld_;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQ; ++j)
        {
            int const q = thread_idx + j * T;
            if constexpr (kQuadAlongN)
            {
                off_[j] = LongIndex(q % (BN / 4)) * 4 +
                          LongIndex(q / (BN / 4)) * ld_;
            }
            else
            {
                off_[j] = LongIndex(q >> 1) + LongIndex((q & 1) * 4) * ld_;
            }
        }
    }

    CUTLASS_DEVICE
    void store(Fragment const &frag)
    {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQ; ++j)
        {
            if constexpr (kQuadAlongN)
            {
                *reinterpret_cast<float4 *>(base_ + off_[j]) =
                    *reinterpret_cast<const float4 *>(&frag[4 * j]);
            }
            else
            {
                float4 const f =
                    *reinterpret_cast<const float4 *>(&frag[4 * j]);
                Element *p = base_ + off_[j];
                p[0] = f.x;
                p[ld_] = f.y;
                p[LongIndex(2) * ld_] = f.z;
                p[LongIndex(3) * ld_] = f.w;
            }
        }
    }

    CUTLASS_DEVICE
    QuadStoreB &operator++()
    {
        base_ += stage_;
        return *this;
    }

    CUTLASS_DEVICE
    void add_tile_offset(MatrixCoord const &c)
    {
        base_ += LongIndex(c.row()) * stage_;
    }

  private:
    Element *base_;
    LongIndex off_[kQ];
    LongIndex stage_;
    int ld_;
};

//! Whether an iterator's Params carry the chunk shift (the table-aware
//! classes); the native CUTLASS iterators take the layout alone.
template <typename It> struct TakesShift : cutlass::platform::false_type
{
};

template <typename GL, int AC, int R, int T>
struct TakesShift<QuadLoadA<GL, AC, R, T>> : cutlass::platform::true_type
{
};

template <typename GL, int AC, int R, int T>
struct TakesShift<QuadLoadB<GL, AC, R, T>> : cutlass::platform::true_type
{
};

template <typename ThreadMap,
    typename Element,
    bool ScatterD,
    typename PermuteDLayout,
    bool UseCUDAStore>
struct TakesShift<
    cutlass::epilogue::threadblock::ChunkedEpilogueIterator<ThreadMap,
        Element,
        ScatterD,
        PermuteDLayout,
        UseCUDAStore>> : cutlass::platform::true_type
{
};

template <typename It, typename L>
CUTLASS_HOST_DEVICE typename It::Params iter_params(
    L const &layout, int shift) noexcept
{
    if constexpr (TakesShift<It>::value)
    {
        return typename It::Params(layout, shift);
    }
    else
    {
        (void) shift;
        return typename It::Params(layout);
    }
}

// =====================================================================
// The assembly: CUTLASS 2.11's SIMT fp32 MmaPipelined on a BM x BN x 8
// threadblock tile (warp WM x WN x 8, 2 stages) with, per operand,
// either the quad loaders above or -- for a flat operand -- CUTLASS's
// own PredicatedTileIterator and smem store; and the SIMT epilogue on
// either the chunk-table-aware output iterator (C flat or table at
// runtime) or CUTLASS's own predicated one (flat C).
// =====================================================================

template <typename LA,
    typename LB,
    int BM,
    int BN,
    int WM,
    int WN,
    int ACA,
    int ACB,
    bool NativeFlat,
    bool NativeC>
struct MixedGemm
{
    using LayoutA = LA;
    using LayoutB = LB;
    using ThreadblockShape = cutlass::gemm::GemmShape<BM, BN, 8>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, 8>;
    using MmaCore =
        cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
            WarpShape,
            cutlass::gemm::GemmShape<1, 1, 1>,
            float,
            LA,
            float,
            LB,
            float,
            RM,
            cutlass::arch::OpClassSimt,
            2,
            cutlass::arch::OpClassSimt>;
    static int const kThreads = MmaCore::kThreads;
    static bool const kAQuadAlongK = cutlass::platform::is_same<LA, RM>::value;
    static bool const kBQuadAlongN = cutlass::platform::is_same<LB, RM>::value;
    using Cfg = cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        float,
        float,
        float,
        float>;

    using NativeA = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<BM, 8>,
        float,
        LA,
        1,
        typename MmaCore::IteratorThreadMapA,
        Cfg::kAlignmentA>;
    using NativeB = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<8, BN>,
        float,
        LB,
        0,
        typename MmaCore::IteratorThreadMapB,
        Cfg::kAlignmentB>;

    static bool const kUseNativeA = NativeFlat && ACA == 0;
    static bool const kUseNativeB = NativeFlat && ACB == 0;

    using IteratorA =
        cond_t<kUseNativeA, NativeA, QuadLoadA<LA, ACA, BM, kThreads>>;
    using SmemIteratorA = cond_t<kUseNativeA,
        typename MmaCore::SmemIteratorA,
        QuadStoreA<kAQuadAlongK, BM, kThreads>>;
    using IteratorB =
        cond_t<kUseNativeB, NativeB, QuadLoadB<LB, ACB, BN, kThreads>>;
    using SmemIteratorB = cond_t<kUseNativeB,
        typename MmaCore::SmemIteratorB,
        QuadStoreB<kBQuadAlongN, BN, kThreads>>;

    using Mma =
        cutlass::gemm::threadblock::MmaPipelined<typename MmaCore::Shape,
            IteratorA,
            SmemIteratorA,
            IteratorB,
            SmemIteratorB,
            float,
            RM,
            typename MmaCore::MmaPolicy>;

    using EpiParts =
        cutlass::epilogue::threadblock::DefaultEpilogueSimt<ThreadblockShape,
            typename Mma::Operator,
            EpiOp,
            1>;
    using OutputTileIterator = cond_t<NativeC,
        cutlass::epilogue::threadblock::PredicatedTileIterator<
            typename EpiParts::OutputTileThreadMap,
            float>,
        cutlass::epilogue::threadblock::ChunkedEpilogueIterator<
            typename EpiParts::OutputTileThreadMap,
            float>>;
    using Epilogue = cutlass::epilogue::threadblock::Epilogue<ThreadblockShape,
        typename Mma::Operator,
        EpiParts::kPartitionsK,
        OutputTileIterator,
        typename EpiParts::AccumulatorFragmentIterator,
        typename EpiParts::WarpTileIterator,
        typename EpiParts::SharedLoadIterator,
        EpiOp,
        typename EpiParts::Padding>;
};

// =====================================================================
// Two kernels over one Mma and Epilogue.
//
// MixedKernel (plain): CTA b owns the whole k of output tile b of the
// identity-swizzled tile grid (tile t: m tile t % tiles_m, n tile
// t / tiles_m) and finishes C through the epilogue. It carries no
// split logic on purpose: a kernel that also holds the split schedule
// and the partial-store path costs the mainloop 16-24 registers, which
// on this architecture is the difference between two CTAs resident per
// SM and one.
//
// MixedPieceKernel: the pieces of a deterministic wave-balanced K split.
// The tiles [full, tiles) are cut into `slices` ascending-k panel ranges
// (the K panels spread as evenly as possible, in fixed order); CTA b
// computes piece b = (tile full + b / slices, range b % slices) and
// writes its RAW partial to its own contiguous slot of the workspace.
// The plain kernel on the tiles [0, full), this one, and the reduction
// below run back to back on the stream: fixed count and order, no
// atomics.
// =====================================================================

template <typename Mma_, typename Epilogue_> struct MixedKernel
{
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using OutputOp = typename Epilogue::OutputOp;
    static int const kM = Mma::Shape::kM;
    static int const kN = Mma::Shape::kN;
    static int const kK = Mma::Shape::kK;
    static int const kThreadCount = 32 * Mma::WarpCount::kCount;

    struct Params
    {
        cutlass::gemm::GemmCoord problem_size;
        int tiles_m; //!< output tiles along m
        int tiles_n; //!< output tiles along n
        typename Mma::IteratorA::Params params_A;
        typename Mma::IteratorA::TensorRef ref_A;
        typename Mma::IteratorB::Params params_B;
        typename Mma::IteratorB::TensorRef ref_B;
        typename Epilogue::OutputTileIterator::Params params_C;
        typename Epilogue::OutputTileIterator::TensorRef ref_C;
        typename Epilogue::OutputTileIterator::Params params_D;
        typename Epilogue::OutputTileIterator::TensorRef ref_D;
        typename OutputOp::Params output_op;

        CUTLASS_HOST_DEVICE
        Params() : tiles_m(0), tiles_n(0), output_op() {}
    };

    union SharedStorage
    {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

    CUTLASS_HOST_DEVICE
    MixedKernel() = default;

    CUTLASS_DEVICE
    void operator()(Params const &p, SharedStorage &shared_storage)
    {
        int const b = int(blockIdx.x);
        int const m_tile = b % p.tiles_m;
        int const n_tile = b / p.tiles_m;
        int const K = (p.problem_size.k() + kK - 1) / kK; //!< panels in k
        int const thread_idx = int(threadIdx.x);

        typename Mma::IteratorA iterator_A(p.params_A,
            p.ref_A.data(),
            {p.problem_size.m(), K * kK},
            thread_idx,
            {m_tile * kM, 0},
            nullptr);
        typename Mma::IteratorB iterator_B(p.params_B,
            p.ref_B.data(),
            {K * kK, p.problem_size.n()},
            thread_idx,
            {0, n_tile * kN},
            nullptr);

        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = __shfl_sync(0xffffffffu, thread_idx / 32, 0);
        int lane_idx = thread_idx % 32;

        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
        typename Mma::FragmentC accumulators;
        accumulators.clear();
        mma(K, accumulators, iterator_A, iterator_B, accumulators);

        Epilogue epilogue(
            shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
        cutlass::MatrixCoord const threadblock_offset(
            m_tile * kM, n_tile * kN);
        typename Epilogue::OutputTileIterator iterator_C(p.params_C,
            p.ref_C.data(),
            p.problem_size.mn(),
            thread_idx,
            threadblock_offset,
            nullptr);
        typename Epilogue::OutputTileIterator iterator_D(p.params_D,
            p.ref_D.data(),
            p.problem_size.mn(),
            thread_idx,
            threadblock_offset,
            nullptr);
        OutputOp output_op(p.output_op);
        epilogue(output_op, iterator_D, accumulators, iterator_C);
    }
};

template <typename Mma_, typename Epilogue_> struct MixedPieceKernel
{
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using OutputOp = typename Epilogue::OutputOp;
    static int const kM = Mma::Shape::kM;
    static int const kN = Mma::Shape::kN;
    static int const kK = Mma::Shape::kK;
    static int const kThreadCount = 32 * Mma::WarpCount::kCount;

    struct Params
    {
        cutlass::gemm::GemmCoord problem_size;
        int tiles_m;      //!< output tiles along m
        int split_full;   //!< tiles [0, split_full) are not this kernel's
        int split_slices; //!< pieces per remaining tile
        typename Mma::IteratorA::Params params_A;
        typename Mma::IteratorA::TensorRef ref_A;
        typename Mma::IteratorB::Params params_B;
        typename Mma::IteratorB::TensorRef ref_B;
        //! A partial slot: one tile, row-major, ld kN, flat.
        typename Epilogue::OutputTileIterator::Params params_ws;
        float *partials; //!< piece b owns slot b
        CUTLASS_HOST_DEVICE
        Params() :
            tiles_m(0), split_full(0), split_slices(1), partials(nullptr)
        {
        }
    };

    union SharedStorage
    {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

    CUTLASS_HOST_DEVICE
    MixedPieceKernel() = default;

    CUTLASS_DEVICE
    void operator()(Params const &p, SharedStorage &shared_storage)
    {
        int const b = int(blockIdx.x);
        int const s = p.split_slices;
        int const tile = p.split_full + b / s;
        int const slice = b % s;
        int const K = (p.problem_size.k() + kK - 1) / kK; //!< panels in k
        int const q = K / s, r = K - q * s;
        int const kb = slice * q + (slice < r ? slice : r);
        int const ke = kb + q + (slice < r ? 1 : 0);
        int const m_tile = tile % p.tiles_m;
        int const n_tile = tile / p.tiles_m;
        int const thread_idx = int(threadIdx.x);

        typename Mma::IteratorA iterator_A(p.params_A,
            p.ref_A.data(),
            {p.problem_size.m(), ke * kK},
            thread_idx,
            {m_tile * kM, kb * kK},
            nullptr);
        typename Mma::IteratorB iterator_B(p.params_B,
            p.ref_B.data(),
            {ke * kK, p.problem_size.n()},
            thread_idx,
            {kb * kK, n_tile * kN},
            nullptr);

        int warp_idx = __shfl_sync(0xffffffffu, thread_idx / 32, 0);
        int lane_idx = thread_idx % 32;

        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
        typename Mma::FragmentC accumulators;
        accumulators.clear();
        mma(ke - kb, accumulators, iterator_A, iterator_B, accumulators);

        Epilogue epilogue(
            shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
        float *slot = p.partials + int64_t(b) * (kM * kN);
        typename Epilogue::OutputTileIterator iterator_D(
            p.params_ws, slot, {kM, kN}, thread_idx, {0, 0}, nullptr);
        OutputOp partial_op(typename OutputOp::Params(1.f, 0.f));
        epilogue(partial_op, iterator_D, accumulators, iterator_D);
    }
};

//! The launch wrappers, as CUTLASS's own Kernel<> (no occupancy hint:
//! the compiler keeps the family's register budget) and with the hint
//! that two CTAs stay resident per SM (launch bounds; for the kernels
//! the compiler would otherwise give more than half the register file).
//! A hint of ONE CTA per SM is not the neutral choice: the compiler
//! then takes 224 registers for a mainloop that needs 116-137.
template <typename Kernel>
CUTLASS_DEVICE void mixed_device_body(
    typename Kernel::Params const &params, int *shared_storage_base) noexcept
{
    Kernel op;
    op(params,
        *reinterpret_cast<typename Kernel::SharedStorage *>(
            shared_storage_base));
}

template <typename Kernel>
__global__ void mixed_device_kernel(typename Kernel::Params const params)
{
    extern __shared__ int SharedStorageBase[];
    mixed_device_body<Kernel>(params, SharedStorageBase);
}

template <typename Kernel>
__global__ __launch_bounds__(Kernel::kThreadCount,
    2) void mixed_device_kernel_two(typename Kernel::Params const params)
{
    extern __shared__ int SharedStorageBase[];
    mixed_device_body<Kernel>(params, SharedStorageBase);
}

//! Launch one of them for `num_ctas` CTAs on `stream`; false (nothing
//! launched, or the launch failed) on a bad grid. Occup 1 asks for two
//! CTAs resident per SM.
template <typename Kernel, int Occup>
bool launch_kernel(typename Kernel::Params const &params,
    int num_ctas,
    cudaStream_t stream) noexcept
{
    if (num_ctas <= 0)
    {
        return true;
    }
    void (*kernel)(typename Kernel::Params const);
    if constexpr (Occup == 1)
    {
        kernel = &mixed_device_kernel_two<Kernel>;
    }
    else
    {
        kernel = &mixed_device_kernel<Kernel>;
    }
    int const smem = int(sizeof(typename Kernel::SharedStorage));
    kernel<<<num_ctas, Kernel::kThreadCount, smem, stream>>>(params);
    return cudaGetLastError() == cudaSuccess;
}

//! Deterministic finish of the split tiles: C = alpha * (sum of the
//! tile's partials, ascending slice) + beta * C, beta == 0 never reading
//! C. One thread per 4 consecutive outputs of a tile row (they share a
//! chunk: classes are at least 2^18 floats and the run starts at a
//! multiple of 4 elements, so it cannot straddle a chunk boundary).
//! Vec4 needs ld % 4 == 0 (tile row starts are multiples of 64).
template <int kM, int kN, bool Vec4>
__global__ void mixed_reduce(const float *__restrict__ ws,
    void *c,
    int c_shift,
    float alpha,
    float beta,
    int ld,
    int tiles_m,
    int full,
    int tiles,
    int slices)
{
    constexpr int kTile = kM * kN;      //!< outputs per tile
    constexpr int kWide = Vec4 ? 4 : 1; //!< outputs per thread
    constexpr int kPer = kTile / kWide; //!< threads per tile
    long long const total = static_cast<long long>(tiles - full) * kPer;
    long long const stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total;
         i += stride)
    {
        int const t = int(i / kPer);
        int const q = int(i % kPer);
        int const tile = full + t;
        int const m_tile = tile % tiles_m;
        int const n_tile = tile / tiles_m;
        int const row = q / (kN / kWide);
        int const col = (q % (kN / kWide)) * kWide;
        int64_t const e = int64_t(m_tile * kM + row) * ld + n_tile * kN + col;
        float *pc;
        if (c_shift == 0)
        {
            pc = static_cast<float *>(c) + e;
        }
        else
        {
            int const sh = c_shift >= 62 ? 62 : c_shift;
            int64_t const mask = (int64_t(1) << sh) - 1;
            pc = const_cast<float *>(
                     reinterpret_cast<float const *const *>(c)[e >> sh]) +
                 (e & mask);
        }
        float out[kWide];
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < kWide; ++w)
        {
            out[w] = 0.f;
        }
        float const *base = ws + int64_t(t) * slices * kTile + row * kN + col;
        for (int s = 0; s < slices; ++s)
        {
            CUTLASS_PRAGMA_UNROLL
            for (int w = 0; w < kWide; ++w)
            {
                out[w] += base[s * kTile + w];
            }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < kWide; ++w)
        {
            out[w] = alpha * out[w];
        }
        if (beta != 0.f)
        {
            CUTLASS_PRAGMA_UNROLL
            for (int w = 0; w < kWide; ++w)
            {
                out[w] += beta * pc[w];
            }
        }
        if constexpr (Vec4)
        {
            *reinterpret_cast<float4 *>(pc) =
                *reinterpret_cast<const float4 *>(out);
        }
        else
        {
            CUTLASS_PRAGMA_UNROLL
            for (int w = 0; w < kWide; ++w)
            {
                pc[w] = out[w];
            }
        }
    }
}

} // namespace nntile::kernel::gemm_chunked::ext::mixed_detail
