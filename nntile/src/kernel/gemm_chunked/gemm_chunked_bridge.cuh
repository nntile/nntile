/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * [campaign2/p7-bridge] THE CHUNKED BRIDGE: CUTLASS 2.11's exact SIMT
 * fp32 mainloop (MmaPipelined, threadblock 128x128x8, warp 32x64x8,
 * 2 stages, 256 threads -- the p5 "def" engine schedule) with the
 * global->smem stage swapped for CHUNK-TABLE-AWARE LOADERS at
 * PANEL-GRANULARITY RESOLUTION.
 *
 * Why this file exists (the round-6 lesson, g/r6-*): plugging table
 * resolution into CUTLASS's own scalar threadmaps costs one table
 * LDG.E.64 per ELEMENT access (208 table vs 144 data loads per
 * mainloop iteration -- 4.98-6.94 TF, dead). The bridge's iterators
 * own QUAD-CONTIGUOUS runs instead (4 floats along each operand's
 * storage-contiguous direction): ONE table resolve + ONE LDG.128 per
 * thread per operand per k-panel, i.e. table traffic is O(rows per
 * panel) -- amortized to noise (the p7 directive's pass/fail metric).
 * The quad ownership needs a matching smem-store side (written here);
 * the FFMA mainloop, warp-level readers, epilogue arithmetic, swizzle
 * and grid are CUTLASS's own templates, byte-unmodified -- the 96-98%
 * FMA-issue ceiling of the def engine rides for free.
 *
 * Per-combo quad maps (256 threads; A tile 128m x 8k, B tile 8k x 128n;
 * A storage: TA==0 RowMajor m x k (k-contig), TA==1 ColumnMajor k x m
 * (m-contig); B: TB==0 RowMajor k x n (n-contig), TB==1 ColumnMajor
 * n x k (k-contig)):
 *   contig-m/n side (A CM, B RM): thread t -> k-row t>>5, run (t&31)*4
 *     -> LDG.128 + STS.128 (smem is m/n-contiguous in that direction)
 *   contig-k side (A RM, B CM): thread t -> row t>>1, k-quad (t&1)*4
 *     -> LDG.128 + 4 STS.32 (transposed scalar smem stores, exactly
 *        CUTLASS's own store pattern for these sides)
 * The 16B intra-chunk law (classes >= 2^18 floats, class-aligned
 * bases, 4-aligned quad starts) forbids quad straddling chunk
 * boundaries, so the hot path never predicates on chunk distance.
 *
 * The epilogue and kernel-level template are the round-6 clones
 * (g/ch_epi.h, g/ch_kernel.h -- table-shift-aware, r6-verified
 * correct at chunk classes 1..8 MiB and all transpose combos);
 * the global iterators and smem stores are new. AC in {0,1} selects
 * flat (shift==0) vs table addressing per operand at compile time
 * (the campaign's kAcFlat/kAcTable axis; only (0,0) and (1,1) are
 * instantiated -- mixed addressing declines at launch).
 *
 * @version 1.0.0
 * */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/device_kernel.h"

namespace cutlass {
namespace transform {
namespace threadblock {

//! A-side global iterator: quad-contiguous, panel-resolving.
//! GL = layout::RowMajor (A is m x k, k-contiguous) or
//!      layout::ColumnMajor (A is k x m, m-contiguous).
//! Interface: the MmaPipelined/kernel::GemmChunked contract
//! (Params/TensorRef/Fragment/load/++/clear_mask; ctor as called by
//! the kernel: (Params, ptr, extent (m,k), thread_id, tile_offset
//! (m0,k0), gather=nullptr)).
//!
//! Tails = false (every instantiation before the tails piece): the
//! extent is ignored, as the shape law hands the iterator only
//! 128-multiples. Tails = true: the extent is honored -- a quad
//! outside it is zero-filled, a quad that is only partly inside or
//! not 16-byte aligned is loaded element by element -- so any shape
//! runs (a 1600-wide row, the 50257-wide vocabulary).
template <typename GL, int AC, bool Tails = false>
class BridgeQuadA {
public:
    using Element = float;
    using Layout = GL;
    using Index = int32_t;
    using LongIndex = int64_t;
    using TensorRef = TensorRef<Element const, Layout>;
    using TensorCoord = MatrixCoord;
    using Fragment = Array<Element, 4>;
    using AccessType = AlignedArray<Element, 4, 4>;

    static bool const kIsRowMajor
        = platform::is_same<GL, layout::RowMajor>::value;

    struct Params {
        int shift = 0;
        //! The operand's TRUE leading dimension (elements): the packed
        //! layout stride -- A-RM/B-CM carry k, A-CM/B-RM carry m/n.
        //! Round 2 fix: the split-K slices construct the iterator with
        //! a SLICE-CAPPED extent (problem_size_k of THIS slice's end),
        //! so the ld must come from here, never from the extent.
        LongIndex ld = 0;
        CUTLASS_HOST_DEVICE Params() = default;
        CUTLASS_HOST_DEVICE
        Params(Layout const &layout, int chunk_shift)
            : shift(chunk_shift), ld(layout.stride(0)) {}
    };

    CUTLASS_DEVICE
    BridgeQuadA(Params const &params, Element const *ptr,
        MatrixCoord extent, int thread_id, MatrixCoord tile_offset,
        int const * = nullptr)
        : ptr_(ptr)
    {
        if constexpr(kIsRowMajor)
        {
            //! A m x k row-major: quad of 4 k-consecutive floats of row
            //! (m0 + t>>1), starting k (t&1)*4 inside the panel.
            elem0_ = LongIndex(tile_offset.row() + (thread_id >> 1))
                * params.ld + tile_offset.column() + (thread_id & 1) * 4;
            d_ = 8;
            if constexpr(Tails)
            {
                //! The strided coordinate is the row m, the contiguous
                //! one the column k.
                strided_ = tile_offset.row() + (thread_id >> 1);
                strided_lim_ = extent.row();
                contig_ = tile_offset.column() + (thread_id & 1) * 4;
                contig_lim_ = extent.column();
            }
        }
        else
        {
            //! A k x m column-major storage: quad of 4 m-consecutive
            //! floats of k-row (k0 + t>>5), starting m (t&31)*4 inside
            //! the tile.
            elem0_ = LongIndex(tile_offset.column() + (thread_id >> 5))
                * params.ld + tile_offset.row() + (thread_id & 31) * 4;
            d_ = 8 * params.ld;
            if constexpr(Tails)
            {
                //! The strided coordinate is the row k, the contiguous
                //! one the column m.
                strided_ = tile_offset.column() + (thread_id >> 5);
                strided_lim_ = extent.column();
                contig_ = tile_offset.row() + (thread_id & 31) * 4;
                contig_lim_ = extent.row();
            }
        }
        se_ = params.shift >= 62 ? 62 : params.shift;
        valid_ = true;
    }

    //! Address of one element of the operand.
    CUTLASS_DEVICE Element const *at_(LongIndex elem) const
    {
        if constexpr(AC == 0)
        {
            return ptr_ + elem;
        }
        else
        {
            int64_t const mask = (int64_t(1) << se_) - 1;
            return reinterpret_cast<Element const *const *>(ptr_)[elem >> se_]
                + (elem & mask);
        }
    }

    CUTLASS_DEVICE
    void load(Fragment &frag) const
    {
        if(!valid_)
        {
            return; //!< masked (phantom last panel): caller pre-clears
        }
        if constexpr(Tails)
        {
            if(strided_ >= strided_lim_)
            {
                frag.clear(); //!< row outside the operand: zero-fill
                return;
            }
            if(contig_ + 4 <= contig_lim_ && (elem0_ & 3) == 0)
            {
                //! Whole quad inside and 16-byte aligned: one load (a
                //! 4-aligned quad never straddles a chunk).
                *reinterpret_cast<float4 *>(&frag)
                    = *reinterpret_cast<const float4 *>(at_(elem0_));
            }
            else
            {
                //! A tail quad, or a row that does not start 16-byte
                //! aligned: element by element, each through its own
                //! chunk.
                CUTLASS_PRAGMA_UNROLL
                for(int i = 0; i < 4; ++i)
                {
                    frag[i] = contig_ + i < contig_lim_
                        ? *at_(elem0_ + i)
                        : Element(0);
                }
            }
            return;
        }
        *reinterpret_cast<float4 *>(&frag)
            = *reinterpret_cast<const float4 *>(at_(elem0_));
    }

    CUTLASS_DEVICE BridgeQuadA &operator++()
    {
        elem0_ += d_;
        if constexpr(Tails)
        {
            // One panel further along K: K is the contiguous coordinate
            // of a row-major A, the strided one of a column-major A.
            if constexpr(kIsRowMajor)
            {
                contig_ += 8;
            }
            else
            {
                strided_ += 8;
            }
        }
        return *this;
    }

    CUTLASS_DEVICE void clear_mask(bool b) { valid_ = !b && valid_; }

private:
    Element const *ptr_;
    LongIndex elem0_;
    LongIndex d_;
    int se_;
    bool valid_;
    //! Tails only: the thread's strided coordinate and its limit, its
    //! contiguous coordinate (the quad's first element) and its limit.
    int strided_ = 0;
    int strided_lim_ = 0;
    int contig_ = 0;
    int contig_lim_ = 0;
};

//! B-side global iterator (mirror of BridgeQuadA; tile 8k x 128n).
//! GL = layout::RowMajor (B is k x n, n-contiguous) or
//!      layout::ColumnMajor (B is n x k, k-contiguous).
//! Tails as for BridgeQuadA.
template <typename GL, int AC, bool Tails = false>
class BridgeQuadB {
public:
    using Element = float;
    using Layout = GL;
    using Index = int32_t;
    using LongIndex = int64_t;
    using TensorRef = TensorRef<Element const, Layout>;
    using TensorCoord = MatrixCoord;
    using Fragment = Array<Element, 4>;
    using AccessType = AlignedArray<Element, 4, 4>;

    static bool const kIsRowMajor
        = platform::is_same<GL, layout::RowMajor>::value;

    struct Params {
        int shift = 0;
        //! The operand's TRUE leading dimension (elements; see the
        //! round-2 note on BridgeQuadA::Params -- the extent is
        //! slice-capped under split-K, the layout stride is not).
        LongIndex ld = 0;
        CUTLASS_HOST_DEVICE Params() = default;
        CUTLASS_HOST_DEVICE
        Params(Layout const &layout, int chunk_shift)
            : shift(chunk_shift), ld(layout.stride(0)) {}
    };

    CUTLASS_DEVICE
    BridgeQuadB(Params const &params, Element const *ptr,
        MatrixCoord extent, int thread_id, MatrixCoord tile_offset,
        int const * = nullptr)
        : ptr_(ptr)
    {
        if constexpr(kIsRowMajor)
        {
            //! B k x n row-major: quad of 4 n-consecutive floats of
            //! k-row (k0 + t>>5), starting n (t&31)*4 inside the tile.
            elem0_ = LongIndex(tile_offset.row() + (thread_id >> 5))
                * params.ld + tile_offset.column() + (thread_id & 31) * 4;
            d_ = 8 * params.ld;
            if constexpr(Tails)
            {
                strided_ = tile_offset.row() + (thread_id >> 5);
                strided_lim_ = extent.row();
                contig_ = tile_offset.column() + (thread_id & 31) * 4;
                contig_lim_ = extent.column();
            }
        }
        else
        {
            //! B n x k column-major storage: quad of 4 k-consecutive
            //! floats of row (n0 + t>>1), starting k (t&1)*4 inside the
            //! panel.
            elem0_ = LongIndex(tile_offset.column() + (thread_id >> 1))
                * params.ld + tile_offset.row() + (thread_id & 1) * 4;
            d_ = 8;
            if constexpr(Tails)
            {
                strided_ = tile_offset.column() + (thread_id >> 1);
                strided_lim_ = extent.column();
                contig_ = tile_offset.row() + (thread_id & 1) * 4;
                contig_lim_ = extent.row();
            }
        }
        se_ = params.shift >= 62 ? 62 : params.shift;
        valid_ = true;
    }

    //! Address of one element of the operand.
    CUTLASS_DEVICE Element const *at_(LongIndex elem) const
    {
        if constexpr(AC == 0)
        {
            return ptr_ + elem;
        }
        else
        {
            int64_t const mask = (int64_t(1) << se_) - 1;
            return reinterpret_cast<Element const *const *>(ptr_)[elem >> se_]
                + (elem & mask);
        }
    }

    CUTLASS_DEVICE
    void load(Fragment &frag) const
    {
        if(!valid_)
        {
            return;
        }
        if constexpr(Tails)
        {
            if(strided_ >= strided_lim_)
            {
                frag.clear();
                return;
            }
            if(contig_ + 4 <= contig_lim_ && (elem0_ & 3) == 0)
            {
                *reinterpret_cast<float4 *>(&frag)
                    = *reinterpret_cast<const float4 *>(at_(elem0_));
            }
            else
            {
                CUTLASS_PRAGMA_UNROLL
                for(int i = 0; i < 4; ++i)
                {
                    frag[i] = contig_ + i < contig_lim_
                        ? *at_(elem0_ + i)
                        : Element(0);
                }
            }
            return;
        }
        *reinterpret_cast<float4 *>(&frag)
            = *reinterpret_cast<const float4 *>(at_(elem0_));
    }

    CUTLASS_DEVICE BridgeQuadB &operator++()
    {
        elem0_ += d_;
        if constexpr(Tails)
        {
            // One panel further along K: K is the strided coordinate of
            // a row-major B, the contiguous one of a column-major B.
            if constexpr(kIsRowMajor)
            {
                strided_ += 8;
            }
            else
            {
                contig_ += 8;
            }
        }
        return *this;
    }

    CUTLASS_DEVICE void clear_mask(bool b) { valid_ = !b && valid_; }

private:
    Element const *ptr_;
    LongIndex elem0_;
    LongIndex d_;
    int se_;
    bool valid_;
    int strided_ = 0;
    int strided_lim_ = 0;
    int contig_ = 0;
    int contig_lim_ = 0;
};

//! A-side smem store iterator (the template MmaPipelined sees):
//! accepts the quad Fragment of BridgeQuadA and lands it in the
//! policy's ColumnMajor (m-contiguous) A stage. RM selects the quad
//! direction (the GLOBAL layout of A): k-quads (RM, A m x k) store 4
//! scalars at stride ld -- the transposed store, CUTLASS's own pattern,
//! conflict-free at ld = BM+4; m-quads (!RM) store one float4
//! (conflict-free: each 8 lanes cover 32 distinct banks of one 128B
//! window).
template <bool RM>
class BridgeStoreA {
public:
    using Element = float;
    using Layout = layout::ColumnMajor;
    using Index = int32_t;
    using LongIndex = int64_t;
    using TensorRef = TensorRef<Element, Layout>;
    using TensorCoord = MatrixCoord;

    CUTLASS_DEVICE
    BridgeStoreA(TensorRef const &ref, int thread_idx)
        : base_(ref.data())
    {
        ld_ = int(ref.stride(0));
        stage_ = LongIndex(8) * ld_;
        row_ = thread_idx >> 1;
        kq_ = (thread_idx & 1) * 4;
        kk_ = thread_idx >> 5;
        mr_ = (thread_idx & 31) * 4;
    }

    CUTLASS_DEVICE
    void store(Array<Element, 4> const &frag)
    {
        if constexpr(RM)
        {
            Element *p = base_ + row_ + LongIndex(kq_) * ld_;
            CUTLASS_PRAGMA_UNROLL
            for(int i = 0; i < 4; ++i)
            {
                p[LongIndex(i) * ld_] = frag[i];
            }
        }
        else
        {
            *reinterpret_cast<float4 *>(base_ + mr_ + LongIndex(kk_) * ld_)
                = *reinterpret_cast<const float4 *>(&frag);
        }
    }

    CUTLASS_DEVICE BridgeStoreA &operator++()
    {
        base_ += stage_;
        return *this;
    }

    CUTLASS_DEVICE void add_tile_offset(MatrixCoord const &c)
    {
        base_ += LongIndex(c.column()) * stage_;
    }

private:
    Element *base_;
    LongIndex stage_;
    int ld_;
    int row_, kq_, kk_, mr_;
};

//! B-side smem store: RowMajor (n-contiguous) B stage. RM (the GLOBAL
//! layout of B is RowMajor, k x n): n-quads store one float4;
//! transposed k-quads (!RM) store 4 scalars at stride ld
//! (conflict-free at ld = BN+4).
template <bool RM>
class BridgeStoreB {
public:
    using Element = float;
    using Layout = layout::RowMajor;
    using Index = int32_t;
    using LongIndex = int64_t;
    using TensorRef = TensorRef<Element, Layout>;
    using TensorCoord = MatrixCoord;

    CUTLASS_DEVICE
    BridgeStoreB(TensorRef const &ref, int thread_idx)
        : base_(ref.data())
    {
        ld_ = int(ref.stride(0));
        stage_ = LongIndex(8) * ld_;
        kk_ = thread_idx >> 5;
        nr_ = (thread_idx & 31) * 4;
        row_ = thread_idx >> 1;
        kq_ = (thread_idx & 1) * 4;
    }

    CUTLASS_DEVICE
    void store(Array<Element, 4> const &frag)
    {
        if constexpr(RM)
        {
            *reinterpret_cast<float4 *>(base_ + nr_ + LongIndex(kk_) * ld_)
                = *reinterpret_cast<const float4 *>(&frag);
        }
        else
        {
            Element *p = base_ + row_ + LongIndex(kq_) * ld_;
            CUTLASS_PRAGMA_UNROLL
            for(int i = 0; i < 4; ++i)
            {
                p[LongIndex(i) * ld_] = frag[i];
            }
        }
    }

    CUTLASS_DEVICE BridgeStoreB &operator++()
    {
        base_ += stage_;
        return *this;
    }

    CUTLASS_DEVICE void add_tile_offset(MatrixCoord const &c)
    {
        base_ += LongIndex(c.row()) * stage_;
    }

private:
    Element *base_;
    LongIndex stage_;
    int ld_;
    int kk_, nr_, row_, kq_;
};

} // namespace transform
} // namespace threadblock
} // namespace cutlass

// =====================================================================
// [campaign2/p7-bridge] The epilogue tile iterator is the ROUND-6
// clone (g/ch_epi.h, mechanically derived from
// cutlass/epilogue/threadblock/predicated_tile_iterator.h): the
// RowMajor primary specialization with per-guarded-access chunk
// resolution through the C/D table (row-granular via the register
// chunk hint -- one table load per output row per tile). r6-verified
// correct at chunk classes 2/4/8 MiB, all combos, tails and beta=1.
// =====================================================================

namespace cutlass {
namespace epilogue {
namespace threadblock {

//! Last-chunk register hint (0 = off; set via HINT=1 when generating).
static bool const CH_HINT = 1;
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_,         ///< Element data type
  bool ScatterD = false,     ///< Scatter D operand or not
  typename PermuteDLayout = layout::NoPermute,     ///< Permute D operand or not
  bool UseCUDAStore = false
>
class ChunkedEpilogueIterator {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static_assert( ThreadMap::Iterations::kRow > 0,"ThreadMap::Iterations::kRow must be > 0");
  static_assert( ThreadMap::Iterations::kGroup > 0,"ThreadMap::Iterations::kGroup must be > 0");
  static_assert( ThreadMap::Iterations::kCluster > 0,"ThreadMap::Iterations::kCluster must be > 0");
  static_assert( ThreadMap::Iterations::kColumn > 0,"ThreadMap::Iterations::kColumn must be > 0");

  /// Fragment object
  using Fragment = Array<
    Element, 
    ThreadMap::Iterations::kColumn * 
    ThreadMap::Iterations::kRow * 
    ThreadMap::Iterations::kGroup * 
    ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  //
  // Parameters struct
  //

  /// Uses a non-template class
  struct Params : PredicatedTileIteratorParams {
    using Base = PredicatedTileIteratorParams;

    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout): 
      PredicatedTileIteratorParams(
        layout.stride(0) * int(sizeof(AccessType)) / kElementsPerAccess,
        make_OutputTileThreadMapDesc<ThreadMap>()
      ) 
    { }

    CUTLASS_HOST_DEVICE
    Params(Base const &base) :
      Base(base) { }

    /// Chunk-table shift (round 6): 0 = flat pointer operand, else the
    /// element log2 of the power-of-two chunk class; >=62 = one chunk.
    int shift = 0;

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout, int chunk_shift) :
      PredicatedTileIteratorParams(
        layout.stride(0) * int(sizeof(AccessType)) / kElementsPerAccess,
        make_OutputTileThreadMapDesc<ThreadMap>()
      ), shift(chunk_shift)
    { }
  };

  /// Mask object
  struct Mask {

    static int const kCount = ThreadMap::Iterations::kColumn;

    /// Predicate state
    bool predicates[kCount];

    //
    // Mask
    //
    CUTLASS_HOST_DEVICE
    Mask() {
      enable();
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_HOST_DEVICE void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = false;
      }
    }

    ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
    CUTLASS_DEVICE void enable() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = true;
      }
    }
  };

private:

  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.
  Params params_;

  /// Byte-level pointer. This pointer is usually for both load() and store(), unless PermuteD is performed. When having PermuteD, byte_pointer_ is only for load().
  uint8_t *byte_pointer_;

  /// Byte-level pointer for store(). Due to PermuteD Op, store_byte_pointer_ may be with different address computation compared to byte_pointer_.
  uint8_t *store_byte_pointer_;

  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in rows
  Index extent_row_;

  /// Extent of the matrix tile in rows
  Index extent_column_;

  /// A thread's starting row position (assuming steady-state predicates have been computed)
  Index thread_start_row_;

  /// A thread's starting column
  Index thread_start_column_;

  /// Internal state counter
  int state_[3];

  /// Scatter indices
  int const *indices_; 

  /// Chunk-table shift for C/D (0 = flat pointer; runtime)
  int chunk_shift_;
  /// Flat virtual base (== chunk table when chunk_shift_ != 0)
  uint8_t const *chunk_base_;
  /// Last-chunk hint: one row's stores are consecutive elements of one
  /// chunk; the row-to-row jump only occasionally changes the chunk.
  mutable int64_t hint_idx_ = -1;
  mutable Element const *hint_base_ = nullptr;

  /// Whether to perform Permute Op
  bool PermuteD;
  /// PermuteDLayout
  mutable PermuteDLayout permute_layout_;
 
  //
  // Static asserts about internal strides
  //

  static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(PredicatedTileIteratorParams::stride) == 8, "Expected 64b strides");

private:

  //
  // Methods
  //

public:

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  ChunkedEpilogueIterator(
    Params const & params,
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord(),
    int const *indices = nullptr
  ): 
    params_(params), indices_(indices),
    chunk_shift_(params.shift),
    chunk_base_(reinterpret_cast<uint8_t const *>(pointer))
  {

    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_row_ = extent.row();
    extent_column_ = extent.column();

    thread_start_row_ = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {

      mask_.predicates[c] = ((thread_offset.column() 
        + ThreadMap::Delta::kColumn * c) < extent.column());
    }

    // Null pointer performs no accesses
    if (!pointer) {
      mask_.clear();
    }

    if (ScatterD && !indices) {
      mask_.clear();
    }

    // Initialize byte_pointer_
    byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) + 
      LongIndex(thread_offset.row()) * LongIndex(params_.stride) + 
      LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;

    if (ScatterD) {
      byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
        LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;
    }

    // store_byte_pointer_ is set to be the same with byte_pointer_ unless PermuteD is used.
    store_byte_pointer_ = byte_pointer_;

    // Initialize PermuteD. If PermuteD is true, store_byte_pointer_ is initialized accordingly.
    if (platform::is_same<PermuteDLayout, layout::NoPermute>::value) {
      PermuteD = false;
    }else{
      PermuteD = true;
      store_byte_pointer_ = reinterpret_cast<uint8_t *>(pointer);
      permute_layout_ = PermuteDLayout(extent, 
                          params_.stride * kElementsPerAccess / sizeof(AccessType));
    }

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    store_byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  /// Resolve linear element index through the chunk table (round 6),
  /// caching the last chunk entry (register hit within a row).
  CUTLASS_DEVICE
  AccessType *chunk_resolve_(int64_t elem) const {
    int const sh = chunk_shift_ >= 62 ? 62 : chunk_shift_;
    int64_t const mask = (int64_t(1) << sh) - 1;
    int64_t const idx = elem >> sh;
    Element const *base;
    if (CH_HINT and idx == hint_idx_) {
      base = hint_base_;
    } else {
      Element const *const *table
          = reinterpret_cast<Element const *const *>(chunk_base_);
      base = table[idx];
      if (CH_HINT) { hint_idx_ = idx; hint_base_ = base; }
    }
    Element *resolved = const_cast<Element *>(base + (elem & mask));
    return reinterpret_cast<AccessType *>(resolved);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, int64_t byte_offset) const {

    uint8_t *byte_pointer = byte_pointer_;
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx = 
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow 
            + group * ThreadMap::Delta::kGroup 
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          if (ScatterD && row_guard) {
            assert(indices_);

            memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset +
              LongIndex(indices_[row_offset + thread_start_row_]) * LongIndex(params_.stride));
          }

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            if (chunk_shift_ != 0) {
              if (guard) {
                int64_t elem = int64_t(byte_pointer + byte_offset - chunk_base_) * 8
                    / int(sizeof_bits<Element>::value)
                    + int64_t(column) * ThreadMap::Delta::kColumn;
                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                    frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn +
                             column],
                    (void *)chunk_resolve_(elem), true);
              }
            } else {
              cutlass::arch::global_load<
                AccessType, 
                sizeof(AccessType)
              >(
                  frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn +
                           column],
                  (void *)&memory_pointer[column * ThreadMap::Delta::kColumn /
                                          kElementsPerAccess],
                  guard);
            }
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            if (!ScatterD) {
              byte_pointer += params_.increment_row;
            }
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, int64_t byte_offset) const {
    uint8_t *byte_pointer = store_byte_pointer_;
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx = 
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow 
            + group * ThreadMap::Delta::kGroup 
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          if (ScatterD && row_guard) {
            assert(indices_);

            memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset +
              LongIndex(indices_[row_offset + thread_start_row_]) * LongIndex(params_.stride));
          }

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            int col_offset = column * ThreadMap::Delta::kColumn;

            if (chunk_shift_ != 0) {
              if (guard) {
                int64_t elem = int64_t(byte_pointer + byte_offset - chunk_base_) * 8
                    / int(sizeof_bits<Element>::value)
                    + int64_t(col_offset);
                cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                    frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                    (void *)chunk_resolve_(elem), true);
              }
              continue;
            }
            
            if (PermuteD) {
              int col = col_offset + thread_start_column_;
              int row = row_offset + thread_start_row_;

              TensorCoord init_coord(row, col);

              // Locate memory_pointer
              memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset
                 + permute_layout_(init_coord) * sizeof(AccessType) / kElementsPerAccess);
            }

            if (UseCUDAStore) {
              if (guard) {
                memory_pointer[0] =
                    frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];
              }
            } else {
              cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                  frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                  (void *)&memory_pointer[0],
                  guard);
            }
            
            if (!PermuteD) {
              memory_pointer += (ThreadMap::Delta::kColumn / kElementsPerAccess);
            }
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            if (!ScatterD && !PermuteD) {
              byte_pointer += params_.increment_row;
            }
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) const {

    store_with_byte_offset(frag, 0);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void downsample_load_with_byte_offset(Fragment &frag, int64_t byte_offset, int convolution_P, int convolution_Q, int add_P, int add_Q, int problem_N) const {

    uint8_t *byte_pointer = byte_pointer_;
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx = 
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow 
            + group * ThreadMap::Delta::kGroup 
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          int output_row = row_offset + thread_start_row_;
          int output_N = output_row / (convolution_P * convolution_Q);
          int output_PQ = output_row % (convolution_P * convolution_Q);
          int output_P = output_PQ / convolution_Q;
          int output_Q = output_PQ % convolution_Q;

          int input_row = output_N * 2 * convolution_P * 2 * convolution_Q +
            (2 * output_P + add_P) * 2 * convolution_Q + 2 * output_Q + add_Q;

          int64_t byte_offset = (input_row-output_row)*problem_N*sizeof(float);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            cutlass::arch::global_load<
              AccessType, 
              sizeof(AccessType)
            >(
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn +
                         column],
                (void *)&memory_pointer[column * ThreadMap::Delta::kColumn /
                                        kElementsPerAccess],
                guard);
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            byte_pointer += params_.increment_row;
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void upsample_load_with_byte_offset(Fragment &frag, int64_t byte_offset, int convolution_P, int convolution_Q, int add_P, int add_Q, int problem_N) const {

    uint8_t *byte_pointer = byte_pointer_;
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx = 
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow 
            + group * ThreadMap::Delta::kGroup 
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          int output_row = row_offset + thread_start_row_;
          int output_N = output_row / (convolution_P * convolution_Q);
          int output_PQ = output_row % (convolution_P * convolution_Q);
          int output_P = output_PQ / convolution_Q;
          int output_Q = output_PQ % convolution_Q;
          int row_add_P = add_P;
          int row_add_Q = add_Q;
	  if (output_P > convolution_P - 2) row_add_P = 0;
	  if (output_Q > convolution_Q - 2) row_add_Q = 0;

          int input_row = output_N * (convolution_P/2) * (convolution_Q/2) +
            ((output_P + row_add_P)/2) * (convolution_Q/2) + (output_Q + row_add_Q)/2;

          int64_t byte_offset = (input_row-output_row)*problem_N*sizeof(float);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            cutlass::arch::global_load<
              AccessType, 
              sizeof(AccessType)
            >(
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn +
                         column],
                (void *)&memory_pointer[column * ThreadMap::Delta::kColumn /
                                        kElementsPerAccess],
                guard);
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            byte_pointer += params_.increment_row;
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  CUTLASS_DEVICE
  MatrixCoord thread_start() const {
    return MatrixCoord(thread_start_row_, thread_start_column_);
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_row() const {
    return thread_start_row_;
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_column() const {
    return thread_start_column_;
  }

  /// Extent of the matrix in rows
  CUTLASS_DEVICE
  Index extent_row() const {
    return extent_row_;
  }

  /// Extent of the matrix in columns
  CUTLASS_DEVICE
  Index extent_column() const {
    return extent_column_;
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  ChunkedEpilogueIterator &operator++() {

    ++state_[0];

    if (!ScatterD && !PermuteD) {
      store_byte_pointer_ += params_.advance_row;
    }

    if (!ScatterD) {
      byte_pointer_ += params_.advance_row;
    }

    thread_start_row_ += ThreadMap::Shape::kRow;
    
    if (state_[0] == ThreadMap::Count::kRow) {

      state_[0] = 0;
      ++state_[1];
      byte_pointer_ += params_.advance_group;
      store_byte_pointer_ += params_.advance_group;

      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) * 
        ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {

        state_[1] = 0;
        ++state_[2];
        byte_pointer_ += params_.advance_cluster;
        store_byte_pointer_ += params_.advance_cluster;

        thread_start_row_ += ThreadMap::Count::kGroup * 
          ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
          byte_pointer_ += params_.advance_tile;
          store_byte_pointer_ += params_.advance_tile;

          thread_start_row_ += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow
            * ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;
        }
      }
    }

    return *this;
  }

  /// Advances a number of positions to load or store
  CUTLASS_HOST_DEVICE
  ChunkedEpilogueIterator &operator+=(int increment)
  {
    // Row
    state_[0] += increment;
    int increment_row = state_[0] / ThreadMap::Count::kRow;
    state_[0] = state_[0] % ThreadMap::Count::kRow;

    byte_pointer_ += (params_.advance_row * increment);
    store_byte_pointer_ += (params_.advance_row * increment);
    thread_start_row_ += (ThreadMap::Shape::kRow * increment);

    // Group
    state_[1] += increment_row;
    int increment_group = state_[1] / ThreadMap::Count::kGroup;
    state_[1] = state_[1] % ThreadMap::Count::kGroup;

    byte_pointer_ += (params_.advance_group * increment_row);
    store_byte_pointer_ += (params_.advance_group * increment_row);
    thread_start_row_ +=
        (ThreadMap::Shape::kGroup - 1) *
        ThreadMap::Shape::kRow *
        ThreadMap::Count::kRow *
        increment_row;


    // Cluster
    state_[2] += increment_group;
    int increment_cluster = state_[2] / ThreadMap::Count::kCluster;
    state_[2] = state_[2] % ThreadMap::Count::kCluster;

    byte_pointer_ += (params_.advance_cluster * increment_group);
    store_byte_pointer_ += (params_.advance_cluster * increment_group);
    thread_start_row_ +=
        ThreadMap::Count::kGroup *
        ThreadMap::Shape::kGroup *
        ThreadMap::Count::kRow *
        ThreadMap::Shape::kRow *
        increment_group;

    // Tile
    byte_pointer_ += (params_.advance_tile * increment_cluster);
    store_byte_pointer_ += (params_.advance_tile * increment_cluster);
    thread_start_row_ +=
        ThreadMap::Shape::kGroup *
        ThreadMap::Shape::kRow *
        ThreadMap::Shape::kCluster *
        ThreadMap::Shape::kTile *
        increment_cluster;

    return *this;
  }

  ///< Efficiently disables all accesses guarded by mask
  CUTLASS_DEVICE void clear_mask() {
    mask_.clear();
  }

  ///< Efficiently enables all accesses guarded by mask
  CUTLASS_DEVICE void enable_mask() {
    mask_.enable();
  }

  ///< Sets the mask
  CUTLASS_DEVICE void get_mask(Mask &mask) const {
    mask = mask_;
  }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const &mask) {
    mask_ = mask;
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load output tile from global memory in epilogue.
///
/// Satisfies: ReadableTileIterator | InterleavedPredicatedTileIterator | ForwardTileIterator
///

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

// =====================================================================
// [campaign2/p7-bridge] kernel::GemmChunked: the ROUND-6 clone of
// cutlass/gemm/kernel/gemm.h (g/ch_kernel.h) whose Params carry the
// per-operand chunk-table shifts; single k-slice (no split-K,
// deterministic ascending k).
// =====================================================================

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool SplitKSerial               ///! If true, code supporting split-K via serial reduction is enabled.
>
struct GemmChunked {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
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
    int *semaphore;
    int gemm_k_size;
    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), semaphore(0), gemm_k_size(0) { }

    // Chunk-table shifts (round 6): 0 = flat pointer operand, else the
    // element log2 of the power-of-two chunk class; the operand
    // TensorRef data pointers carry the chunk-pointer TABLE in that
    // case.
    int shift_A;
    int shift_B;
    int shift_C;
    int shift_D;

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D,
      typename OutputOp::Params output_op = typename OutputOp::Params(),
      int *workspace = nullptr,
      int const *gather_A_indices = nullptr,
      int const *gather_B_indices = nullptr,
      int const *scatter_D_indices = nullptr,
      int chunk_shift_A = 0,
      int chunk_shift_B = 0,
      int chunk_shift_C = 0,
      int chunk_shift_D = 0
    ):
      problem_size(problem_size),
      grid_tiled_shape(grid_tiled_shape),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(ref_A.layout(), chunk_shift_A),
      ref_A(ref_A),
      params_B(ref_B.layout(), chunk_shift_B),
      ref_B(ref_B),
      params_C(ref_C.layout(), chunk_shift_C),
      ref_C(ref_C),
      params_D(ref_D.layout(), chunk_shift_D),
      ref_D(ref_D),
      output_op(output_op),
      shift_A(chunk_shift_A),
      shift_B(chunk_shift_B),
      shift_C(chunk_shift_C),
      shift_D(chunk_shift_D),
      gather_A_indices(gather_A_indices),
      gather_B_indices(gather_B_indices),
      scatter_D_indices(scatter_D_indices) {

      int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
      
      gemm_k_size = gemm_k_iterations * Mma::Shape::kK;

    semaphore = workspace;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  GemmChunked() { } 

  /// Determines whether kernel satisfies alignment
  CUTLASS_HOST_DEVICE
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size,
    typename Mma::IteratorA::TensorRef ref_A,
    typename Mma::IteratorB::TensorRef ref_B,
    typename Epilogue::OutputTileIterator::TensorRef ref_C,
    typename Epilogue::OutputTileIterator::TensorRef ref_D) {

    static int const kAlignmentA = (platform::is_same<typename Mma::IteratorA::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorA::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB =  (platform::is_same<typename Mma::IteratorB::Layout,
                                                       layout::RowMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorB::Layout,
                                                        layout::RowMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_A, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.k() * params.gemm_k_size,
    };

    cutlass::MatrixCoord tb_offset_B{
      threadblock_tile_offset.k() * params.gemm_k_size,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k = min(
      params.problem_size.k(), 
      (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A,
      params.gather_A_indices);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      params.ref_B.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B,
      params.gather_B_indices);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    if (!kSplitKSerial || gemm_k_iterations > 0) {
      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
    }

    //
    // Epilogue
    //

    OutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      params.ref_C.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      params.scatter_D_indices
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      params.scatter_D_indices
    );

    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());

    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op, iterator_D, accumulators, iterator_C); 
    
    //
    // Release the semaphore
    //

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }

      semaphore.release(lock);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass


// =====================================================================
// [campaign2/p7-bridge] device::GemmBridge: the def-engine assembly
// (g/ch_device.h structure) with the global iterators swapped for the
// quad-contiguous table-resolving loaders and the smem stores for the
// quad-direction stores. Everything between the iterators -- the
// MmaPipelined mainloop, warp-level SIMT mma, DefaultEpilogueSimt
// chain, identity swizzle, 192-CTA-style plain grid -- is CUTLASS's
// own, byte-unmodified.
// =====================================================================

namespace cutlass {
namespace gemm {
namespace device {

template <
  typename ElementA_, typename LayoutA_, typename ElementB_,
  typename LayoutB_, typename ElementC_, typename LayoutC_,
  typename ElementAccumulator_ = float,
  typename OperatorClass_ = arch::OpClassSimt,
  typename ArchTag_ = arch::Sm50,
  typename ThreadblockShape_ = GemmShape<128, 128, 8>,
  typename WarpShape_ = GemmShape<32, 64, 8>,
  typename InstructionShape_ = GemmShape<1, 1, 1>,
  typename EpilogueOutputOp_ =
      epilogue::thread::LinearCombination<ElementC_, 1,
          ElementAccumulator_, ElementAccumulator_>,
  typename ThreadblockSwizzle_ =
      gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  int Stages = 2, int ACA = 1, int ACB = 1>
struct GemmBridge {

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static int const kStages = Stages;
  static int const kACA = ACA;
  static int const kACB = ACB;

  static_assert(kStages == 2, "SIMT MmaPipelined is hard-gated to 2 stages");
  static_assert(platform::is_same<OperatorClass, arch::OpClassSimt>::value,
                "bridge built on the SIMT schedule only");
  static_assert(platform::is_same<ElementA, float>::value &&
                platform::is_same<ElementB, float>::value &&
                platform::is_same<ElementC, float>::value,
                "fp32 campaign: float operands (chunk table of float*)");

  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC, arch::OpClassSimt, 2,
      arch::OpClassSimt>;

  using IteratorA = cutlass::transform::threadblock::BridgeQuadA<
      LayoutA, ACA>;
  using IteratorB = cutlass::transform::threadblock::BridgeQuadB<
      LayoutB, ACB>;

  static bool const kAQuadAlongK
      = platform::is_same<LayoutA, layout::RowMajor>::value;
  static bool const kBQuadAlongN
      = platform::is_same<LayoutB, layout::RowMajor>::value;

  using Mma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA,
      cutlass::transform::threadblock::BridgeStoreA<kAQuadAlongK>,
      IteratorB,
      cutlass::transform::threadblock::BridgeStoreB<kBQuadAlongN>,
      ElementAccumulator, LayoutC, typename MmaCore::MmaPolicy>;

  using EpiParts = cutlass::epilogue::threadblock::DefaultEpilogueSimt<
      ThreadblockShape, typename Mma::Operator, EpilogueOutputOp, 1>;

  using OutputTileIterator =
      cutlass::epilogue::threadblock::ChunkedEpilogueIterator<
          typename EpiParts::OutputTileThreadMap, ElementC>;

  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
      ThreadblockShape, typename Mma::Operator, EpiParts::kPartitionsK,
      OutputTileIterator, typename EpiParts::AccumulatorFragmentIterator,
      typename EpiParts::WarpTileIterator, typename EpiParts::SharedLoadIterator,
      EpilogueOutputOp, typename EpiParts::Padding>;

  using GemmKernel = cutlass::gemm::kernel::GemmChunked<
      Mma, Epilogue, ThreadblockSwizzle, false>;

  struct Arguments {
    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    TensorRef<ElementC const, LayoutC> ref_C;
    TensorRef<ElementC, LayoutC> ref_D;
    typename EpilogueOutputOp::Params epilogue;
    int shift_A;
    int shift_B;
    int shift_C;
    int shift_D;

    CUTLASS_HOST_DEVICE
    Arguments()
        : problem_size(0, 0, 0), shift_A(0), shift_B(0), shift_C(0),
          shift_D(0) {}
  };

private:
  typename GemmKernel::Params params_;

public:
  GemmBridge() {}

  static size_t get_workspace_size(Arguments const &) { return 0; }

  Status initialize(Arguments const &args, void * = nullptr,
                    cudaStream_t = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord grid_shape =
        threadblock_swizzle.get_tiled_shape(args.problem_size,
            {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
            1);

    params_ = typename GemmKernel::Params{
        args.problem_size,
        grid_shape,
        args.ref_A.non_const_ref(),
        args.ref_B.non_const_ref(),
        args.ref_C.non_const_ref(),
        args.ref_D,
        args.epilogue,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        args.shift_A,
        args.shift_B,
        args.shift_C,
        args.shift_D};
    return Status::kSuccess;
  }

  Status update(Arguments const &args, void *workspace = nullptr) {
    return initialize(args, workspace);
  }

  Status run(cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;
    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    cudaError_t result;
    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(
          Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);
    result = cudaGetLastError();
    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

  Status operator()(Arguments const &args, void *workspace = nullptr,
                    cudaStream_t stream = nullptr) {
    Status status = initialize(args, workspace, stream);
    if (status == Status::kSuccess) {
      status = run(stream);
    }
    return status;
  }
};

} // namespace device
} // namespace gemm
} // namespace cutlass

// =====================================================================
// [campaign2/p7-bridge, round 2] kernel::GemmChunkedSplitKParallel: the
// library's gemm_splitk_parallel.h structure on the bridge's Mma --
// each (tile, k-slice) CTA computes its k-range's RAW partial into a
// flat workspace (slice-major, stride m*n; partial output op is the
// identity LinearCombination(1,0), no source read). The deterministic
// chunk-aware reduction that finishes C lives in the kernel file
// (bridge_sk_reduce): fixed ascending-slice summation, no atomics.
// Chunk plumbing mirrors the GemmChunked clone above (A/B table +
// shifts; D is the FLAT workspace, shift_D == 0 by construction).
// =====================================================================

#include "cutlass/gemm/kernel/gemm_splitk_parallel.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template <
  typename Mma_,
  typename Epilogue_,
  typename ThreadblockSwizzle_
>
struct GemmChunkedSplitKParallel {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  static int const kAlignmentK = Mma::Operator::Shape::kK;

  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorB::TensorRef ref_B;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;
    int64_t splitk_slice_stride;
    int gemm_k_size;
    int shift_A;
    int shift_B;
    int shift_D;

    CUTLASS_HOST_DEVICE
    Params() : swizzle_log_tile(0), splitk_slice_stride(0), gemm_k_size(0),
        shift_A(0), shift_B(0), shift_D(0) { }

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size_,
      cutlass::gemm::GemmCoord const & grid_tiled_shape_,
      typename Mma::IteratorA::TensorRef ref_A_,
      typename Mma::IteratorB::TensorRef ref_B_,
      typename Epilogue::OutputTileIterator::TensorRef ref_D_,
      typename OutputOp::Params output_op_ =
          typename OutputOp::Params(),
      int chunk_shift_A = 0,
      int chunk_shift_B = 0,
      int chunk_shift_D = 0
    ):
      problem_size(problem_size_),
      grid_tiled_shape(grid_tiled_shape_),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape_)),
      params_A(ref_A_.layout(), chunk_shift_A),
      ref_A(ref_A_),
      params_B(ref_B_.layout(), chunk_shift_B),
      ref_B(ref_B_),
      params_D(ref_D_.layout(), chunk_shift_D),
      ref_D(ref_D_),
      output_op(output_op_),
      splitk_slice_stride(
          int64_t(problem_size_.m()) * int64_t(problem_size_.n())),
      shift_A(chunk_shift_A),
      shift_B(chunk_shift_B),
      shift_D(chunk_shift_D) {

      int const full_gemm_k_iterations = problem_size_.k() / Mma::Shape::kK;
      int const gemm_k_iterations
          = full_gemm_k_iterations / grid_tiled_shape_.k();

      gemm_k_size = gemm_k_iterations * Mma::Shape::kK;
    }
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  CUTLASS_HOST_DEVICE
  GemmChunkedSplitKParallel() { }

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
      return;
    }

    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.k() * params.gemm_k_size,
    };

    cutlass::MatrixCoord tb_offset_B{
      threadblock_tile_offset.k() * params.gemm_k_size,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    int problem_size_k;
    if (threadblock_tile_offset.k() + 1 == params.grid_tiled_shape.k()) {
      problem_size_k = params.problem_size.k();
    }
    else {
      problem_size_k = (threadblock_tile_offset.k() + 1) * params.gemm_k_size;
    }

    int gemm_k_iterations = (problem_size_k - tb_offset_A.column()
        + Mma::Shape::kK - 1) / Mma::Shape::kK;

    int thread_idx = threadIdx.x;

    typename Mma::IteratorA iterator_A(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A,
      nullptr);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      params.ref_B.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B,
      nullptr);

    int warp_idx = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B,
        accumulators);

    OutputOp output_op(params.output_op);

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    iterator_D.add_pointer_offset(
        params.splitk_slice_stride * threadblock_tile_offset.k());

    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    epilogue(output_op, iterator_D, accumulators, iterator_D);
  }
};

} // namespace kernel
} // namespace gemm
} // namespace cutlass
