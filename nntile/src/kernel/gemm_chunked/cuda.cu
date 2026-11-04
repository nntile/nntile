/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm_chunked/cuda.cu
 * Native chunked fp32 GEMM kernels (see cuda.hh). One parametric kernel
 * family over (TA, TB, BM, BN, BK); the picker instantiates per shape
 * family. All operands resolve through chunk tables (or a flat base, or
 * the one-shot sentinel) with per-row resolution in the panel loaders.
 *
 * @version 1.0.0
 * */

// Corresponding header (round-7 working tree: local shim copy)
#include "nntile/kernel/gemm_chunked/cuda.hh"

#ifdef NNTILE_USE_CUDA

// Standard libraries
#include <cstdio>

namespace nntile::kernel::gemm_chunked
{

namespace
{

//! Resolve element `elem` of one operand's linear space to a device
//! pointer (see Operand in cuda.hh for the three addressing modes).
//! `elem` already carries any batch offset: the batch stride moves the
//! element index, hence the chunk, exactly as the flat layout does.
__device__ __forceinline__ const float *chunk_at(
    const void *tab_or_flat, int shift, long long elem)
{
    if(shift == 0)
    {
        return reinterpret_cast<const float *>(tab_or_flat) + elem;
    }
    const float *const *tab = reinterpret_cast<const float *const *>(
        tab_or_flat);
    if(shift >= 62)
    {
        return tab[0] + elem;
    }
    long long const mask = (1LL << shift) - 1;
    return tab[elem >> shift] + (elem & mask);
}

__device__ __forceinline__ float *chunk_at_mut(
    void *tab_or_flat, int shift, long long elem)
{
    if(shift == 0)
    {
        return reinterpret_cast<float *>(tab_or_flat) + elem;
    }
    float *const *tab = reinterpret_cast<float *const *>(tab_or_flat);
    if(shift >= 62)
    {
        return tab[0] + elem;
    }
    long long const mask = (1LL << shift) - 1;
    return tab[elem >> shift] + (elem & mask);
}

//! Addressing classes for the hot loader (compile-time axis): FLAT is
//! a plain base pointer (shift == 0), TABLE is the chunk-pointer table
//! (any other shift; shifts >= 62 clamp to 62, where elem >> 62 == 0
//! for every representable element — exactly the one-shot sentinel).
constexpr int kAcFlat = 0;
constexpr int kAcTable = 1;

//! Count of VALID LEADING elements of a 16-run: the `ok(i)` predicate
//! family (outer row/k bound && i < linear limit) collapsed to one int.
__device__ __forceinline__ int nv_of(bool outer, int limit)
{
    int nv = outer ? limit : 0;
    return nv < 0 ? 0 : (nv > 16 ? 16 : nv);
}

//! The 128-bit intra-chunk law (owner observation, round 5): chunk
//! classes are power-of-two byte sizes of at least 1 MiB and chunk
//! bases are class-aligned, so every 16-byte-ALIGNED 128-bit access
//! lies wholly inside ONE chunk -- a straddle would need the access
//! start within 16 bytes of a power-of-two class boundary, which
//! alignment forbids.
//! Consequence: the loaders' only fast-path predicate is (elem0 & 3) ==
//! 0 plus shape bounds; boundary-distance arithmetic is unnecessary for
//! aligned vector accesses. Unaligned runs (odd leading dims, e.g. the
//! 50257-wide vocab rows) and tails still resolve element-wise.
struct Run16
{
    float4 v[4];
};

//! Cold tail of the run loader: per-element chunk resolution (chunk
//! straddles, 4-element misalignment, boundary tails). Outlined with
//! __noinline__ so its register frame never inflates the hot loop's
//! live set. Elements >= nvalid zero-fill and are never resolved —
//! exactly the `ok(i)` semantics it replaces.
__noinline__ __device__ Run16 load_run16_cold(const void *tab, int shift,
    long long elem0, int nvalid)
{
    Run16 r;
#pragma unroll
    for(int i = 0; i < 16; ++i)
    {
        (&r.v[i >> 2].x)[i & 3]
            = i < nvalid ? *chunk_at(tab, shift, elem0 + i) : 0.f;
    }
    return r;
}

//! Hot run loader, specialized on the addressing class: 16 valid,
//! 4-aligned, straddle-free runs take the straight-line float4 path
//! (one resolved base pointer + four LDG.128, no per-element fallback
//! code in the hot body); everything else falls into the outlined
//! cold tail with the original semantics. A run straddles a chunk
//! boundary iff its offset inside the chunk reaches span-15 (one
//! compare, hoisted here from the old split arithmetic).
template <int AC>
__device__ __forceinline__ Run16 load_run16(const void *tab, int shift,
    long long elem0, int nvalid)
{
    if constexpr(AC == kAcFlat)
    {
        if(nvalid == 16 && (elem0 & 3) == 0)
        {
            const float *q0 = reinterpret_cast<const float *>(tab) + elem0;
            Run16 r;
#pragma unroll
            for(int q = 0; q < 4; ++q)
            {
                r.v[q] = *reinterpret_cast<const float4 *>(q0 + q * 4);
            }
            return r;
        }
    }
    else
    {
        int const se = shift >= 62 ? 62 : shift;
        long long const span = 1LL << se;
        if(nvalid == 16 && (elem0 & 3) == 0
            && (elem0 & (span - 1)) < span - 15)
        {
            const float *q0
                = reinterpret_cast<const float *const *>(tab)[elem0 >> se]
                + (elem0 & (span - 1));
            Run16 r;
#pragma unroll
            for(int q = 0; q < 4; ++q)
            {
                r.v[q] = *reinterpret_cast<const float4 *>(q0 + q * 4);
            }
            return r;
        }
    }
    return load_run16_cold(tab, shift, elem0, nvalid);
}

//! Cold tail of the C epilogue: the per-element store for 4-groups
//! that are boundary tails, misaligned, or chunk-straddling. Outlined
//! so its frame stays out of the hot function's live set. Elements
//! beyond `span` are not resolved at all.
__noinline__ __device__ void store_c_tail(void *c, int c_shift,
    long long elem0, int span, bool beta_zero, float beta, float4 v)
{
    float w[4] = {v.x, v.y, v.z, v.w};
    for(int e = 0; e < span; ++e)
    {
        float *el = chunk_at_mut(c, c_shift, elem0 + e);
        if(!beta_zero)
        {
            w[e] += beta * (*el);
        }
        *el = w[e];
    }
}

struct Params
{
    const void *a;
    int a_shift;
    const void *b;
    int b_shift;
    void *c;
    int c_shift;
    int m;
    int n;
    int k;
    int batch;
    long long sa; //!< A batch stride in elements (0 = broadcast)
    long long sb; //!< B batch stride in elements (0 = broadcast)
    long long sc; //!< C batch stride in elements
    float alpha;
    float beta;
};

//! XOR swizzle at float4-group granularity: a row of BK floats holds
//! BK/4 float4 groups; group g of row r stores at g ^ (r & (groups-1)).
//! With BK=32 (8 groups) the swizzle is 3 bits; consecutive rows land
//! on distinct banks for both the (transposing) writes and the
//! column-ish reads of the FFMA loop. Applied symmetrically by the
//! loader and the compute loop.
//! KEYSH selects which row bits drive the XOR: 0 for k-row-major
//! layouts (the k row index itself varies across the warp), 3 for
//! m/n-row-major layouts, where the FFMA loop reads rows striding by 8
//! (TM) — row&7 is then warp-constant and keying on it would make
//! every read an 8-way bank conflict; row>>3 varies instead.
template <int W, int KEYSH = 0>
__device__ __forceinline__ int swz_col(int col, int row)
{
    constexpr int Groups = W / 4;
    return ((col >> 2)
        ^ (((row >> KEYSH) ^ row) & (Groups - 1))) * 4
        + (col & 3);
}

//! The one kernel family: C(m,n) = alpha*op(A)*op(B) + beta*C over
//! chunk-table operands, C-order row-major math. Configurable thread
//! tile TM x TN (config A: 8x8 over 256 threads; config B: 8x16 over
//! 128 threads — half the shared-memory traffic per FMA, the sm75
//! SIMT sweet spot at 128 accumulator registers). Lane grid is always
//! 8 rows x 4 cols; warp tiles derive from TM/TN. Double-buffered
//! shared-memory pipeline (global->register prefetch of the next
//! panel; no cp.async on sm_75). Panels sit in smem in STORAGE
//! orientation with an XOR swizzle, so every load, store and FFMA read
//! finds its 4-groups contiguous in exactly one direction.
//! Register cap probe for the round-4 hypothesis: D_MAXREG compiles the
//! whole classic family under a hard __maxnreg__ ceiling (255 default =
//! no constraint; 180 = the hypothesis's target live set).
#ifndef D_MAXREG
#define D_MAXREG 255
#endif

#if D_MAXREG < 255
#define D_LB_QUALS __maxnreg__(D_MAXREG)
#else
#define D_LB_QUALS __launch_bounds__(THREADS, MINB)
#endif

template <int TA, int TB, int ACA, int ACB, int BM, int BN, int BK, int TM,
    int TN, int THREADS, int MINB = 1, int STAGES = 2, int PIPE = 2>
__global__ D_LB_QUALS void sgemm_chunked_kernel(Params p)
{
    constexpr int kLanes = 32;
    constexpr int kLRows = 8;
    constexpr int kLCols = 4;
    constexpr int kWarpM = kLRows * TM;      // 64
    constexpr int kWarpN = kLCols * TN;      // 32 or 64
    constexpr int kWRows = BM / kWarpM;      // 2
    constexpr int kWCols = BN / kWarpN;      // 4 or 2
    constexpr int kWarps = kWRows * kWCols;
    static_assert(kWarps * kLanes == THREADS, "thread count mismatch");
    static_assert(BM % kWarpM == 0 && BN % kWarpN == 0, "tile split");
    constexpr int kRunsA = BM * BK / 16 / THREADS;
    constexpr int kRunsB = BK * BN / 16 / THREADS;
    //! Split-loader mode (the 2-CTA/SM config): the A panel's runs
    //! (BM*BK/16) and B panel's (BK*BN/16) together equal THREADS, so
    //! the first threads own A whole-panel runs and the rest own B's —
    //! one Run16 per thread per panel, half the register staging.
    constexpr bool kSplit = (BM * BK / 16) + (BK * BN / 16) == THREADS
        && (BM * BK / 16) < THREADS;
    static_assert(
        (kRunsA >= 1 && kRunsB >= 1) || kSplit, "loader under-subscribed");

    extern __shared__ float smem[];
    float *As = smem;
    float *Bs = smem + STAGES * BM * BK;

    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const warp_row = warp / kWCols;
    int const warp_col = warp % kWCols;
    int const lane_row = lane >> 2;
    int const lane_col = lane & 3;
    int const thr_m0 = warp_row * kWarpM + lane_row * TM;
    int const thr_n0 = warp_col * kWarpN + lane_col * TN;

    float areg[kRunsA > 0 ? kRunsA : 1][16];
    float breg[kRunsB > 0 ? kRunsB : 1][16];

    //! Load one operand panel into the staging registers. General mode:
    //! every thread loads kRunsA runs of A and kRunsB runs of B. Split
    //! mode (kSplit): the A panel's BM*BK/16 runs go to the low threads
    //! and B's to the rest — one run per thread, halving the staging
    //! registers (the 2-CTA/SM budget).
    auto load_panel = [&](int panel, int m0, int n0, long long a_base,
                          long long b_base) {
        int const k0 = panel * BK;
        if constexpr(kSplit)
        {
            constexpr int kARuns = BM * BK / 16;
            if(tid < kARuns)
            {
                if(TA == 0)
                {
                    // Whole 16-k run of row m0+tid.
                    int const grow = m0 + tid;
                    long long const elem0 = a_base
                        + static_cast<long long>(grow) * p.k + k0;
                    Run16 const run = load_run16<ACA>(p.a, p.a_shift,
                        elem0, nv_of(grow < p.m, p.k - k0));
#pragma unroll
                    for(int q = 0; q < 4; ++q)
                    {
                        areg[0][q * 4 + 0] = run.v[q].x;
                        areg[0][q * 4 + 1] = run.v[q].y;
                        areg[0][q * 4 + 2] = run.v[q].z;
                        areg[0][q * 4 + 3] = run.v[q].w;
                    }
                }
                else
                {
                    // 16-m run of k row k0 + (tid>>3).
                    int const kk = tid >> 3;
                    int const mm0 = (tid & 7) * 16;
                    int const gk = k0 + kk;
                    long long const elem0 = a_base
                        + static_cast<long long>(gk) * p.m + m0 + mm0;
                    Run16 const run = load_run16<ACA>(p.a, p.a_shift,
                        elem0,
                        nv_of(gk < p.k, p.m - (m0 + mm0)));
#pragma unroll
                    for(int q = 0; q < 4; ++q)
                    {
                        areg[0][q * 4 + 0] = run.v[q].x;
                        areg[0][q * 4 + 1] = run.v[q].y;
                        areg[0][q * 4 + 2] = run.v[q].z;
                        areg[0][q * 4 + 3] = run.v[q].w;
                    }
                }
            }
            else
            {
                int const t = tid - kARuns;
                if(TB == 0)
                {
                    int const kk = t >> 3;
                    int const nn0 = (t & 7) * 16;
                    int const gk = k0 + kk;
                    long long const elem0 = b_base
                        + static_cast<long long>(gk) * p.n + n0 + nn0;
                    Run16 const run = load_run16<ACB>(p.b, p.b_shift,
                        elem0,
                        nv_of(gk < p.k, p.n - (n0 + nn0)));
#pragma unroll
                    for(int q = 0; q < 4; ++q)
                    {
                        breg[0][q * 4 + 0] = run.v[q].x;
                        breg[0][q * 4 + 1] = run.v[q].y;
                        breg[0][q * 4 + 2] = run.v[q].z;
                        breg[0][q * 4 + 3] = run.v[q].w;
                    }
                }
                else
                {
                    int const gn = n0 + t;
                    long long const elem0 = b_base
                        + static_cast<long long>(gn) * p.k + k0;
                    Run16 const run = load_run16<ACB>(p.b, p.b_shift,
                        elem0, nv_of(gn < p.n, p.k - k0));
#pragma unroll
                    for(int q = 0; q < 4; ++q)
                    {
                        breg[0][q * 4 + 0] = run.v[q].x;
                        breg[0][q * 4 + 1] = run.v[q].y;
                        breg[0][q * 4 + 2] = run.v[q].z;
                        breg[0][q * 4 + 3] = run.v[q].w;
                    }
                }
            }
            return;
        }
        if(TA == 0)
        {
            // A stored [m,k]: runs of consecutive k inside one row.
            constexpr int kTPR = THREADS / BM; // threads per row
            int const mm = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsA; ++r)
            {
                int const kk0 = slot * (BK / kTPR) + r * 16;
                int const grow = m0 + mm;
                long long const elem0 = a_base
                    + static_cast<long long>(grow) * p.k + k0 + kk0;
                Run16 const run = load_run16<ACA>(p.a, p.a_shift, elem0,
                    nv_of(grow < p.m, p.k - (k0 + kk0)));
#pragma unroll
                for(int q = 0; q < 4; ++q)
                {
                    areg[r][q * 4 + 0] = run.v[q].x;
                    areg[r][q * 4 + 1] = run.v[q].y;
                    areg[r][q * 4 + 2] = run.v[q].z;
                    areg[r][q * 4 + 3] = run.v[q].w;
                }
            }
        }
        else
        {
            // A stored [k,m]: runs of consecutive m inside one k row.
            constexpr int kTPR = THREADS / BK; // threads per k row
            int const kk = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsA; ++r)
            {
                int const mm0 = slot * (BM / kTPR) + r * 16;
                int const gk = k0 + kk;
                long long const elem0 = a_base
                    + static_cast<long long>(gk) * p.m + m0 + mm0;
                Run16 const run = load_run16<ACA>(p.a, p.a_shift, elem0,
                    nv_of(gk < p.k, p.m - (m0 + mm0)));
#pragma unroll
                for(int q = 0; q < 4; ++q)
                {
                    areg[r][q * 4 + 0] = run.v[q].x;
                    areg[r][q * 4 + 1] = run.v[q].y;
                    areg[r][q * 4 + 2] = run.v[q].z;
                    areg[r][q * 4 + 3] = run.v[q].w;
                }
            }
        }
        if(TB == 0)
        {
            // B stored [k,n]: runs of consecutive n inside one k row.
            constexpr int kTPR = THREADS / BK;
            int const kk = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsB; ++r)
            {
                int const nn0 = slot * (BN / kTPR) + r * 16;
                int const gk = k0 + kk;
                long long const elem0 = b_base
                    + static_cast<long long>(gk) * p.n + n0 + nn0;
                Run16 const run = load_run16<ACB>(p.b, p.b_shift, elem0,
                    nv_of(gk < p.k, p.n - (n0 + nn0)));
#pragma unroll
                for(int q = 0; q < 4; ++q)
                {
                    breg[r][q * 4 + 0] = run.v[q].x;
                    breg[r][q * 4 + 1] = run.v[q].y;
                    breg[r][q * 4 + 2] = run.v[q].z;
                    breg[r][q * 4 + 3] = run.v[q].w;
                }
            }
        }
        else
        {
            // B stored [n,k]: runs of consecutive k inside one n row.
            constexpr int kTPR = THREADS / BN;
            int const nn = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsB; ++r)
            {
                int const kk0 = slot * (BK / kTPR) + r * 16;
                int const gn = n0 + nn;
                long long const elem0 = b_base
                    + static_cast<long long>(gn) * p.k + k0 + kk0;
                Run16 const run = load_run16<ACB>(p.b, p.b_shift, elem0,
                    nv_of(gn < p.n, p.k - (k0 + kk0)));
#pragma unroll
                for(int q = 0; q < 4; ++q)
                {
                    breg[r][q * 4 + 0] = run.v[q].x;
                    breg[r][q * 4 + 1] = run.v[q].y;
                    breg[r][q * 4 + 2] = run.v[q].z;
                    breg[r][q * 4 + 3] = run.v[q].w;
                }
            }
        }
    };

    //! Store the staged runs into the swizzled panel layouts, mirroring
    //! the loader's ownership (split mode: only the owning half moves).
    auto store_panel = [&](int stage, int m0, int n0) {
        float *as = As + stage * BM * BK;
        float *bs = Bs + stage * BK * BN;
        if constexpr(kSplit)
        {
            constexpr int kARuns = BM * BK / 16;
            if(tid < kARuns)
            {
                if(TA == 0)
                {
#pragma unroll
                    for(int q = 0; q < 4; ++q)
                    {
                        int const kk = q * 4;
                        *reinterpret_cast<float4 *>(
                            as + tid * BK + swz_col<BK, 3>(kk, tid))
                            = *reinterpret_cast<float4 *>(areg[0] + q * 4);
                    }
                }
                else
                {
                    int const kk = tid >> 3;
                    int const mm0 = (tid & 7) * 16;
#pragma unroll
                    for(int q = 0; q < 4; ++q)
                    {
                        int const mm = mm0 + q * 4;
                        *reinterpret_cast<float4 *>(
                            as + kk * BM + swz_col<BM>(mm, kk))
                            = *reinterpret_cast<float4 *>(areg[0] + q * 4);
                    }
                }
            }
            else
            {
                int const t = tid - kARuns;
                if(TB == 0)
                {
                    int const kk = t >> 3;
                    int const nn0 = (t & 7) * 16;
#pragma unroll
                    for(int q = 0; q < 4; ++q)
                    {
                        *reinterpret_cast<float4 *>(
                            bs + kk * BN + nn0 + q * 4)
                            = *reinterpret_cast<float4 *>(breg[0] + q * 4);
                    }
                }
                else
                {
#pragma unroll
                    for(int q = 0; q < 4; ++q)
                    {
                        int const kk = q * 4;
                        *reinterpret_cast<float4 *>(
                            bs + t * BK + swz_col<BK, 3>(kk, t))
                            = *reinterpret_cast<float4 *>(breg[0] + q * 4);
                    }
                }
            }
            return;
        }
        if(TA == 0)
        {
            constexpr int kTPR = THREADS / BM;
            int const mm = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsA; ++r)
            {
                int const kk0 = slot * (BK / kTPR) + r * 16;
#pragma unroll
                for(int q = 0; q < 4; ++q)
                {
                    int const kk = kk0 + q * 4;
                    *reinterpret_cast<float4 *>(
                        as + mm * BK + swz_col<BK, 3>(kk, mm))
                        = *reinterpret_cast<float4 *>(areg[r] + q * 4);
                }
            }
        }
        else
        {
            constexpr int kTPR = THREADS / BK;
            int const kk = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsA; ++r)
            {
                int const mm0 = slot * (BM / kTPR) + r * 16;
#pragma unroll
                for(int q = 0; q < 4; ++q)
                {
                    int const mm = mm0 + q * 4;
                    *reinterpret_cast<float4 *>(
                        as + kk * BM + swz_col<BM>(mm, kk))
                        = *reinterpret_cast<float4 *>(areg[r] + q * 4);
                }
            }
        }
        if(TB == 0)
        {
            constexpr int kTPR = THREADS / BK;
            int const kk = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsB; ++r)
            {
                int const nn0 = slot * (BN / kTPR) + r * 16;
#pragma unroll
                for(int q = 0; q < 4; ++q)
                {
                    *reinterpret_cast<float4 *>(
                        bs + kk * BN + nn0 + q * 4)
                        = *reinterpret_cast<float4 *>(breg[r] + q * 4);
                }
            }
        }
        else
        {
            constexpr int kTPR = THREADS / BN;
            int const nn = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsB; ++r)
            {
                int const kk0 = slot * (BK / kTPR) + r * 16;
#pragma unroll
                for(int q = 0; q < 4; ++q)
                {
                    int const kk = kk0 + q * 4;
                    *reinterpret_cast<float4 *>(
                        bs + nn * BK + swz_col<BK, 3>(kk, nn))
                        = *reinterpret_cast<float4 *>(breg[r] + q * 4);
                }
            }
        }
    };

    float acc[TM][TN];
#pragma unroll
    for(int i = 0; i < TM; ++i)
    {
#pragma unroll
        for(int j = 0; j < TN; ++j)
        {
            acc[i][j] = 0.f;
        }
    }

    auto compute = [&](int panel) {
        float const *as = As + (panel % STAGES) * BM * BK;
        float const *bs = Bs + (panel % STAGES) * BK * BN;
        //! PIPE-deep shared-load pipeline: the vectorized operand
        //! fragments (av4 for NoTrans A, bv4 for Trans B) rotate
        //! through PIPE register sets; the LDS.128 group feeding
        //! k-step t+PIPE-1 is issued BEFORE step t's FFMA block, so
        //! every shared load has PIPE-1 full FFMA blocks of latency
        //! cover instead of stalling the warp on the scoreboard (the
        //! campaign's short_scoreboard killer). PIPE=1 is the
        //! unpipelined Triton shape (load at loop top, consume
        //! immediately). The transposed per-q loads (Trans A / NoTrans
        //! B) are already interleaved with the FFMAs and stay as they
        //! were. Each extra pipe step costs 64 registers on the NT
        //! combo — the register attack's currency.
        constexpr int K4 = BK / 4;
        float4 av4[PIPE][TM];
        float4 bv4[PIPE][TN];
        if(TA == 0)
        {
#pragma unroll
            for(int f = 0; f < PIPE - 1; ++f)
            {
#pragma unroll
                for(int i = 0; i < TM; ++i)
                {
                    av4[f][i] = *reinterpret_cast<const float4 *>(
                        as + (thr_m0 + i) * BK
                            + swz_col<BK, 3>(4 * f, thr_m0 + i));
                }
            }
        }
        if(TB == 1)
        {
#pragma unroll
            for(int f = 0; f < PIPE - 1; ++f)
            {
#pragma unroll
                for(int j = 0; j < TN; ++j)
                {
                    bv4[f][j] = *reinterpret_cast<const float4 *>(
                        bs + (thr_n0 + j) * BK
                            + swz_col<BK, 3>(4 * f, thr_n0 + j));
                }
            }
        }
        // Vectorized FFMA loop, k in steps of 4: the swizzle keeps
        // every 4-group contiguous along exactly one axis per
        // layout (NoTrans A along k, Trans A along m, NoTrans B
        // along n, Trans B along k), so LDS.128 replaces four
        // LDS.32 issue slots. The fragment group feeding k-step
        // t+PIPE-1 is loaded before step t's FFMA block.
#pragma unroll
            for(int kk = 0; kk < BK; kk += 4)
            {
                int const t = kk >> 2;
                int const s = t % PIPE;
                if(t + PIPE - 1 < K4)
                {
                    int const kpref = kk + 4 * (PIPE - 1);
                    if(TA == 0)
                    {
#pragma unroll
                        for(int i = 0; i < TM; ++i)
                        {
                            av4[(t + PIPE - 1) % PIPE][i] =
                                *reinterpret_cast<const float4 *>(as
                                    + (thr_m0 + i) * BK
                                    + swz_col<BK, 3>(kpref, thr_m0 + i));
                        }
                    }
                    if(TB == 1)
                    {
#pragma unroll
                        for(int j = 0; j < TN; ++j)
                        {
                            bv4[(t + PIPE - 1) % PIPE][j] =
                                *reinterpret_cast<const float4 *>(bs
                                    + (thr_n0 + j) * BK
                                    + swz_col<BK, 3>(kpref, thr_n0 + j));
                        }
                    }
                }
                //! Transposed-side fragments (Trans A reads along m,
                //! NoTrans B along n) rotate ONE q-step ahead of their
                //! consumers: the LDS.128 group feeding q+1 is issued
                //! before q's 64-FFMA block (the same latency-runway
                //! treatment the vectorized av4/bv4 groups get at
                //! kk-step granularity).
                float4 af[2][TM / 4];
                float4 bf[2][TN / 4];
                if(TA == 1)
                {
#pragma unroll
                    for(int g = 0; g < TM; g += 4)
                    {
                        af[0][g >> 2] = *reinterpret_cast<const float4 *>(
                            as + kk * BM
                                + swz_col<BM>(thr_m0 + g, kk));
                    }
                }
                if(TB == 0)
                {
#pragma unroll
                    for(int g = 0; g < TN; g += 4)
                    {
                        bf[0][g >> 2] = *reinterpret_cast<const float4 *>(
                            bs + kk * BN + thr_n0 + g);
                    }
                }
#pragma unroll
                for(int q = 0; q < 4; ++q)
                {
                    float av[TM];
                    float bv[TN];
                    if(TA == 0)
                    {
#pragma unroll
                        for(int i = 0; i < TM; ++i)
                        {
                            av[i] = (&av4[s][i].x)[q];
                        }
                    }
                    else
                    {
                        if(q + 1 < 4)
                        {
#pragma unroll
                            for(int g = 0; g < TM; g += 4)
                            {
                                af[(q + 1) & 1][g >> 2] =
                                    *reinterpret_cast<const float4 *>(
                                        as + (kk + q + 1) * BM
                                            + swz_col<BM>(thr_m0 + g,
                                                kk + q + 1));
                            }
                        }
#pragma unroll
                        for(int g = 0; g < TM; g += 4)
                        {
                            float4 const a = af[q & 1][g >> 2];
                            av[g + 0] = a.x;
                            av[g + 1] = a.y;
                            av[g + 2] = a.z;
                            av[g + 3] = a.w;
                        }
                    }
                    if(TB == 0)
                    {
                        if(q + 1 < 4)
                        {
#pragma unroll
                            for(int g = 0; g < TN; g += 4)
                            {
                                bf[(q + 1) & 1][g >> 2] =
                                    *reinterpret_cast<const float4 *>(
                                        bs + (kk + q + 1) * BN + thr_n0 + g);
                            }
                        }
#pragma unroll
                        for(int g = 0; g < TN; g += 4)
                        {
                            float4 const b = bf[q & 1][g >> 2];
                            bv[g + 0] = b.x;
                            bv[g + 1] = b.y;
                            bv[g + 2] = b.z;
                            bv[g + 3] = b.w;
                        }
                    }
                    else
                    {
#pragma unroll
                        for(int j = 0; j < TN; ++j)
                        {
                            bv[j] = (&bv4[s][j].x)[q];
                        }
                    }
#pragma unroll
                    for(int i = 0; i < TM; ++i)
                    {
#pragma unroll
                        for(int j = 0; j < TN; ++j)
                        {
                            acc[i][j] += av[i] * bv[j];
                        }
                    }
                }
            }
        };
    // Persistent tile loop: the launcher caps the grid at the SM count
    // times the co-residency; blocks grid-stride over the (n, m, batch)
    // tile space, n-fastest so consecutive blocks share A row bands.
    // For grids at or below the cap this executes exactly one
    // iteration — the same code path.
    int const tiles_x = (p.n + BN - 1) / BN;
    int const tiles_y = (p.m + BM - 1) / BM;
    long long const ntiles = static_cast<long long>(tiles_x) * tiles_y
        * p.batch;
    for(long long tile = static_cast<long long>(blockIdx.x)
                       + static_cast<long long>(blockIdx.y) * gridDim.x;
        tile < ntiles;
        tile += static_cast<long long>(gridDim.x) * gridDim.y)
    {
        long long const rem = tile / p.batch;
        int const bi = static_cast<int>(tile % p.batch);
        int const m0 = static_cast<int>(rem % tiles_y) * BM;
        int const n0 = static_cast<int>(rem / tiles_y) * BN;
        long long const a_base = bi * p.sa;
        long long const b_base = bi * p.sb;
        long long const c_base = bi * p.sc;

        int const npanels = (p.k + BK - 1) / BK;

        if constexpr(STAGES == 1)
        {
            //! Triton-shape pipeline (the 2-CTA/SM configuration): ONE
            //! smem panel per operand (32 KiB) co-resides with a second
            //! CTA on the SM, so the co-resident CTA's FFMA stream hides
            //! each CTA's barrier and smem-load latencies. The panel's
            //! global->register fetch is issued a FULL PANEL of compute
            //! before its smem store; the store only overwrites the
            //! live buffer after a barrier separates it from the
            //! compute that just read it. Two barriers per panel, same
            //! as the double-buffered shape, and ascending-k order is
            //! untouched.
            load_panel(0, m0, n0, a_base, b_base);
            store_panel(0, m0, n0);
            if(1 < npanels)
            {
                load_panel(1, m0, n0, a_base, b_base);
            }
            __syncthreads();
            for(int panel = 0; panel < npanels; ++panel)
            {
                compute(panel);
                if(panel + 1 < npanels)
                {
                    __syncthreads();
                    store_panel(0, m0, n0);
                    if(panel + 2 < npanels)
                    {
                        load_panel(panel + 2, m0, n0, a_base, b_base);
                    }
                    __syncthreads();
                }
            }
        }
        else
        {
            load_panel(0, m0, n0, a_base, b_base);
            for(int panel = 0; panel < npanels; ++panel)
            {
                int const stage = panel & 1;
                store_panel(stage, m0, n0);
                if(panel + 1 < npanels)
                {
                    load_panel(panel + 1, m0, n0, a_base, b_base);
                }
                __syncthreads();
                compute(panel);
                __syncthreads();
            }
        }

        // ---- epilogue: alpha/beta and the chunk-resolved C store. The
        // C row may straddle chunk boundaries (odd n guarantees it), so
        // every 4-element group resolves its own base exactly like the
        // loaders; the vectorized store only fires when the whole group
        // lies in one chunk at a 16B-aligned offset. ----
        bool const beta_zero = (p.beta == 0.f);
#pragma unroll
        for(int i = 0; i < TM; ++i)
        {
            int const gm = m0 + thr_m0 + i;
            if(gm >= p.m)
            {
                continue;
            }
#pragma unroll
            for(int j = 0; j < TN; j += 4)
            {
                int const gn = n0 + thr_n0 + j;
                if(gn >= p.n)
                {
                    continue;
                }
                long long const elem0 = c_base
                    + static_cast<long long>(gm) * p.n + gn;
                float v[4];
                v[0] = p.alpha * acc[i][j + 0];
                v[1] = p.alpha * acc[i][j + 1];
                v[2] = p.alpha * acc[i][j + 2];
                v[3] = p.alpha * acc[i][j + 3];
                // Group fully in bounds, 16B-aligned, one chunk?
                int const span = gn + 3 < p.n ? 4 : p.n - gn;
                bool const same_chunk = p.c_shift == 0
                    || p.c_shift >= 62
                    || ((elem0 >> p.c_shift)
                        == ((elem0 + 3) >> p.c_shift));
                if(span == 4 && (elem0 & 3) == 0 && same_chunk)
                {
                    float4 *dst = reinterpret_cast<float4 *>(
                        chunk_at_mut(p.c, p.c_shift, elem0));
                    if(!beta_zero)
                    {
                        float4 const old = *dst;
                        v[0] += p.beta * old.x;
                        v[1] += p.beta * old.y;
                        v[2] += p.beta * old.z;
                        v[3] += p.beta * old.w;
                    }
                    *dst = make_float4(v[0], v[1], v[2], v[3]);
                }
                else
                {
                    store_c_tail(p.c, p.c_shift, elem0, span, beta_zero,
                        p.beta, make_float4(v[0], v[1], v[2], v[3]));
                }
            }
        }
        // Reset the accumulators for the next grid-stride tile.
#pragma unroll
        for(int i = 0; i < TM; ++i)
        {
#pragma unroll
            for(int j = 0; j < TN; ++j)
            {
                acc[i][j] = 0.f;
            }
        }
    }
}

//! Generalized run loader: R consecutive elements (R in {4,8,16}) with
//! EXACTLY the addressing semantics of load_run16 — a run takes the
//! float4 vector path only when every element is valid AND the whole
//! run sits inside one chunk at a 16B-aligned offset; anything else
//! (straddle, tail, misalignment) resolves per element through
//! chunk_at, zero-filling rejected elements and never resolving an
//! out-of-table index.
template <int R, class Ok>
__device__ __forceinline__ void load_run(const void *tab, int shift,
    long long elem0, Ok ok, float *dst)
{
    bool all = true;
#pragma unroll
    for(int i = 0; i < R; ++i)
    {
        all = all && ok(i);
    }
    if(all)
    {
        long long split = R;
        if(shift > 0 && shift < 62)
        {
            long long const b = ((elem0 >> shift) + 1) << shift;
            long long const s = b - elem0;
            split = s < R ? s : R;
        }
        if(split == R && (elem0 & 3) == 0)
        {
            const float *q0 = chunk_at(tab, shift, elem0);
#pragma unroll
            for(int q = 0; q < R / 4; ++q)
            {
                *reinterpret_cast<float4 *>(dst + q * 4)
                    = *reinterpret_cast<const float4 *>(q0 + q * 4);
            }
        }
        else
        {
#pragma unroll
            for(int i = 0; i < R; ++i)
            {
                dst[i] = *chunk_at(tab, shift, elem0 + i);
            }
        }
    }
    else
    {
#pragma unroll
        for(int i = 0; i < R; ++i)
        {
            dst[i] = ok(i) ? *chunk_at(tab, shift, elem0 + i) : 0.f;
        }
    }
}

//! Multi-stage thin-panel variant of the kernel family (the sm75 SIMT
//! direction): same lane grid (8x4), thread tile and swizzled smem
//! layout as sgemm_chunked_kernel, but panels are BK=8/16-deep, smem
//! holds S stages, and the global->register->smem pipeline keeps TWO
//! panels of loads in flight: the loader for panel q issues at the top
//! of panel q-2's compute and its smem store lands at the end of panel
//! q-1 (two full panels of latency runway per load). The main loop is
//! unrolled by 2 so the register staging buffers rotate with
//! compile-time indices. Loader granularity generalizes the 16-float
//! run to RUN floats (RUN=8 for BK=16, RUN=4 for BK=8) with identical
//! chunk addressing semantics (see load_run above).
template <int TA, int TB, int BM, int BN, int BK, int TM, int TN,
    int THREADS, int RUN, int S, int MINB>
__global__ __launch_bounds__(THREADS, MINB) void sgemm_chunked_kernel_ms(
    Params p)
{
    constexpr int kLanes = 32;
    constexpr int kLRows = 8;
    constexpr int kLCols = 4;
    constexpr int kWarpM = kLRows * TM;
    constexpr int kWarpN = kLCols * TN;
    constexpr int kWRows = BM / kWarpM;
    constexpr int kWCols = BN / kWarpN;
    constexpr int kWarps = kWRows * kWCols;
    static_assert(kWarps * kLanes == THREADS, "thread count mismatch");
    static_assert(BM % kWarpM == 0 && BN % kWarpN == 0, "tile split");
    constexpr int kRunsA = BM * BK / RUN / THREADS;
    constexpr int kRunsB = BK * BN / RUN / THREADS;
    static_assert(kRunsA >= 1 && kRunsB >= 1, "loader under-subscribed");
    static_assert(RUN == 4 || RUN == 8 || RUN == 16, "run width");
    static_assert(S >= 2, "need at least two stages");

    extern __shared__ float smem[];
    float *As = smem;
    float *Bs = smem + S * BM * BK;

    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const warp_row = warp / kWCols;
    int const warp_col = warp % kWCols;
    int const lane_row = lane >> 2;
    int const lane_col = lane & 3;
    int const thr_m0 = warp_row * kWarpM + lane_row * TM;
    int const thr_n0 = warp_col * kWarpN + lane_col * TN;

    int const m0 = blockIdx.y * BM;
    int const n0 = blockIdx.x * BN;
    int const bi = blockIdx.z;
    long long const a_base = bi * p.sa;
    long long const b_base = bi * p.sb;
    long long const c_base = bi * p.sc;

    int const npanels = (p.k + BK - 1) / BK;

    float areg[2][kRunsA][RUN];
    float breg[2][kRunsB][RUN];

    auto ld = [&](int panel, int buf) {
        int const k0 = panel * BK;
        if(TA == 0)
        {
            constexpr int kTPR = THREADS / BM;
            int const mm = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsA; ++r)
            {
                int const kk0 = slot * (BK / kTPR) + r * RUN;
                int const grow = m0 + mm;
                long long const elem0 = a_base
                    + static_cast<long long>(grow) * p.k + k0 + kk0;
                load_run<RUN>(p.a, p.a_shift, elem0,
                    [&](int i) {
                        return grow < p.m && k0 + kk0 + i < p.k;
                    },
                    areg[buf][r]);
            }
        }
        else
        {
            constexpr int kTPR = THREADS / BK;
            int const kk = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsA; ++r)
            {
                int const mm0 = slot * (BM / kTPR) + r * RUN;
                int const gk = k0 + kk;
                long long const elem0 = a_base
                    + static_cast<long long>(gk) * p.m + m0 + mm0;
                load_run<RUN>(p.a, p.a_shift, elem0,
                    [&](int i) {
                        return gk < p.k && m0 + mm0 + i < p.m;
                    },
                    areg[buf][r]);
            }
        }
        if(TB == 0)
        {
            constexpr int kTPR = THREADS / BK;
            int const kk = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsB; ++r)
            {
                int const nn0 = slot * (BN / kTPR) + r * RUN;
                int const gk = k0 + kk;
                long long const elem0 = b_base
                    + static_cast<long long>(gk) * p.n + n0 + nn0;
                load_run<RUN>(p.b, p.b_shift, elem0,
                    [&](int i) {
                        return gk < p.k && n0 + nn0 + i < p.n;
                    },
                    breg[buf][r]);
            }
        }
        else
        {
            constexpr int kTPR = THREADS / BN;
            int const nn = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsB; ++r)
            {
                int const kk0 = slot * (BK / kTPR) + r * RUN;
                int const gn = n0 + nn;
                long long const elem0 = b_base
                    + static_cast<long long>(gn) * p.k + k0 + kk0;
                load_run<RUN>(p.b, p.b_shift, elem0,
                    [&](int i) {
                        return gn < p.n && k0 + kk0 + i < p.k;
                    },
                    breg[buf][r]);
            }
        }
    };

    auto st = [&](int panel, int buf) {
        float *as = As + (panel % S) * BM * BK;
        float *bs = Bs + (panel % S) * BK * BN;
        if(TA == 0)
        {
            constexpr int kTPR = THREADS / BM;
            int const mm = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsA; ++r)
            {
                int const kk0 = slot * (BK / kTPR) + r * RUN;
#pragma unroll
                for(int q = 0; q < RUN / 4; ++q)
                {
                    int const kk = kk0 + q * 4;
                    *reinterpret_cast<float4 *>(
                        as + mm * BK + swz_col<BK, 3>(kk, mm))
                        = *reinterpret_cast<float4 *>(areg[buf][r] + q * 4);
                }
            }
        }
        else
        {
            constexpr int kTPR = THREADS / BK;
            int const kk = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsA; ++r)
            {
                int const mm0 = slot * (BM / kTPR) + r * RUN;
#pragma unroll
                for(int q = 0; q < RUN / 4; ++q)
                {
                    int const mm = mm0 + q * 4;
                    *reinterpret_cast<float4 *>(
                        as + kk * BM + swz_col<BM>(mm, kk))
                        = *reinterpret_cast<float4 *>(areg[buf][r] + q * 4);
                }
            }
        }
        if(TB == 0)
        {
            constexpr int kTPR = THREADS / BK;
            int const kk = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsB; ++r)
            {
                int const nn0 = slot * (BN / kTPR) + r * RUN;
#pragma unroll
                for(int q = 0; q < RUN / 4; ++q)
                {
                    *reinterpret_cast<float4 *>(bs + kk * BN + nn0 + q * 4)
                        = *reinterpret_cast<float4 *>(breg[buf][r] + q * 4);
                }
            }
        }
        else
        {
            constexpr int kTPR = THREADS / BN;
            int const nn = tid / kTPR;
            int const slot = tid % kTPR;
#pragma unroll
            for(int r = 0; r < kRunsB; ++r)
            {
                int const kk0 = slot * (BK / kTPR) + r * RUN;
#pragma unroll
                for(int q = 0; q < RUN / 4; ++q)
                {
                    int const kk = kk0 + q * 4;
                    *reinterpret_cast<float4 *>(
                        bs + nn * BK + swz_col<BK, 3>(kk, nn))
                        = *reinterpret_cast<float4 *>(breg[buf][r] + q * 4);
                }
            }
        }
    };

    float acc[TM][TN];
#pragma unroll
    for(int i = 0; i < TM; ++i)
    {
#pragma unroll
        for(int j = 0; j < TN; ++j)
        {
            acc[i][j] = 0.f;
        }
    }

    auto compute = [&](int panel) {
        float const *as = As + (panel % S) * BM * BK;
        float const *bs = Bs + (panel % S) * BK * BN;
#pragma unroll
        for(int kk = 0; kk < BK; kk += 4)
        {
            float4 av4[TM];
            if(TA == 0)
            {
#pragma unroll
                for(int i = 0; i < TM; ++i)
                {
                    av4[i] = *reinterpret_cast<const float4 *>(
                        as + (thr_m0 + i) * BK
                            + swz_col<BK, 3>(kk, thr_m0 + i));
                }
            }
            float4 bv4[TN];
            if(TB == 1)
            {
#pragma unroll
                for(int j = 0; j < TN; ++j)
                {
                    bv4[j] = *reinterpret_cast<const float4 *>(
                        bs + (thr_n0 + j) * BK
                            + swz_col<BK, 3>(kk, thr_n0 + j));
                }
            }
#pragma unroll
            for(int q = 0; q < 4; ++q)
            {
                float av[TM];
                float bv[TN];
                if(TA == 0)
                {
#pragma unroll
                    for(int i = 0; i < TM; ++i)
                    {
                        av[i] = (&av4[i].x)[q];
                    }
                }
                else
                {
#pragma unroll
                    for(int g = 0; g < TM; g += 4)
                    {
                        float4 const a = *reinterpret_cast<const float4 *>(
                            as + (kk + q) * BM
                                + swz_col<BM>(thr_m0 + g, kk + q));
                        av[g + 0] = a.x;
                        av[g + 1] = a.y;
                        av[g + 2] = a.z;
                        av[g + 3] = a.w;
                    }
                }
                if(TB == 0)
                {
#pragma unroll
                    for(int g = 0; g < TN; g += 4)
                    {
                        float4 const b = *reinterpret_cast<const float4 *>(
                            bs + (kk + q) * BN + thr_n0 + g);
                        bv[g + 0] = b.x;
                        bv[g + 1] = b.y;
                        bv[g + 2] = b.z;
                        bv[g + 3] = b.w;
                    }
                }
                else
                {
#pragma unroll
                    for(int j = 0; j < TN; ++j)
                    {
                        bv[j] = (&bv4[j].x)[q];
                    }
                }
#pragma unroll
                for(int i = 0; i < TM; ++i)
                {
#pragma unroll
                    for(int j = 0; j < TN; ++j)
                    {
                        acc[i][j] += av[i] * bv[j];
                    }
                }
            }
        }
    };

    // Prologue: panels 0,1 into staging buffers 0,1; panel 0 lands in
    // smem stage 0. Main loop (unrolled by 2 for compile-time buffer
    // rotation): at panel q, issue the loader for q+2 (top), compute q
    // (middle, between barriers), store q+1 from the other buffer
    // (end). Two panels of global loads are in flight throughout.
    ld(0, 0);
    if(1 < npanels)
    {
        ld(1, 1);
    }
    st(0, 0);
    int base = 0;
    for(; base + 1 < npanels; base += 2)
    {
        if(base + 2 < npanels)
        {
            ld(base + 2, 0);
        }
        __syncthreads();
        compute(base);
        __syncthreads();
        st(base + 1, 1);
        if(base + 3 < npanels)
        {
            ld(base + 3, 1);
        }
        __syncthreads();
        compute(base + 1);
        __syncthreads();
        if(base + 2 < npanels)
        {
            st(base + 2, 0);
        }
    }
    if(base < npanels)
    {
        __syncthreads();
        compute(base);
        __syncthreads();
    }

    // ---- epilogue: identical to sgemm_chunked_kernel ----
    bool const beta_zero = (p.beta == 0.f);
#pragma unroll
    for(int i = 0; i < TM; ++i)
    {
        int const gm = m0 + thr_m0 + i;
        if(gm >= p.m)
        {
            continue;
        }
#pragma unroll
        for(int j = 0; j < TN; j += 4)
        {
            int const gn = n0 + thr_n0 + j;
            if(gn >= p.n)
            {
                continue;
            }
            long long const elem0 = c_base
                + static_cast<long long>(gm) * p.n + gn;
            float v[4];
            v[0] = p.alpha * acc[i][j + 0];
            v[1] = p.alpha * acc[i][j + 1];
            v[2] = p.alpha * acc[i][j + 2];
            v[3] = p.alpha * acc[i][j + 3];
            int const span = gn + 3 < p.n ? 4 : p.n - gn;
            bool const same_chunk = p.c_shift == 0 || p.c_shift >= 62
                || ((elem0 >> p.c_shift)
                    == ((elem0 + 3) >> p.c_shift));
            if(span == 4 && (elem0 & 3) == 0 && same_chunk)
            {
                float4 *dst = reinterpret_cast<float4 *>(
                    chunk_at_mut(p.c, p.c_shift, elem0));
                if(!beta_zero)
                {
                    float4 const old = *dst;
                    v[0] += p.beta * old.x;
                    v[1] += p.beta * old.y;
                    v[2] += p.beta * old.z;
                    v[3] += p.beta * old.w;
                }
                *dst = make_float4(v[0], v[1], v[2], v[3]);
            }
            else
            {
#pragma unroll
                for(int e = 0; e < 4; ++e)
                {
                    if(gn + e < p.n)
                    {
                        float *el = chunk_at_mut(p.c, p.c_shift,
                            elem0 + e);
                        if(!beta_zero)
                        {
                            v[e] += p.beta * (*el);
                        }
                        *el = v[e];
                    }
                }
            }
        }
    }
}

//! Clamp of (outer-bound, remaining-elements) to a run-8 validity count.
__device__ __forceinline__ int nv8(bool outer, int rem)
{
    if(!outer || rem <= 0)
    {
        return 0;
    }
    return rem < 8 ? rem : 8;
}

//! Round 7 (Track B): the CUTLASS 2.11 classic-SIMT schedule ported onto
//! the chunk-table loaders — the pristine reference measured 10.7 TF flat
//! (see r7-census.txt for the SASS census of the 'def' kernel).
//! Schedule: 128x128x8 tile, 256 threads, warp aspect 32x64 with the
//! CUTLASS SIMT lane grid (4 rows x 8 cols, from simt_get_warp_threads_m:
//! warp kM<kN -> 4 rows), 8x8 thread tile. Shared panels sit in the
//! CUTLASS canonical orientation — A m-contiguous (column-major),
//! B n-contiguous (row-major) — with their transpose padding
//! (simt_transpose_padding(32,8,32)=4) as the ONLY bank scheme: kLdA=BM+4
//! when A comes from row-major global (transposed scalar stores), kLdB=BN+4
//! when B comes from column-major global. No XOR swizzle: the skew alone
//! makes every STS and LDS.128 conflict-free for this access pattern
//! (census-verified). Pipeline: ONE barrier per k-panel (the single-barrier
//! double buffer: compute(q) reads stage q&1 while stores land in (q+1)&1;
//! the trailing barrier separates both directions). Global loads go
//! through the run-8 chunk discipline: threads 0..127 own one 8-run of A
//! per panel (a whole k-row slice for TA==0, an m-run for TA==1), threads
//! 128..255 own B's — one chunk resolution per 8 elements, the 16:1-style
//! amortization the round-6 library iterators could not reach.

//! Run-8 staging returned BY VALUE (register-resident; the round-7 local
//! memory lesson: a pointer destination forces the staging array into
//! local memory and every panel pays LDL/STL long-scoreboard stalls).
struct Run8
{
    float v[8];
};

//! Run-8 loader with the addressing-class split of load_run16: 8 valid,
//! 4-aligned, straddle-free runs take the float4x2 path (one resolved
//! base pointer, two LDG.128); everything else falls to the outlined
//! cold tail with per-element chunk resolution and zero fill.
__noinline__ __device__ Run8 load_run8_cold(const void *tab, int shift,
    long long elem0, int nvalid)
{
    Run8 r;
#pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        r.v[i] = i < nvalid ? *chunk_at(tab, shift, elem0 + i) : 0.f;
    }
    return r;
}

template <int AC>
__device__ __forceinline__ Run8 load_run8(const void *tab, int shift,
    long long elem0, int nvalid)
{
    if constexpr(AC == kAcFlat)
    {
        if(nvalid == 8 && (elem0 & 3) == 0)
        {
            const float *q0 = reinterpret_cast<const float *>(tab) + elem0;
            Run8 r;
            *reinterpret_cast<float4 *>(r.v)
                = *reinterpret_cast<const float4 *>(q0);
            *reinterpret_cast<float4 *>(r.v + 4)
                = *reinterpret_cast<const float4 *>(q0 + 4);
            return r;
        }
    }
    else
    {
        int const se = shift >= 62 ? 62 : shift;
        long long const span = 1LL << se;
        if(nvalid == 8 && (elem0 & 3) == 0
            && (elem0 & (span - 1)) < span - 7)
        {
            const float *q0
                = reinterpret_cast<const float *const *>(tab)[elem0 >> se]
                + (elem0 & (span - 1));
            Run8 r;
            *reinterpret_cast<float4 *>(r.v)
                = *reinterpret_cast<const float4 *>(q0);
            *reinterpret_cast<float4 *>(r.v + 4)
                = *reinterpret_cast<const float4 *>(q0 + 4);
            return r;
        }
    }
    return load_run8_cold(tab, shift, elem0, nvalid);
}

template <int TA, int TB, int ACA, int ACB, int BM, int BN, int BK,
    int THREADS, int MINB = 1>
__global__ __launch_bounds__(THREADS, MINB) void sgemm_chunked_kernel_cut(
    Params p)
{
    constexpr int kLRows = 4;
    constexpr int kLCols = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int kWarpM = kLRows * TM;      // 32
    constexpr int kWarpN = kLCols * TN;      // 64
    constexpr int kWRows = BM / kWarpM;      // 4
    constexpr int kWCols = BN / kWarpN;      // 2
    constexpr int kWarps = kWRows * kWCols;  // 8
    static_assert(kWarps * 32 == THREADS, "thread count mismatch");
    static_assert(BM % kWarpM == 0 && BN % kWarpN == 0, "tile split");
    //! Skewed leading dims for BOTH operands (132 floats). CUTLASS
    //! skews only the transposed-store side, but their loader maps
    //! consecutive threads to consecutive rows; OUR run-8 ownership
    //! strides a warp's threads 8 columns apart, which makes ld=128
    //! float4 stores 4-way conflicted. With ld=BM+4 every warp-wide
    //! STS.128 lands on 32 distinct banks ((33k + 2j + h) mod 32) and
    //! every scalar STS row-set stays spread; the LDS.128 fragment
    //! reads keep their broadcast groups either way (ncu-verified).
    constexpr int kLdA = BM + 4;
    constexpr int kLdB = BN + 4;
    constexpr int kStageA = BK * kLdA;
    constexpr int kStageB = BK * kLdB;
    constexpr int kHalf = THREADS / 2;       // 128 A-loader threads
    static_assert(BM * BK == kHalf * 8 && BK * BN == kHalf * 8,
        "panel must split into one run-8 per half-CTA thread");
    static_assert(BK == 8, "run-8 loader pairs with the 8-deep panel");

    extern __shared__ float smem[];
    float *As = smem;
    float *Bs = smem + 2 * kStageA;

    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const warp_row = warp / kWCols;
    int const warp_col = warp % kWCols;
    int const lane_row = lane & 3;
    int const lane_col = lane >> 2;
    int const thr_m0 = warp_row * kWarpM + lane_row * TM;
    int const thr_n0 = warp_col * kWarpN + lane_col * TN;

    float acc[TM][TN];
    //! Double staging buffer, panel q lives in buffer q&1. ONE array
    //! serves both operand sides: each thread owns either an A run or a
    //! B run, never both (halves the staging registers).
    float areg[2][8];
#pragma unroll
    for(int i = 0; i < TM; ++i)
    {
#pragma unroll
        for(int j = 0; j < TN; ++j)
        {
            acc[i][j] = 0.f;
        }
    }

    //! Load one operand's 8-run for panel `q` into the staging buffer.
    //! Ownership: tid<kHalf owns A's run, the rest own B's (the panel is
    //! exactly 128 run-8s per operand side). The destination copies use
    //! buf-branched CONSTANT indices so the staging arrays stay in
    //! registers (a dynamic index or an escaping pointer demotes them to
    //! local memory — the round-7 long-scoreboard lesson).
    auto ld = [&](int q, int buf, int m0, int n0, long long a_base,
                  long long b_base) {
        int const k0 = q * BK;
        Run8 r;
        if(tid < kHalf)
        {
            if(TA == 0)
            {
                // Whole 8-k slice of row m0+tid: one contiguous run.
                int const grow = m0 + tid;
                long long const elem0 = a_base
                    + static_cast<long long>(grow) * p.k + k0;
                r = load_run8<ACA>(p.a, p.a_shift, elem0,
                    nv8(grow < p.m, p.k - k0));
            }
            else
            {
                // 8-m run of k row k0 + (tid>>4).
                int const kk = tid >> 4;
                int const mm0 = (tid & 15) * 8;
                long long const elem0 = a_base
                    + static_cast<long long>(k0 + kk) * p.m + m0 + mm0;
                r = load_run8<ACA>(p.a, p.a_shift, elem0,
                    nv8(k0 + kk < p.k, p.m - (m0 + mm0)));
            }
        }
        else
        {
            int const t = tid - kHalf;
            if(TB == 0)
            {
                // 8-n run of k row k0 + (t>>4).
                int const kk = t >> 4;
                int const nn0 = (t & 15) * 8;
                long long const elem0 = b_base
                    + static_cast<long long>(k0 + kk) * p.n + n0 + nn0;
                r = load_run8<ACB>(p.b, p.b_shift, elem0,
                    nv8(k0 + kk < p.k, p.n - (n0 + nn0)));
            }
            else
            {
                // Whole 8-k slice of row n0+t.
                int const gn = n0 + t;
                long long const elem0 = b_base
                    + static_cast<long long>(gn) * p.k + k0;
                r = load_run8<ACB>(p.b, p.b_shift, elem0,
                    nv8(gn < p.n, p.k - k0));
            }
        }
        if(buf == 0)
        {
#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                areg[0][i] = r.v[i];
            }
        }
        else
        {
#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                areg[1][i] = r.v[i];
            }
        }
    };

    //! Store the staged run into panel q's smem stage. Transposed sides
    //! (TA==0 A, TB==1 B) store 8 scalars into the m/n-contiguous layout;
    //! the +4 skew spreads a warp's 32 consecutive rows across 32 banks
    //! (k*skew is a multiple of 4 floats). Direct sides store two float4s.
    auto st = [&](int q, int buf) {
        float *as = As + (q & 1) * kStageA;
        float *bs = Bs + (q & 1) * kStageB;
        if(tid < kHalf)
        {
            if(TA == 0)
            {
#pragma unroll
                for(int e = 0; e < 8; ++e)
                {
                    as[e * kLdA + tid]
                        = buf == 0 ? areg[0][e] : areg[1][e];
                }
            }
            else
            {
                int const kk = tid >> 4;
                int const mm0 = (tid & 15) * 8;
#pragma unroll
                for(int h = 0; h < 2; ++h)
                {
                    float const *src = buf == 0 ? areg[0] : areg[1];
                    *reinterpret_cast<float4 *>(as + kk * kLdA + mm0 + h * 4)
                        = make_float4(src[h * 4 + 0], src[h * 4 + 1],
                            src[h * 4 + 2], src[h * 4 + 3]);
                }
            }
        }
        else
        {
            int const t = tid - kHalf;
            if(TB == 0)
            {
                int const kk = t >> 4;
                int const nn0 = (t & 15) * 8;
#pragma unroll
                for(int h = 0; h < 2; ++h)
                {
                    float const *src = buf == 0 ? areg[0] : areg[1];
                    *reinterpret_cast<float4 *>(bs + kk * kLdB + nn0 + h * 4)
                        = make_float4(src[h * 4 + 0], src[h * 4 + 1],
                            src[h * 4 + 2], src[h * 4 + 3]);
                }
            }
            else
            {
#pragma unroll
                for(int e = 0; e < 8; ++e)
                {
                    bs[e * kLdB + t] = buf == 0 ? areg[0][e] : areg[1][e];
                }
            }
        }
    };

    //! Compute panel q: 8 k-steps, each 4 LDS.128 (A m0..m0+7 frag, B
    //! n0..n0+7 frag) with one-step-ahead fragment prefetch, 64 FFMA per
    //! step — the census shape (32 LDS.U.128 + 512 FFMA per panel).
    auto compute = [&](int q) {
        float const *as = As + (q & 1) * kStageA;
        float const *bs = Bs + (q & 1) * kStageB;
        float4 av4[2][2];
        float4 bv4[2][2];
#pragma unroll
        for(int h = 0; h < 2; ++h)
        {
            av4[0][h] = *reinterpret_cast<const float4 *>(as + thr_m0 + h * 4);
            bv4[0][h] = *reinterpret_cast<const float4 *>(bs + thr_n0 + h * 4);
        }
#pragma unroll
        for(int kk = 0; kk < BK; ++kk)
        {
            int const cur = kk & 1;
            int const nxt = (kk + 1) & 1;
            if(kk + 1 < BK)
            {
#pragma unroll
                for(int h = 0; h < 2; ++h)
                {
                    av4[nxt][h] = *reinterpret_cast<const float4 *>(
                        as + (kk + 1) * kLdA + thr_m0 + h * 4);
                    bv4[nxt][h] = *reinterpret_cast<const float4 *>(
                        bs + (kk + 1) * kLdB + thr_n0 + h * 4);
                }
            }
#pragma unroll
            for(int i = 0; i < 2; ++i)
            {
#pragma unroll
                for(int j = 0; j < 2; ++j)
                {
                    float4 const a = av4[cur][i];
                    float4 const b = bv4[cur][j];
                    float *acc0 = &acc[i * 4][j * 4];
                    acc0[0] += a.x * b.x;
                    acc0[1] += a.x * b.y;
                    acc0[2] += a.x * b.z;
                    acc0[3] += a.x * b.w;
                    acc0[TN + 0] += a.y * b.x;
                    acc0[TN + 1] += a.y * b.y;
                    acc0[TN + 2] += a.y * b.z;
                    acc0[TN + 3] += a.y * b.w;
                    acc0[2 * TN + 0] += a.z * b.x;
                    acc0[2 * TN + 1] += a.z * b.y;
                    acc0[2 * TN + 2] += a.z * b.z;
                    acc0[2 * TN + 3] += a.z * b.w;
                    acc0[3 * TN + 0] += a.w * b.x;
                    acc0[3 * TN + 1] += a.w * b.y;
                    acc0[3 * TN + 2] += a.w * b.z;
                    acc0[3 * TN + 3] += a.w * b.w;
                }
            }
        }
    };

    // Persistent tile loop (same discipline as the classic family).
    int const tiles_x = (p.n + BN - 1) / BN;
    int const tiles_y = (p.m + BM - 1) / BM;
    long long const ntiles = static_cast<long long>(tiles_x) * tiles_y
        * p.batch;
    for(long long tile = static_cast<long long>(blockIdx.x)
                       + static_cast<long long>(blockIdx.y) * gridDim.x;
        tile < ntiles;
        tile += static_cast<long long>(gridDim.x) * gridDim.y)
    {
        long long const rem = tile / p.batch;
        int const bi = static_cast<int>(tile % p.batch);
        int const m0 = static_cast<int>(rem % tiles_y) * BM;
        int const n0 = static_cast<int>(rem / tiles_y) * BN;
        long long const a_base = bi * p.sa;
        long long const b_base = bi * p.sb;
        long long const c_base = bi * p.sc;
        int const npanels = (p.k + BK - 1) / BK;

        // Single-barrier double buffer: compute(q) reads stage q&1 while
        // panel q+1's staged run lands in stage (q+1)&1; one barrier per
        // panel orders both directions (store->next compute, compute->
        // next store).
        ld(0, 0, m0, n0, a_base, b_base);
        st(0, 0);
        if(1 < npanels)
        {
            ld(1, 1, m0, n0, a_base, b_base);
        }
        __syncthreads();
        for(int q = 0; q < npanels; ++q)
        {
            compute(q);
            if(q + 1 < npanels)
            {
                st(q + 1, (q + 1) & 1);
            }
            if(q + 2 < npanels)
            {
                ld(q + 2, q & 1, m0, n0, a_base, b_base);
            }
            __syncthreads();
        }

        // ---- epilogue: the classic family's chunk-resolved C store ----
        bool const beta_zero = (p.beta == 0.f);
#pragma unroll
        for(int i = 0; i < TM; ++i)
        {
            int const gm = m0 + thr_m0 + i;
            if(gm >= p.m)
            {
                continue;
            }
#pragma unroll
            for(int j = 0; j < TN; j += 4)
            {
                int const gn = n0 + thr_n0 + j;
                if(gn >= p.n)
                {
                    continue;
                }
                long long const elem0 = c_base
                    + static_cast<long long>(gm) * p.n + gn;
                float v[4];
                v[0] = p.alpha * acc[i][j + 0];
                v[1] = p.alpha * acc[i][j + 1];
                v[2] = p.alpha * acc[i][j + 2];
                v[3] = p.alpha * acc[i][j + 3];
                int const span = gn + 3 < p.n ? 4 : p.n - gn;
                bool const same_chunk = p.c_shift == 0 || p.c_shift >= 62
                    || ((elem0 >> p.c_shift)
                        == ((elem0 + 3) >> p.c_shift));
                if(span == 4 && (elem0 & 3) == 0 && same_chunk)
                {
                    float4 *dst = reinterpret_cast<float4 *>(
                        chunk_at_mut(p.c, p.c_shift, elem0));
                    if(!beta_zero)
                    {
                        float4 const old = *dst;
                        v[0] += p.beta * old.x;
                        v[1] += p.beta * old.y;
                        v[2] += p.beta * old.z;
                        v[3] += p.beta * old.w;
                    }
                    *dst = make_float4(v[0], v[1], v[2], v[3]);
                }
                else
                {
                    store_c_tail(p.c, p.c_shift, elem0, span, beta_zero,
                        p.beta, make_float4(v[0], v[1], v[2], v[3]));
                }
            }
        }
#pragma unroll
        for(int i = 0; i < TM; ++i)
        {
#pragma unroll
            for(int j = 0; j < TN; ++j)
            {
                acc[i][j] = 0.f;
            }
        }
    }
}

//! Kernel configuration ids (the picker's currency).

//! 0: 8x8x32 over 256 threads (baseline)
//! 1: 8x8x32 over 128 threads on a 128x64 tile, single-buffered
//!    (the Triton shape: 24 KiB smem, 2 CTAs/SM, deep register
//!    prefetch with a full panel of latency runway)
//! 6: the 128x128x32 tile, 256 threads, ONE smem stage (32 KiB) and
//!    the unpipelined (PIPE=1) compute loop — Triton's literal shape
//!    under the register attack; bench-only so far.
constexpr int kCfgA88 = 0;
constexpr int kCfgT1 = 1;
constexpr int kCfgMsWgrad = 4;
constexpr int kCfgMsSmall = 5;
constexpr int kCfgS1 = 6;
constexpr int kCfgCutB = 7;

template <int CFG>
struct CfgTraits;

template <>
struct CfgTraits<kCfgA88>
{
    static constexpr int BM = 128, BN = 128, BK = 32, TM = 8, TN = 8,
        THREADS = 256, MINB = 1, STAGES = 2, PIPE = 2;
};

//! The Triton-shape configuration: 128x64x32 tile, 8x8 thread tile
//! over 128 threads (4 warps of 64x32), ONE smem stage (24 KiB) and
//! two CTAs per SM — 16 warps/SM hide each CTA's barrier and
//! shared-memory latencies, and the 256-register budget of
//! __launch_bounds__(128, 2) fits the 64 accumulators + 48 staging +
//! compute transients without spilling.
template <>
struct CfgTraits<kCfgT1>
{
    static constexpr int BM = 128, BN = 64, BK = 32, TM = 8, TN = 8,
        THREADS = 128, MINB = 2, STAGES = 1, PIPE = 2;
};

//! The single-buffer 32 KiB Triton literal shape on the full
//! 128x128x32 tile with 256 threads: global loads issue at loop top
//! into the staging registers, land in smem after the barrier, and
//! the compute loop runs PIPE=1 (no double-buffered LDS fragments).
//! The register attack's "does the lean body fly?" probe.
template <>
struct CfgTraits<kCfgS1>
{
    static constexpr int BM = 128, BN = 128, BK = 32, TM = 8, TN = 8,
        THREADS = 256, MINB = 1, STAGES = 1, PIPE = 1;
};




//! The round-7 CUTLASS-port configuration: 128x128x8 tile, warp aspect
//! 32x64, CUTLASS lane grid 4x8, single-barrier double buffer.
template <>
struct CfgTraits<kCfgCutB>
{
    static constexpr int BM = 128, BN = 128, BK = 8, THREADS = 256, MINB = 2;
};

//! Occupancy-driven grid-stride launcher for the CUTLASS-port kernel
//! (same launch discipline as the classic family).
template <int TA, int TB, int ACA, int ACB, int MINB>
bool launch_cut(Params &p, cudaStream_t stream)
{
    using T = CfgTraits<kCfgCutB>;
    constexpr int kLdA = T::BM + 4;
    constexpr int kLdB = T::BN + 4;
    constexpr int kSmem = 2 * (T::BK * kLdA + T::BK * kLdB)
        * int(sizeof(float));
    constexpr auto kKernel = sgemm_chunked_kernel_cut<TA, TB, ACA, ACB,
        T::BM, T::BN, T::BK, T::THREADS, MINB>;
    static bool configured = [] {
        bool ok = cudaFuncSetAttribute(kKernel,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     kSmem <= 48 * 1024 ? 48 * 1024 : kSmem) == cudaSuccess;
        // sm_75 carveout: the default L1/smem split caps the occupancy
        // calculator at one block; prefer the max-smem carveout so
        // 2-3 co-resident CTAs count (the kernel is small enough either
        // way, the calculator just needs the config).
        cudaFuncSetAttribute(kKernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            (int)cudaSharedmemCarveoutMaxShared);
        return ok;
    }();
    if(!configured)
    {
        return false;
    }
    static int blocks_per_sm = [] {
        int b = 0;
        cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b,
            reinterpret_cast<void const *>(kKernel), T::THREADS, kSmem);
        if(std::getenv("NNHAUL_DEBUG_OCC"))
        {
            cudaFuncAttributes at;
            cudaFuncGetAttributes(&at, kKernel);
            std::fprintf(stderr,
                "cutB occ: blocks=%d err=%s kSmem=%d regs=%d "
                "localBytes=%d staticSmem=%d maxThreads=%d\n",
                b, cudaGetErrorString(e), kSmem, at.numRegs,
                at.localSizeBytes, at.sharedSizeBytes, at.maxThreadsPerBlock);
        }
        return b > 0 ? b : 1;
    }();
    static int sm_count = [] {
        int d = 0;
        if(cudaGetDevice(&d) != cudaSuccess)
        {
            return 1;
        }
        int n = 0;
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, d) != cudaSuccess)
        {
            return 1;
        }
        n = prop.multiProcessorCount;
        return n > 0 ? n : 1;
    }();
    long long const tiles = static_cast<long long>(
        (p.n + T::BN - 1) / T::BN)
        * ((p.m + T::BM - 1) / T::BM) * p.batch;
    long long const want
        = static_cast<long long>(sm_count) * blocks_per_sm;
    int const grid_x = static_cast<int>(tiles < want ? tiles : want);
    void *args[] = {&p};
    auto err = cudaLaunchKernel(reinterpret_cast<void const *>(kKernel),
        dim3(grid_x, 1, 1), dim3(T::THREADS), args, kSmem, stream);
    return err == cudaSuccess;
}

template <int TA, int TB, int MINB = CfgTraits<kCfgCutB>::MINB>
bool launch_cut_ac(Params &p, cudaStream_t stream)
{
    int const aca = p.a_shift == 0 ? kAcFlat : kAcTable;
    int const acb = p.b_shift == 0 ? kAcFlat : kAcTable;
    if(aca == kAcFlat && acb == kAcFlat)
    {
        return launch_cut<TA, TB, kAcFlat, kAcFlat, MINB>(p, stream);
    }
    if(aca == kAcFlat)
    {
        return launch_cut<TA, TB, kAcFlat, kAcTable, MINB>(p, stream);
    }
    if(acb == kAcFlat)
    {
        return launch_cut<TA, TB, kAcTable, kAcFlat, MINB>(p, stream);
    }
    return launch_cut<TA, TB, kAcTable, kAcTable, MINB>(p, stream);
}

template <>
struct CfgTraits<kCfgMsWgrad>
{
    static constexpr int BM = 128, BN = 64, BK = 16, TM = 8, TN = 4,
        THREADS = 256, RUN = 4, STAGES = 2, MINB = 2;
};
template <>
struct CfgTraits<kCfgMsSmall>
{
    static constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 8,
        THREADS = 128, RUN = 8, STAGES = 2, MINB = 4;
};

template <int TA, int TB, int ACA, int ACB, int CFG>
constexpr auto kKernelOf = sgemm_chunked_kernel<TA, TB, ACA, ACB,
    CfgTraits<CFG>::BM, CfgTraits<CFG>::BN, CfgTraits<CFG>::BK,
    CfgTraits<CFG>::TM, CfgTraits<CFG>::TN, CfgTraits<CFG>::THREADS,
    CfgTraits<CFG>::MINB, CfgTraits<CFG>::STAGES, CfgTraits<CFG>::PIPE>;

template <int TA, int TB, int CFG>
bool launch_cfg_ms(Params &p, cudaStream_t stream)
{
    using T = CfgTraits<CFG>;
    constexpr int kSmem = T::STAGES
        * (T::BM * T::BK + T::BK * T::BN) * int(sizeof(float));
    dim3 const grid((p.n + T::BN - 1) / T::BN, (p.m + T::BM - 1) / T::BM,
        p.batch);
    auto kernel = sgemm_chunked_kernel_ms<TA, TB, T::BM, T::BN, T::BK,
        T::TM, T::TN, T::THREADS, T::RUN, T::STAGES, T::MINB>;
    static bool configured = [] {
        return cudaFuncSetAttribute(
            sgemm_chunked_kernel_ms<TA, TB, T::BM, T::BN, T::BK, T::TM,
                T::TN, T::THREADS, T::RUN, T::STAGES, T::MINB>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSmem <= 48 * 1024 ? 48 * 1024 : kSmem) == cudaSuccess;
    }();
    if(!configured)
    {
        return false;
    }
    void *args[] = {&p};
    auto err = cudaLaunchKernel(
        reinterpret_cast<void const *>(kernel), grid,
        dim3(T::THREADS), args, kSmem, stream);
    return err == cudaSuccess;
}

template <int TA, int TB, int ACA, int ACB, int CFG>
bool launch_cfg(Params &p, cudaStream_t stream)
{
    using T = CfgTraits<CFG>;
    constexpr int kSmem
        = T::STAGES * (T::BM * T::BK + T::BK * T::BN) * int(sizeof(float));
    constexpr auto kernel = kKernelOf<TA, TB, ACA, ACB, CFG>;
    static bool configured = [] {
        return cudaFuncSetAttribute(
            sgemm_chunked_kernel<TA, TB, ACA, ACB, T::BM, T::BN, T::BK,
                T::TM, T::TN, T::THREADS, T::MINB, T::STAGES, T::PIPE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSmem <= 48 * 1024 ? 48 * 1024 : kSmem) == cudaSuccess;
    }();
    if(!configured)
    {
        return false;
    }
    // Occupancy-driven grid: enough blocks to fill every slot the SMs
    // can co-reside, one linear launch; the kernel grid-strides over
    // the tile space (a single iteration when tiles <= blocks).
    static int blocks_per_sm = [] {
        int b = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b,
            reinterpret_cast<void const *>(kernel), T::THREADS, kSmem);
        return b > 0 ? b : 1;
    }();
    static int sm_count = [] {
        int d = 0;
        if(cudaGetDevice(&d) != cudaSuccess)
        {
            return 1;
        }
        int n = 0;
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, d) != cudaSuccess)
        {
            return 1;
        }
        n = prop.multiProcessorCount;
        return n > 0 ? n : 1;
    }();
    long long const tiles = static_cast<long long>(
        (p.n + T::BN - 1) / T::BN)
        * ((p.m + T::BM - 1) / T::BM) * p.batch;
    long long const want
        = static_cast<long long>(sm_count) * blocks_per_sm;
    int const grid_x = static_cast<int>(tiles < want ? tiles : want);
    void *args[] = {&p};
    auto err = cudaLaunchKernel(
        reinterpret_cast<void const *>(kernel), dim3(grid_x, 1, 1),
        dim3(T::THREADS), args, kSmem, stream);
    return err == cudaSuccess;
}

//! Addressing-class front end for the classic launcher: flat operands
//! (shift == 0) and table operands (any other shift) hit separately
//! compiled hot bodies — the flat loader is one pointer add, the table
//! loader one indirection plus the hoisted straddle compare.
template <int TA, int TB, int CFG>
bool launch_cfg_ac(Params &p, cudaStream_t stream)
{
    int const aca = p.a_shift == 0 ? kAcFlat : kAcTable;
    int const acb = p.b_shift == 0 ? kAcFlat : kAcTable;
    if(aca == kAcFlat && acb == kAcFlat)
    {
        return launch_cfg<TA, TB, kAcFlat, kAcFlat, CFG>(p, stream);
    }
    if(aca == kAcFlat)
    {
        return launch_cfg<TA, TB, kAcFlat, kAcTable, CFG>(p, stream);
    }
    if(acb == kAcFlat)
    {
        return launch_cfg<TA, TB, kAcTable, kAcFlat, CFG>(p, stream);
    }
    return launch_cfg<TA, TB, kAcTable, kAcTable, CFG>(p, stream);
}

//! One dispatch chain per config id; the multi-stage configs route to
//! their own launcher, the classic family to launch_cfg_ac.
template <int CFG, bool MS>
struct CfgDispatch;

template <int CFG>
struct CfgDispatch<CFG, false>
{
    static bool run(int ta, int tb, Params &p, cudaStream_t stream)
    {
        if(ta == 0 && tb == 0)
        {
            return launch_cfg_ac<0, 0, CFG>(p, stream);
        }
        if(ta == 0 && tb == 1)
        {
            return launch_cfg_ac<0, 1, CFG>(p, stream);
        }
        if(ta == 1 && tb == 0)
        {
            return launch_cfg_ac<1, 0, CFG>(p, stream);
        }
        return launch_cfg_ac<1, 1, CFG>(p, stream);
    }
};

template <int CFG>
struct CfgDispatch<CFG, true>
{
    static bool run(int ta, int tb, Params &p, cudaStream_t stream)
    {
        if(ta == 0 && tb == 0)
        {
            return launch_cfg_ms<0, 0, CFG>(p, stream);
        }
        if(ta == 0 && tb == 1)
        {
            return launch_cfg_ms<0, 1, CFG>(p, stream);
        }
        if(ta == 1 && tb == 0)
        {
            return launch_cfg_ms<1, 0, CFG>(p, stream);
        }
        return launch_cfg_ms<1, 1, CFG>(p, stream);
    }
};

template <int CFG>
bool dispatch_cfg(int ta, int tb, Params &p, cudaStream_t stream)
{
    constexpr bool kMs = CFG == kCfgMsWgrad || CFG == kCfgMsSmall;
    return CfgDispatch<CFG, kMs>::run(ta, tb, p, stream);
}

template <int CFG>
bool dispatch_cfg_ms(int ta, int tb, Params &p, cudaStream_t stream)
{
    if(ta == 0 && tb == 0)
    {
        return launch_cfg_ms<0, 0, CFG>(p, stream);
    }
    if(ta == 0 && tb == 1)
    {
        return launch_cfg_ms<0, 1, CFG>(p, stream);
    }
    if(ta == 1 && tb == 0)
    {
        return launch_cfg_ms<1, 0, CFG>(p, stream);
    }
    return launch_cfg_ms<1, 1, CFG>(p, stream);
}


} // namespace

bool supports(Op transA, Op transB, int m, int n, int k, int batch) noexcept
{
    (void)transA;
    (void)transB;
    // The tile-multiple law (owner directives, rounds 5 and 7): the
    // native family OWNS a call only when every shape dimension except
    // batch is a multiple of the OWNING config's tile -- 128 for the
    // large-shape classic family (S1/wgrad/CutB), 64 for the small-tile
    // batched family (attention). Tile-multiplicity subsumes the old
    // 4-alignment law; tails of every kind disappear from owned calls,
    // and by the intra-chunk law every owned row run is 16B-aligned.
    // Everything else (odd vocab widths, ragged shapes) is a BAD shape:
    // the workspace-staged cuBLAS path owns it.
    if(m <= 0 || n <= 0 || k <= 0 || batch <= 0)
    {
        return false;
    }
    if(batch >= 2 && n <= 64)
    {
        return m % 64 == 0 && n % 64 == 0 && k % 64 == 0;
    }
    return m % 128 == 0 && n % 128 == 0 && k % 128 == 0;
}

namespace
{

//! Tuning/diagnostic hook: NNHAUL_GEMM_CHUNKED_CFG pins one
//! configuration id (the campaign's A/B discipline; read once per
//! process like the ring's hooks). -1 (default) lets the picker choose.
int forced_cfg() noexcept
{
    static int const cached = [] {
        char const *e = std::getenv("NNHAUL_GEMM_CHUNKED_CFG");
        return e == nullptr ? -1 : std::atoi(e);
    }();
    return cached;
}

//! The shape-driven picker. v1 policy: the 8x16 config owns every
//! supported shape (half the shared-memory traffic per FMA of the 8x8
//! baseline); the thin-panel 8x16x16 variant wins on small-k shapes
//! where the 32-deep k panel cannot amortize its pipeline. Specialized
//! families (batched-small, odd-N lm_head) slot in here as they land.
//! The measured ownership policy (GPT-2 training set, 2080 Ti pinned
//! at 1350 MHz, scattered chunks of the default 4 MiB class). The
//! native kernel OWNS a call
//! only where it beats the production alternative — the workspace-
//! staged cuBLAS path (cuBLAS plus gather/scatter traffic): the
//! batched small-n family (attention projections and their grads),
//! where staging costs more than the whole GEMM. Everywhere else the
//! campaign has not yet closed the gap to the cuBLAS bar, the picker
//! returns -1 and the staged path stays in charge. The
//! NNHAUL_GEMM_CHUNKED_CFG pin overrides the policy for measurement.
//! The measured ownership policy (GPT-2 training set, 2080 Ti pinned
//! at 1350 MHz, scattered chunks of the default 4 MiB class). The
//! native kernel OWNS a call
//! only where it beats the production workspace-staged cuBLAS path
//! (cuBLAS plus gather/scatter traffic): the batched small-n family
//! (attention projections and their grads), where staging costs more
//! than the whole GEMM. Everywhere else the campaign has not yet
//! closed the gap to the cuBLAS bar, the picker returns -1 and the
//! staged path stays in charge. The NNHAUL_GEMM_CHUNKED_CFG pin
//! overrides the policy for measurement.
int pick_cfg(Op, Op, int n, int, int, int batch) noexcept
{
    if(batch >= 2 && n <= 64)
    {
        return kCfgMsSmall;
    }
    return -1;
}

} // namespace

const char *pick_name(Op transA, Op transB, int m, int n, int k,
    int batch) noexcept
{
    if(!supports(transA, transB, m, n, k, batch))
    {
        return "";
    }
    int const cfg = forced_cfg() >= 0 ? forced_cfg()
                                      : pick_cfg(transA, transB, m, n, k,
                                            batch);
    switch(cfg)
    {
        case kCfgA88:
            return "chunked-128x128x32-8x8-256t";
        case kCfgT1:
            return "chunked-128x64x32-s1-8x8-128t";
        case kCfgMsWgrad:
            return "chunked-ms-128x64x16-s2-8x4-256t";
        case kCfgMsSmall:
            return "chunked-ms-64x64x16-s2-8x8-128t";
        case kCfgS1:
            return "chunked-128x128x32-s1-p1-8x8-256t";
        case kCfgCutB:
            return "chunked-cut-128x128x8-32x64w-256t";
    }
    return "chunked-?";
}

bool launch_with_cfg(int cfg, Op transA, Op transB, Params &p,
    cudaStream_t stream) noexcept;

bool launch(Op transA, Op transB, int m, int n, int k, int batch,
    float alpha, const Operand &a, const Operand &b, float beta,
    const Operand &c, long long stride_a, long long stride_b,
    long long stride_c, cudaStream_t stream) noexcept
{
    if(!supports(transA, transB, m, n, k, batch))
    {
        return false;
    }
    Params p;
    p.a = a.ptr;
    p.a_shift = a.shift;
    p.b = b.ptr;
    p.b_shift = b.shift;
    p.c = const_cast<void *>(c.ptr);
    p.c_shift = c.shift;
    p.m = m;
    p.n = n;
    p.k = k;
    p.batch = batch;
    p.sa = stride_a;
    p.sb = stride_b;
    p.sc = stride_c;
    p.alpha = alpha;
    p.beta = beta;
    int const picked = pick_cfg(transA, transB, m, n, k, batch);
    int const cfg = forced_cfg() >= 0 ? forced_cfg() : picked;
    if(cfg < 0)
    {
        return false;
    }
    return launch_with_cfg(cfg, transA, transB, p, stream);
}

//! The best native configuration for a shape the ownership policy
//! declined — launch_forced()'s choice.
int pick_cfg_best(Op ta, Op tb, int m, int n, int k, int batch) noexcept
{
    (void)m;
    (void)n;
    if(batch >= 2 && n <= 64)
    {
        return kCfgMsSmall;
    }
    // Round 7 (the CUTLASS-schedule port, 128x128x8 / warp 32x64 /
    // 2-stage single-barrier, 128 regs = 2 CTAs/SM): new best on the
    // large NN family (7.8-8.4 TF scattered) and on qkv_wgrad (7.07);
    // ties MsWgrad on the mlp wgrads. NT stays with round 4's S1
    // (CutB's both-sides-transposed staging loses there: 7.0-7.3 vs
    // 7.6-8.1).
    if(ta == Op::NoTrans && tb == Op::NoTrans)
    {
        return kCfgCutB;
    }
    if(ta == Op::Trans && k >= 1024)
    {
        return kCfgCutB;
    }
    return kCfgS1;
}



//! Launch with the best configuration for the shape REGARDLESS of the
//! ownership policy — the workspace-overflow rescue (a call whose
//! staged bytes exceed the gemm workspace must run natively; there is
//! no third path). Same contract as launch() otherwise.
bool launch_forced(Op transA, Op transB, int m, int n, int k, int batch,
    float alpha, const Operand &a, const Operand &b, float beta,
    const Operand &c, long long stride_a, long long stride_b,
    long long stride_c, cudaStream_t stream) noexcept
{
    if(!supports(transA, transB, m, n, k, batch))
    {
        return false;
    }
    Params p;
    p.a = a.ptr;
    p.a_shift = a.shift;
    p.b = b.ptr;
    p.b_shift = b.shift;
    p.c = const_cast<void *>(c.ptr);
    p.c_shift = c.shift;
    p.m = m;
    p.n = n;
    p.k = k;
    p.batch = batch;
    p.sa = stride_a;
    p.sb = stride_b;
    p.sc = stride_c;
    p.alpha = alpha;
    p.beta = beta;
    int const cfg = pick_cfg_best(transA, transB, m, n, k, batch);
    return launch_with_cfg(cfg, transA, transB, p, stream);
}

//! Launch with an explicit configuration id — the campaign pin and the
//! rescue path above.
bool launch_with_cfg(int cfg, Op transA, Op transB, Params &p,
    cudaStream_t stream) noexcept
{
    int const ta = static_cast<int>(transA);
    int const tb = static_cast<int>(transB);
    switch(cfg)
    {
        case kCfgA88:
            return dispatch_cfg<kCfgA88>(ta, tb, p, stream);
        case kCfgT1:
            return dispatch_cfg<kCfgT1>(ta, tb, p, stream);
        case kCfgMsWgrad:
            return dispatch_cfg<kCfgMsWgrad>(ta, tb, p, stream);
        case kCfgMsSmall:
            return dispatch_cfg<kCfgMsSmall>(ta, tb, p, stream);
        case kCfgS1:
            return dispatch_cfg<kCfgS1>(ta, tb, p, stream);
        case kCfgCutB:
        {
            if(ta == 0 && tb == 0)
            {
                return launch_cut_ac<0, 0>(p, stream);
            }
            if(ta == 0 && tb == 1)
            {
                return launch_cut_ac<0, 1>(p, stream);
            }
            if(ta == 1 && tb == 0)
            {
                return launch_cut_ac<1, 0>(p, stream);
            }
            return launch_cut_ac<1, 1>(p, stream);
        }
    }
    return dispatch_cfg<kCfgA88>(ta, tb, p, stream);
}

} // namespace nntile::kernel::gemm_chunked

#endif // NNTILE_USE_CUDA
