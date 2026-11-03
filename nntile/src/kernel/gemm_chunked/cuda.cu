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

// Corresponding header
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

//! Load one 16-element run starting at absolute element index `elem0`
//! into `dst`, zero-filling elements rejected by `ok(i)`. Vector path
//! only when every element is valid AND the run stays inside one chunk
//! at a 16B-aligned offset; otherwise each element resolves its own
//! chunk (straddles, tails, misalignment) — never an out-of-table
//! index, since invalid elements are not resolved at all.
struct Run16
{
    float4 v[4];
};

template <class Ok>
__device__ __forceinline__ Run16 load_run16(const void *tab, int shift,
    long long elem0, Ok ok)
{
    Run16 r;
    bool all = true;
#pragma unroll
    for(int i = 0; i < 16; ++i)
    {
        all = all && ok(i);
    }
    if(all)
    {
        long long split = 16;
        if(shift > 0 && shift < 62)
        {
            long long const b = ((elem0 >> shift) + 1) << shift;
            long long const s = b - elem0;
            split = s < 16 ? s : 16;
        }
        if(split == 16 && (elem0 & 3) == 0)
        {
            const float *q0 = chunk_at(tab, shift, elem0);
#pragma unroll
            for(int q = 0; q < 4; ++q)
            {
                r.v[q] = *reinterpret_cast<const float4 *>(q0 + q * 4);
            }
        }
        else
        {
#pragma unroll
            for(int i = 0; i < 16; ++i)
            {
                (&r.v[i >> 2].x)[i & 3] = *chunk_at(tab, shift, elem0 + i);
            }
        }
    }
    else
    {
#pragma unroll
        for(int i = 0; i < 16; ++i)
        {
            (&r.v[i >> 2].x)[i & 3]
                = ok(i) ? *chunk_at(tab, shift, elem0 + i) : 0.f;
        }
    }
    return r;
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
template <int TA, int TB, int BM, int BN, int BK, int TM, int TN,
    int THREADS>
__global__ __launch_bounds__(THREADS) void sgemm_chunked_kernel(Params p)
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
    static_assert(kRunsA >= 1 && kRunsB >= 1, "loader under-subscribed");

    extern __shared__ float smem[];
    float *As = smem;
    float *Bs = smem + 2 * BM * BK;

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

    float areg[kRunsA][16];
    float breg[kRunsB][16];

    auto load_panel = [&](int panel) {
        int const k0 = panel * BK;
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
                Run16 const run = load_run16(p.a, p.a_shift, elem0,
                    [&](int i) {
                        return grow < p.m && k0 + kk0 + i < p.k;
                    });
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
                Run16 const run = load_run16(p.a, p.a_shift, elem0,
                    [&](int i) {
                        return gk < p.k && m0 + mm0 + i < p.m;
                    });
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
                Run16 const run = load_run16(p.b, p.b_shift, elem0,
                    [&](int i) {
                        return gk < p.k && n0 + nn0 + i < p.n;
                    });
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
                Run16 const run = load_run16(p.b, p.b_shift, elem0,
                    [&](int i) {
                        return gn < p.n && k0 + kk0 + i < p.k;
                    });
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

    auto store_panel = [&](int stage) {
        float *as = As + stage * BM * BK;
        float *bs = Bs + stage * BK * BN;
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

    load_panel(0);
    for(int panel = 0; panel < npanels; ++panel)
    {
        int const stage = panel & 1;
        store_panel(stage);
        if(panel + 1 < npanels)
        {
            load_panel(panel + 1);
        }
        __syncthreads();
        float const *as = As + stage * BM * BK;
        float const *bs = Bs + stage * BK * BN;
        // Vectorized FFMA loop, k in steps of 4: the swizzle keeps
        // every 4-group contiguous along exactly one axis per layout
        // (NoTrans A along k, Trans A along m, NoTrans B along n,
        // Trans B along k), so LDS.128 replaces four LDS.32 slots.
#pragma unroll 4
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
                    for(int g = 0; g < TM; g += 8)
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
        __syncthreads();
    }

    // ---- epilogue: alpha/beta and the chunk-resolved C store. The C
    // row may straddle chunk boundaries (odd n guarantees it), so every
    // 4-element group resolves its own base exactly like the loaders;
    // the vectorized store only fires when the whole group lies in one
    // chunk at a 16B-aligned offset. ----
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
            // Group fully in bounds, 16B-aligned, one chunk? (the
            // alignment law holds in EVERY addressing mode; odd n
            // simply takes the scalar path more often)
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

//! Kernel configuration ids (the picker's currency).
//! 0: 8x8x32 over 256 threads (baseline)
//! 1: 8x16x32 over 128 threads (half the smem traffic per FMA)
//! 2: 8x16x16 over 128 threads (thin panels, two blocks per SM)
constexpr int kCfgA88 = 0;
constexpr int kCfgB816 = 1;
constexpr int kCfgB816k16 = 2;
constexpr int kCfgCount = 3;

template <int CFG>
struct CfgTraits;

template <>
struct CfgTraits<kCfgA88>
{
    static constexpr int BM = 128, BN = 128, BK = 32, TM = 8, TN = 8,
        THREADS = 256;
};

template <>
struct CfgTraits<kCfgB816>
{
    static constexpr int BM = 128, BN = 128, BK = 32, TM = 8, TN = 16,
        THREADS = 128;
};

template <>
struct CfgTraits<kCfgB816k16>
{
    static constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 16,
        THREADS = 128;
};

template <int TA, int TB, int CFG>
bool launch_cfg(Params &p, cudaStream_t stream)
{
    using T = CfgTraits<CFG>;
    constexpr int kSmem
        = 2 * (T::BM * T::BK + T::BK * T::BN) * int(sizeof(float));
    dim3 const grid((p.n + T::BN - 1) / T::BN, (p.m + T::BM - 1) / T::BM,
        p.batch);
    auto kernel = sgemm_chunked_kernel<TA, TB, T::BM, T::BN, T::BK, T::TM,
        T::TN, T::THREADS>;
    static bool configured = [] {
        return cudaFuncSetAttribute(
            sgemm_chunked_kernel<TA, TB, T::BM, T::BN, T::BK, T::TM,
                T::TN, T::THREADS>,
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

} // namespace

bool supports(Op transA, Op transB, int m, int n, int k, int batch) noexcept
{
    if(m <= 0 || n <= 0 || k <= 0 || batch <= 0)
    {
        return false;
    }
    // Vectorized loader paths need the CONTIGUOUS leading dimension of
    // each storage layout to be a multiple of 4 (chunk bases are 4
    // MiB-aligned, so every run start is then 16B-aligned; straddling
    // runs resolve element-wise and stay correct). C always has a
    // guarded scalar fallback, so n itself may be arbitrary (lm_head's
    // 50257 vocab).
    if(transA == Op::NoTrans && k % 4 != 0)
    {
        return false;
    }
    if(transA == Op::Trans && m % 4 != 0)
    {
        return false;
    }
    if(transB == Op::NoTrans && n % 4 != 0)
    {
        return false;
    }
    if(transB == Op::Trans && k % 4 != 0)
    {
        return false;
    }
    return true;
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
int pick_cfg(Op, Op, int, int, int k, int) noexcept
{
    if(k < 32)
    {
        return kCfgB816k16;
    }
    return kCfgB816;
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
        case kCfgB816:
            return "chunked-128x128x32-8x16-128t";
        case kCfgB816k16:
            return "chunked-128x128x16-8x16-128t";
    }
    return "chunked-?";
}

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
    int const cfg = forced_cfg() >= 0 ? forced_cfg()
                                      : pick_cfg(transA, transB, m, n, k,
                                            batch);
    int const ta = static_cast<int>(transA);
    int const tb = static_cast<int>(transB);
    if(cfg == kCfgA88)
    {
        if(ta == 0 && tb == 0)
        {
            return launch_cfg<0, 0, kCfgA88>(p, stream);
        }
        if(ta == 0 && tb == 1)
        {
            return launch_cfg<0, 1, kCfgA88>(p, stream);
        }
        if(ta == 1 && tb == 0)
        {
            return launch_cfg<1, 0, kCfgA88>(p, stream);
        }
        return launch_cfg<1, 1, kCfgA88>(p, stream);
    }
    if(cfg == kCfgB816k16)
    {
        if(ta == 0 && tb == 0)
        {
            return launch_cfg<0, 0, kCfgB816k16>(p, stream);
        }
        if(ta == 0 && tb == 1)
        {
            return launch_cfg<0, 1, kCfgB816k16>(p, stream);
        }
        if(ta == 1 && tb == 0)
        {
            return launch_cfg<1, 0, kCfgB816k16>(p, stream);
        }
        return launch_cfg<1, 1, kCfgB816k16>(p, stream);
    }
    if(ta == 0 && tb == 0)
    {
        return launch_cfg<0, 0, kCfgB816>(p, stream);
    }
    if(ta == 0 && tb == 1)
    {
        return launch_cfg<0, 1, kCfgB816>(p, stream);
    }
    if(ta == 1 && tb == 0)
    {
        return launch_cfg<1, 0, kCfgB816>(p, stream);
    }
    return launch_cfg<1, 1, kCfgB816>(p, stream);
}

} // namespace nntile::kernel::gemm_chunked

#endif // NNTILE_USE_CUDA
