/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm_chunked/ext/large1024_kernels.cuh
 * The kernels of the large1024 extension piece: a persistent fp32 SIMT GEMM
 * on chunk-table operands with a load-balanced k-split tail. Correct on
 * every transpose combination and chunk class; not routed by the piece's
 * picker yet (about 1.3x cuBLAS on the piece's shapes: the address and
 * load phase at the top of every panel costs 16 to 50 percent on top of a
 * FFMA stream that alone reaches 96 percent of the peak).
 *
 * Tile and thread map (fixed): threadblock 128 x 128 x 8 (k-panel 8), 256
 * threads, 8 warps on a 4 (m) x 2 (n) grid; a warp owns 32 x 64 as 4 x 8
 * lanes, a thread 8 rows x 8 columns. A thread's 8 columns are INTERLEAVED
 * quads (32 h + 4 j + c), so its output stores and its B loads are 16-byte
 * accesses whose 8 lanes cover 128 contiguous bytes.
 *
 * Pipeline (3 shared stages of at most (129 + 129) x 8 floats, 24.3 KiB):
 *   - the global loads of the panels after the ones in flight are issued at
 *     the top of a panel iteration and land in registers; they go to the
 *     stage the barrier at the end of the iteration frees (a panel or more
 *     of latency cover);
 *   - the first k-group of the next panel is read from shared memory during
 *     the last k-group of the current one, so the warp resumes FFMA right
 *     after the barrier with nothing to wait for;
 *   - one __syncthreads() per panel.
 * Loaders (a warp's shared stores always hit 32 distinct banks):
 *   - k-contiguous operand: FOUR panels per load, 8 lanes per row (128
 *     contiguous bytes per quarter-warp); a lane holds four quads of one
 *     panel of the chunk, rows 32 apart; shared as k x m (n) with leading
 *     dimension 129, scalar stores, scalar reads in the mainloop;
 *   - m/n-contiguous operand: one panel per load, 32 lanes per row (512
 *     contiguous bytes); shared as k x m (n) with leading dimension 128,
 *     16-byte stores and reads.
 * Chunk tables are not read per load: a lane caches the chunk entry of
 * each of its rows for the whole job (a k-contiguous row changes chunk
 * once per 2^shift / 8 panels) and reads the table again only when its
 * index changes (m/n-contiguous side) or, for the rare row that crosses a
 * chunk inside the job, at every load.
 *
 * Epilogue: straight from registers, 16-byte stores through the C table.
 *
 * Schedule (MODE 1): one CTA per SM. The first `whole` tiles (a multiple of
 * the CTA count) are computed whole by consecutive CTAs; the remaining
 * `rem` tiles form a tile-major flattened space of rem * P panels cut into
 * one contiguous range per CTA. A range that straddles two tiles becomes two
 * parts. Every part is written as a raw partial into its own workspace slot
 * (2 * CTA + part) and a second kernel sums the parts of each split tile in
 * ascending slot order (= ascending k), then applies alpha and beta. Fixed
 * shape-derived partition, fixed-order reduction, no atomics.
 *
 * @version 1.0.0
 * */

#pragma once

// Standard libraries
#include <cstdint>

// Third-party headers
#include <cuda_runtime.h>

namespace nntile::kernel::gemm_chunked::ext::large1024
{

//! One call's kernel parameters (shared by the GEMM and the fixup kernel).
struct Params
{
    const void *a; //!< A base or chunk table
    const void *b; //!< B base or chunk table
    void *c;       //!< C base or chunk table
    float *ws;     //!< partials: slot * 16384 floats
    int sa;        //!< A chunk shift (0 flat, else log2 chunk floats, <= 62)
    int sb;        //!< B chunk shift
    int sc;        //!< C chunk shift
    int lda;       //!< A leading dimension (elements)
    int ldb;       //!< B leading dimension
    int ldc;       //!< C leading dimension
    int mt;        //!< tiles along m
    int nt;        //!< tiles in total (m-major)
    int P;         //!< k panels per tile (k / 8)
    int whole;     //!< tiles computed whole by one CTA each
    int wpc;       //!< whole tiles per CTA
    int rem;       //!< tiles shared through the k-split ranges (0: none)
    int np;        //!< k-split ranges (= CTA count of the GEMM kernel)
    float alpha;
    float beta;
};

//! Address of C's element `elem`: flat base or through the chunk table
//! (the flatness of C is a runtime shift).
__device__ __forceinline__ float *at_c(
    const void *tab, long long elem, long long mask, int sh) noexcept
{
    void *base = const_cast<void *>(tab);
    if (sh == 0)
    {
        return static_cast<float *>(base) + elem;
    }
    return static_cast<float *const *>(base)[elem >> sh] + (elem & mask);
}

//! Loader lane of a k-contiguous operand (four panels per load): the lane
//! holds four quads of ONE panel of the chunk, rows 32 apart. Fast path
//! (every row stays in one chunk over the job): the chunk base and the
//! in-chunk offset of each row; else the element index and the table.
template <int AC> struct XLane
{
    const float *base[4];
    long long elem[4];
    int off[4];
    const float *const *tab;
    long long mask;
    int sh;
    bool fast;

    //! `row0`: global index of the lane's first row, `e0`: element index of
    //! its first quad, `npan`: panels of the job.
    __device__ __forceinline__ void init(const void *table,
        int shift,
        long long row0,
        long long ld,
        long long e0,
        int npan) noexcept
    {
        tab = static_cast<const float *const *>(table);
        sh = shift;
        mask = (1LL << sh) - 1;
        fast = true;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            long long e = (row0 + 32LL * i) * ld + e0;
            elem[i] = e;
            if constexpr (AC == 1)
            {
                long long i0 = e >> sh;
                base[i] = tab[i0];
                off[i] = int(e & mask);
                if (((e + 8LL * (npan - 1) + 3) >> sh) != i0)
                {
                    fast = false; //!< a row crosses a chunk inside the job
                }
            }
        }
    }

    //! Address of the lane's quad of row i in the chunk `chunk` (32 floats
    //! of k per chunk).
    __device__ __forceinline__ const float *at(int i, int chunk) const noexcept
    {
        if constexpr (AC == 0)
        {
            return reinterpret_cast<const float *>(tab) + elem[i] +
                   32LL * chunk;
        }
        else if (fast)
        {
            return base[i] + off[i] + 32 * chunk;
        }
        long long e = elem[i] + 32LL * chunk;
        return tab[e >> sh] + (e & mask);
    }
};

//! Loader lane of an m/n-contiguous operand (one panel per load): the
//! lane's chunk entry is cached and read again when its index changes.
template <int AC> struct YLane
{
    const float *base;
    long long elem;
    long long step;
    long long idx;
    const float *const *tab;
    long long mask;
    int sh;

    __device__ __forceinline__ void init(
        const void *table, int shift, long long e0, long long st) noexcept
    {
        tab = static_cast<const float *const *>(table);
        base = nullptr;
        elem = e0;
        step = st;
        idx = -1;
        mask = (1LL << shift) - 1;
        sh = shift;
    }

    //! Address of the lane's quad of the current panel (the whole warp
    //! calls), then the next panel.
    __device__ __forceinline__ const float *at() noexcept
    {
        if constexpr (AC == 0)
        {
            return reinterpret_cast<const float *>(tab) + elem;
        }
        long long i = elem >> sh;
        if (__any_sync(0xffffffffu, i != idx))
        {
            if (i != idx)
            {
                base = tab[i];
                idx = i;
            }
        }
        return base + (elem & mask);
    }
};

//! acc[r][c] += a[r] * b[c]: 64 FFMA.
__device__ __forceinline__ void ffma(
    const float (&a)[8], const float (&b)[8], float (&acc)[8][8]) noexcept
{
#pragma unroll
    for (int r = 0; r < 8; ++r)
    {
#pragma unroll
        for (int c = 0; c < 8; ++c)
        {
            acc[r][c] = fmaf(a[r], b[c], acc[r][c]);
        }
    }
}

//! One k-group of this thread's 8 A rows and 8 B columns from the stage
//! `stg` (the A tile at its start, the B tile LDA * 8 floats on). AX (BX):
//! A (B) is the k-contiguous operand (leading dimension LDA (LDB) = 129,
//! scalar accesses); else leading dimension 128, 16-byte accesses.
template <bool AX, bool BX, int LDA, int LDB>
__device__ __forceinline__ void lds(float (&a)[8],
    float (&b)[8],
    const float *stg,
    int oa,
    int ob,
    int kk) noexcept
{
    if constexpr (AX)
    {
        const float *pa = stg + kk * LDA + oa;
#pragma unroll
        for (int r = 0; r < 8; ++r)
        {
            a[r] = pa[r];
        }
    }
    else
    {
        const float4 *pa =
            reinterpret_cast<const float4 *>(stg + kk * 128 + oa);
        float4 a0 = pa[0];
        float4 a1 = pa[1];
        a[0] = a0.x;
        a[1] = a0.y;
        a[2] = a0.z;
        a[3] = a0.w;
        a[4] = a1.x;
        a[5] = a1.y;
        a[6] = a1.z;
        a[7] = a1.w;
    }
    const float *bs = stg + LDA * 8;
    if constexpr (BX)
    {
        const float *pb = bs + kk * LDB + ob;
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                b[4 * h + c] = pb[32 * h + c];
            }
        }
    }
    else
    {
        const float4 *pb =
            reinterpret_cast<const float4 *>(bs + kk * 128 + ob);
        float4 b0 = pb[0];
        float4 b1 = pb[8];
        b[0] = b0.x;
        b[1] = b0.y;
        b[2] = b0.z;
        b[3] = b0.w;
        b[4] = b1.x;
        b[5] = b1.y;
        b[6] = b1.z;
        b[7] = b1.w;
    }
}

//! One panel (8 k-groups). On entry (ax, bx) hold k-group 0 of the panel in
//! stage `base_r`; on exit they hold k-group 0 of the panel in stage
//! `base_w` (NEXT: the panel has a successor).
template <bool NEXT, bool AX, bool BX, int LDA, int LDB>
__device__ __forceinline__ void panel(const float *base_r,
    const float *base_w,
    int oa,
    int ob,
    float (&ax)[8],
    float (&bx)[8],
    float (&ay)[8],
    float (&by)[8],
    float (&acc)[8][8]) noexcept
{
    lds<AX, BX, LDA, LDB>(ay, by, base_r, oa, ob, 1);
    ffma(ax, bx, acc);
    lds<AX, BX, LDA, LDB>(ax, bx, base_r, oa, ob, 2);
    ffma(ay, by, acc);
    lds<AX, BX, LDA, LDB>(ay, by, base_r, oa, ob, 3);
    ffma(ax, bx, acc);
    lds<AX, BX, LDA, LDB>(ax, bx, base_r, oa, ob, 4);
    ffma(ay, by, acc);
    lds<AX, BX, LDA, LDB>(ay, by, base_r, oa, ob, 5);
    ffma(ax, bx, acc);
    lds<AX, BX, LDA, LDB>(ax, bx, base_r, oa, ob, 6);
    ffma(ay, by, acc);
    lds<AX, BX, LDA, LDB>(ay, by, base_r, oa, ob, 7);
    ffma(ax, bx, acc);
    if constexpr (NEXT)
    {
        lds<AX, BX, LDA, LDB>(ax, bx, base_w, oa, ob, 0);
    }
    ffma(ay, by, acc);
}

//! The per-thread state of the loaders and the stores of their next panels,
//! shared by the prologue and the loop of a job.
template <int TA, int TB, int ACA, int ACB> struct Job
{
    static constexpr bool AX = TA == 0; //!< A is k-contiguous
    static constexpr bool BX = TB == 1; //!< B is k-contiguous
    static constexpr int LDA = AX ? 129 : 128;
    static constexpr int LDB = BX ? 129 : 128;
    static constexpr int SS = (LDA + LDB) * 8; //!< floats per stage

    float *smem;
    XLane<ACA> xa;
    XLane<ACB> xb;
    YLane<ACA> ya;
    YLane<ACB> yb;
    float4 gxa[4]; //!< the lane's chunk of A (k-contiguous)
    float4 gxb[4]; //!< the lane's chunk of B (k-contiguous)
    float4 ya0, yb0, ya1, yb1;
    int t, xr, xpi, xkg, yk, yq, npan;

    __device__ __forceinline__ Job(
        const Params &p, float *sm, int tile, int k0, int k1) noexcept :
        smem(sm),
        npan(k1 - k0),
        t(threadIdx.x),
        xr(threadIdx.x >> 3),
        xpi((threadIdx.x & 7) >> 1),
        xkg(4 * (threadIdx.x & 1)),
        yk(threadIdx.x >> 5),
        yq(4 * (threadIdx.x & 31))
    {
        const int m0 = (tile % p.mt) << 7;
        const int n0 = (tile / p.mt) << 7;
        if constexpr (AX)
        {
            xa.init(p.a, p.sa, m0 + xr, p.lda, 8LL * k0 + 4 * (t & 7), npan);
        }
        else
        {
            ya.init(p.a, p.sa, (8LL * k0 + yk) * p.lda + m0 + yq, 8LL * p.lda);
        }
        if constexpr (BX)
        {
            xb.init(p.b, p.sb, n0 + xr, p.ldb, 8LL * k0 + 4 * (t & 7), npan);
        }
        else
        {
            yb.init(p.b, p.sb, (8LL * k0 + yk) * p.ldb + n0 + yq, 8LL * p.ldb);
        }
    }

    //! Store the lane's chunk of a k-contiguous operand into stage `st`
    //! (`op`: the operand's base in shared memory, `ld` its leading
    //! dimension there).
    __device__ __forceinline__ void xsts(
        const float4 (&g)[4], float *op, int ld, int st) noexcept
    {
        float *s = smem + st * SS + (op - smem);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float *d = s + 32 * i + xr + xkg * ld;
            d[0] = g[i].x;
            d[ld] = g[i].y;
            d[2 * ld] = g[i].z;
            d[3 * ld] = g[i].w;
        }
    }

    //! Load the chunk `chunk` (panels 4 * chunk .. + 4): the lanes of an
    //! existing panel only.
    template <int AC>
    __device__ __forceinline__ void xload(
        float4 (&g)[4], XLane<AC> &x, int chunk) noexcept
    {
        if (4 * chunk + xpi < npan)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                g[i] = *reinterpret_cast<const float4 *>(x.at(i, chunk));
            }
        }
    }

    //! Store panel `u + 2` of both operands into stage (u + 2) % 3.
    __device__ __forceinline__ void sts(
        int u, const float4 &gay, const float4 &gby) noexcept
    {
        const int st = (u + 2) % 3;
        if constexpr (AX)
        {
            if (xpi == ((u + 2) & 3)) //!< the lane's panel of the chunk
            {
                xsts(gxa, smem, LDA, st);
            }
        }
        else
        {
            *reinterpret_cast<float4 *>(smem + st * SS + (yk << 7) + yq) = gay;
        }
        if constexpr (BX)
        {
            if (xpi == ((u + 2) & 3)) //!< the lane's panel of the chunk
            {
                xsts(gxb, smem + LDA * 8, LDB, st);
            }
        }
        else
        {
            *reinterpret_cast<float4 *>(
                smem + st * SS + LDA * 8 + (yk << 7) + yq) = gby;
        }
    }

    //! Issue the loads of panel `u + 3`: the chunk holding it (the
    //! k-contiguous operands), the panel itself (the others).
    __device__ __forceinline__ void load(
        float4 &gan, float4 &gbn, int u) noexcept
    {
        if constexpr (AX)
        {
            if (((u + 3) & 3) == 0)
            {
                xload<ACA>(gxa, xa, (u + 3) >> 2);
            }
        }
        else
        {
            gan = *reinterpret_cast<const float4 *>(ya.at());
            ya.elem += ya.step;
        }
        if constexpr (BX)
        {
            if (((u + 3) & 3) == 0)
            {
                xload<ACB>(gxb, xb, (u + 3) >> 2);
            }
        }
        else
        {
            gbn = *reinterpret_cast<const float4 *>(yb.at());
            yb.elem += yb.step;
        }
    }
};

//! One job: the tile `tile` over the panels [k0, k1). slot < 0: the job
//! covers the whole k and writes C; else the accumulators go to the
//! partials slot `slot`.
template <int TA, int TB, int ACA, int ACB>
__device__ __forceinline__ void job(
    const Params &p, float *smem, int tile, int k0, int k1, int slot) noexcept
{
    constexpr bool AX = TA == 0;
    constexpr bool BX = TB == 1;
    constexpr int LDA = AX ? 129 : 128;
    constexpr int LDB = BX ? 129 : 128;
    constexpr int SS = (LDA + LDB) * 8;
    Job<TA, TB, ACA, ACB> j(p, smem, tile, k0, k1);
    const int npan = j.npan;
    // Consumer map: lane (li, lj) of warp (wi, wj).
    const int t = j.t;
    const int wi = (t >> 5) & 3;
    const int wj = t >> 7;
    const int li = (t & 31) >> 3;
    const int lj = t & 7;
    const int ca = 32 * wi + 8 * li;
    const int cb = 64 * wj + 4 * lj;
    float acc[8][8];
#pragma unroll
    for (int r = 0; r < 8; ++r)
    {
#pragma unroll
        for (int c = 0; c < 8; ++c)
        {
            acc[r][c] = 0.f;
        }
    }
    float ax[8], bx[8], ay[8], by[8];
    // The previous job's readers must be done before the stages refill.
    __syncthreads();
    // Prologue: chunk 0 of the k-contiguous operands (panels 0 and 1 go to
    // stages 0 and 1, the other panels stay in registers); panels 0 and 1
    // of the others to stages 0 and 1, panel 2 to registers.
    if constexpr (AX)
    {
        j.xload<ACA>(j.gxa, j.xa, 0);
        if (j.xpi < 2)
        {
            j.xsts(j.gxa, smem, LDA, j.xpi);
        }
    }
    else
    {
        j.ya0 = *reinterpret_cast<const float4 *>(j.ya.at());
        j.ya.elem += j.ya.step;
        *reinterpret_cast<float4 *>(smem + (j.yk << 7) + j.yq) = j.ya0;
    }
    if constexpr (BX)
    {
        j.xload<ACB>(j.gxb, j.xb, 0);
        if (j.xpi < 2)
        {
            j.xsts(j.gxb, smem + LDA * 8, LDB, j.xpi);
        }
    }
    else
    {
        j.yb0 = *reinterpret_cast<const float4 *>(j.yb.at());
        j.yb.elem += j.yb.step;
        *reinterpret_cast<float4 *>(smem + LDA * 8 + (j.yk << 7) + j.yq) =
            j.yb0;
    }
    if (npan > 1)
    {
        if constexpr (!AX)
        {
            j.ya1 = *reinterpret_cast<const float4 *>(j.ya.at());
            j.ya.elem += j.ya.step;
            *reinterpret_cast<float4 *>(smem + SS + (j.yk << 7) + j.yq) =
                j.ya1;
        }
        if constexpr (!BX)
        {
            j.yb1 = *reinterpret_cast<const float4 *>(j.yb.at());
            j.yb.elem += j.yb.step;
            *reinterpret_cast<float4 *>(
                smem + SS + LDA * 8 + (j.yk << 7) + j.yq) = j.yb1;
        }
    }
    if (npan > 2)
    {
        // Registers of panel 2: stored by the first loop iteration.
        if constexpr (!AX)
        {
            j.ya0 = *reinterpret_cast<const float4 *>(j.ya.at());
            j.ya.elem += j.ya.step;
        }
        if constexpr (!BX)
        {
            j.yb0 = *reinterpret_cast<const float4 *>(j.yb.at());
            j.yb.elem += j.yb.step;
        }
    }
    __syncthreads();
    lds<AX, BX, LDA, LDB>(ax, bx, smem, ca, cb, 0);
    int u = 0;
    // Steady state: every panel has a successor.
    while (true)
    {
        if (u + 1 >= npan)
        {
            break;
        }
        if (u + 2 < npan)
        {
            j.sts(u, j.ya0, j.yb0); //!< panel u + 2
        }
        if (u + 3 < npan)
        {
            j.load(j.ya1, j.yb1, u); //!< panel u + 3
        }
        panel<true, AX, BX, LDA, LDB>(smem + (u % 3) * SS,
            smem + ((u + 1) % 3) * SS,
            ca,
            cb,
            ax,
            bx,
            ay,
            by,
            acc);
        __syncthreads();
        ++u;
        if (u + 1 >= npan)
        {
            break;
        }
        if (u + 2 < npan)
        {
            j.sts(u, j.ya1, j.yb1);
        }
        if (u + 3 < npan)
        {
            j.load(j.ya0, j.yb0, u);
        }
        panel<true, AX, BX, LDA, LDB>(smem + (u % 3) * SS,
            smem + ((u + 1) % 3) * SS,
            ca,
            cb,
            ax,
            bx,
            ay,
            by,
            acc);
        __syncthreads();
        ++u;
    }
    // Last panel.
    panel<false, AX, BX, LDA, LDB>(
        smem + (u % 3) * SS, nullptr, ca, cb, ax, bx, ay, by, acc);
    const int m0 = (tile % p.mt) << 7;
    const int n0 = (tile / p.mt) << 7;
    if (slot < 0)
    {
        // C = alpha * acc + beta * C, 16-byte stores.
        const long long mask_c = (1LL << p.sc) - 1;
        const bool with_beta = p.beta != 0.f;
        const float alpha = p.alpha;
        const float beta = p.beta;
#pragma unroll
        for (int rr = 0; rr < 8; ++rr)
        {
            const long long row =
                (long long) (m0 + 32 * wi + 8 * li + rr) * p.ldc + n0 +
                64 * wj + 4 * lj;
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                float *dst = at_c(p.c, row + 32 * h, mask_c, p.sc);
                float4 v;
                v.x = alpha * acc[rr][4 * h];
                v.y = alpha * acc[rr][4 * h + 1];
                v.z = alpha * acc[rr][4 * h + 2];
                v.w = alpha * acc[rr][4 * h + 3];
                if (with_beta)
                {
                    float4 o = *reinterpret_cast<const float4 *>(dst);
                    v.x += beta * o.x;
                    v.y += beta * o.y;
                    v.z += beta * o.z;
                    v.w += beta * o.w;
                }
                *reinterpret_cast<float4 *>(dst) = v;
            }
        }
    }
    else
    {
        // Raw partial, tile row-major.
        float *ws = p.ws + (long long) slot * 16384 + (32 * wi + 8 * li) * 128;
        const int col = 64 * wj + 4 * lj;
#pragma unroll
        for (int rr = 0; rr < 8; ++rr)
        {
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                *reinterpret_cast<float4 *>(ws + rr * 128 + col + 32 * h) =
                    make_float4(acc[rr][4 * h],
                        acc[rr][4 * h + 1],
                        acc[rr][4 * h + 2],
                        acc[rr][4 * h + 3]);
            }
        }
    }
}

//! The GEMM kernel.
//!   MODE 0: one CTA per tile (plain grid);
//!   MODE 1: one CTA per SM, whole tiles then the k-split ranges;
//!   MODE 2: one CTA per SM, whole tiles only (balanced count per CTA).
template <int TA, int TB, int ACA, int ACB, int MODE, int MINB>
__global__ void __launch_bounds__(256, MINB) gemm(Params p)
{
    constexpr int SS = ((TA == 0 ? 129 : 128) + (TB == 1 ? 129 : 128)) * 8;
    __shared__ float smem[3 * SS];
    if constexpr (MODE == 0)
    {
        job<TA, TB, ACA, ACB>(p, smem, blockIdx.x, 0, p.P, -1);
    }
    else if constexpr (MODE == 1)
    {
        const int first = blockIdx.x * p.wpc;
        for (int tile = first; tile < first + p.wpc; ++tile)
        {
            job<TA, TB, ACA, ACB>(p, smem, tile, 0, p.P, -1);
        }
        if (p.rem > 0)
        {
            const int len = p.rem * p.P;
            const int x0 = int((long long) blockIdx.x * len / p.np);
            const int x1 = int((long long) (blockIdx.x + 1) * len / p.np);
            if (x1 > x0)
            {
                const int q = x0 / p.P;
                const int k1 = (x1 / p.P == q) ? x1 % p.P : p.P;
                job<TA, TB, ACA, ACB>(
                    p, smem, p.whole + q, x0 % p.P, k1, 2 * blockIdx.x);
                if (x1 / p.P > q && x1 % p.P != 0)
                {
                    job<TA, TB, ACA, ACB>(p,
                        smem,
                        p.whole + x1 / p.P,
                        0,
                        x1 % p.P,
                        2 * blockIdx.x + 1);
                }
            }
        }
    }
    else
    {
        const int first = int((long long) blockIdx.x * p.nt / gridDim.x);
        const int last = int((long long) (blockIdx.x + 1) * p.nt / gridDim.x);
        for (int tile = first; tile < last; ++tile)
        {
            job<TA, TB, ACA, ACB>(p, smem, tile, 0, p.P, -1);
        }
    }
}

//! Finishes the k-split tiles: for every remainder tile, sums the parts
//! that cover it in ascending slot order (ascending k) and writes
//! C = alpha * sum + beta * C. Sixteen 256-thread blocks per tile, four
//! elements (one float4) per thread, tile row-major.
__global__ void __launch_bounds__(256) fixup(Params p)
{
    const int q = blockIdx.x >> 4;
    const int tile = p.whole + q;
    const int m0 = (tile % p.mt) << 7;
    const int n0 = (tile / p.mt) << 7;
    const int len = p.rem * p.P;
    const int lane = threadIdx.x & 31;
    // The parts of every range, by ballot: range b = lane + 32 * j touches
    // tile q with its part A ([x0, x1) starts inside the tile) and its part
    // B ([x0, x1) enters the tile from the previous one).
    unsigned mask_a[3];
    unsigned mask_b[3];
#pragma unroll
    for (int jj = 0; jj < 3; ++jj)
    {
        const int b = lane + 32 * jj;
        bool a = false;
        bool bb = false;
        if (b < p.np)
        {
            const int x0 = int((long long) b * len / p.np);
            const int x1 = int((long long) (b + 1) * len / p.np);
            if (x1 > x0 && x1 > q * p.P && x0 < (q + 1) * p.P)
            {
                a = (x0 / p.P == q);
                bb = (x0 / p.P < q) && (x1 % p.P != 0);
            }
        }
        mask_a[jj] = __ballot_sync(0xffffffffu, a);
        mask_b[jj] = __ballot_sync(0xffffffffu, bb);
    }
    // Ascending slot order is ascending k within the tile.
    int slots[16];
    int ns = 0;
#pragma unroll
    for (int jj = 0; jj < 3; ++jj)
    {
        unsigned ma = mask_a[jj];
        unsigned mb = mask_b[jj];
        while (ma | mb)
        {
            const int i = __ffs(ma | mb) - 1;
            const unsigned bit = 1u << i;
            if (ma & bit)
            {
                slots[ns++] = 2 * (i + 32 * jj);
            }
            if (mb & bit)
            {
                slots[ns++] = 2 * (i + 32 * jj) + 1;
            }
            ma &= ~bit;
            mb &= ~bit;
        }
    }
    // This thread's four elements of the tile (row-major).
    const int e = ((blockIdx.x & 15) * 256 + threadIdx.x) * 4;
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < ns; ++s)
    {
        const float4 w = *reinterpret_cast<const float4 *>(
            p.ws + (long long) slots[s] * 16384 + e);
        sum.x += w.x;
        sum.y += w.y;
        sum.z += w.z;
        sum.w += w.w;
    }
    const long long elem =
        (long long) (m0 + (e >> 7)) * p.ldc + n0 + (e & 127);
    float *dst = at_c(p.c, elem, (1LL << p.sc) - 1, p.sc);
    sum.x *= p.alpha;
    sum.y *= p.alpha;
    sum.z *= p.alpha;
    sum.w *= p.alpha;
    if (p.beta != 0.f)
    {
        const float4 o = *reinterpret_cast<const float4 *>(dst);
        sum.x += p.beta * o.x;
        sum.y += p.beta * o.y;
        sum.z += p.beta * o.z;
        sum.w += p.beta * o.w;
    }
    *reinterpret_cast<float4 *>(dst) = sum;
}

} // namespace nntile::kernel::gemm_chunked::ext::large1024
