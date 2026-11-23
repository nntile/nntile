/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems built on StarPU runtime system.
 *
 * @file tests/kernel/gemm_chunked_gpt2_bench.cc
 * Standalone benchmark (no ctest entry) of the GEMM routes the nnhaul gemm
 * wrapper picks from, over the GEMM shapes a GPT-2 training step issues
 * (torch_nntile/examples/train_gpt2.py: 15 distinct rows per model size):
 *
 *   cublas      plain kernel::cublas::gemm on flat operands;
 *   staged      the workspace-staged body of src/nnhaul/gemm.cc: every
 *               multi-chunk operand is gathered into a contiguous
 *               workspace slot (kernel::copy::chunked_segments), cuBLAS
 *               runs on the slots, a multi-chunk C is scattered back
 *               (kernel::copy::chunked); single-chunk operands are not
 *               staged, exactly as in the flag-off wrapper;
 *   staged_cut  the same body with the CUTLASS flat engine as the GEMM
 *               leg (kernel::gemm_chunked::flat_engine, the wrapper's
 *               flag-off batch-1 beta=0 optimization); timed only on the
 *               rows the engine accepts;
 *   chunked     the native chunked family in place on the chunk tables
 *               (kernel::gemm_chunked::launch_forced: the best native
 *               configuration for the shape, whatever the production
 *               ownership policy says);
 *   chunked_prod  production's own native choice on the same operands
 *               (kernel::gemm_chunked::launch, the ownership picker);
 *               timed only on the rows the picker owns.
 *
 * Operands are laid out like the engine's chunked tiles: a multi-chunk
 * operand is a table of whole-class chunks (the last one shorter), a
 * single-chunk operand is one flat allocation. The chunk class is a
 * command-line argument, so one build sweeps every class the native
 * family supports (any power-of-two class >= 1 MiB). The staged body's
 * copy kernels are compiled for the engine's ONE class (rule 7,
 * kernel::kChunkClassBytes from nnhaul/defs.h): at any other class the
 * staged routes do not exist and the row reports them as absent, so a
 * full three-way table at another class needs a build of the engine
 * for that class. Device chunk tables are built once per row and stay
 * alive: the per-launch chunk-pointer table upload of the wrapper is
 * NOT part of the timed region.
 *
 * Every route is checked against the cuBLAS result on the same data before
 * it is timed. Timings are per-iteration CUDA event pairs; the routes of a
 * row run in interleaved rounds so clock drift and thermal state hit them
 * alike. p10/p50/p90/min are reported over all iterations.
 *
 * Usage: bench_gemm_chunked_gpt2 [class_mib,...] [tokens,...] [models,...]
 *                               [rows,...]
 *   class_mib  chunk classes in MiB (default: the engine's class)
 *   tokens     sequence tokens per step, batch*seq (default 1024,4096)
 *   models     small,medium,large,xl (default all)
 *   rows       row-name prefixes to keep (default: all 15 rows; e.g.
 *              mlp_ for the six MLP rows, mlp_proj_fwd for one row)
 * Output: CSV lines on stdout, one INFO line per row and class and one
 * SAMPLES line per row, class and route. A kept row is timed exactly as
 * it is in a full sweep.
 *
 * @version 1.0.0
 * */

#include "nntile/base_types.hh"
#include "nntile/kernel/copy/cuda.hh"
#include "nntile/kernel/cublas.hh"
#include "nntile/kernel/cuda.hh"
#include "nntile/kernel/gemm_chunked/cuda.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace nntile;

#ifdef NNTILE_USE_CUDA

using kernel::gemm_chunked::Op;
using kernel::gemm_chunked::Operand;

namespace
{

constexpr int kRounds = 5;
constexpr int kWarmup = 3;
constexpr double kTargetMs = 30.0; //!< per round and route
constexpr int kMinIters = 10;
constexpr int kMaxIters = 500;

struct Model
{
    const char *name;
    int hidden;
    int heads;
    int layers;
};

Model const kModels[] = {
    {"small", 768, 12, 12},
    {"medium", 1024, 16, 24},
    {"large", 1280, 20, 36},
    {"xl", 1600, 25, 48},
};
constexpr int kVocab = 50257;
constexpr int kHeadSize = 64;

//! One GEMM row of a training step. `mult` is how many times the step
//! issues it (every layer, forward and backward), for step totals.
struct Row
{
    const char *op;
    Op ta;
    Op tb;
    long long m;
    long long n;
    long long k;
    long long batch;
    long long mult;
};

//! The 15 distinct rows a GPT-2 step issues for one model size.
std::vector<Row> make_rows(Model const &model, long long tokens)
{
    long long const H = model.hidden;
    long long const F = 4 * H;
    long long const L = model.layers;
    long long const heads = model.heads;
    long long const S = tokens;
    return {
        // Q, K, V and O projections share three shapes (4 calls/layer).
        {"qkvO_fwd", Op::NoTrans, Op::Trans, S, H, H, 1, 4 * L},
        {"qkvO_dgrad", Op::NoTrans, Op::NoTrans, S, H, H, 1, 4 * L},
        {"qkvO_wgrad", Op::Trans, Op::NoTrans, H, H, S, 1, 4 * L},
        // scores + dScores; ctx + dQ; dK + dV (2 calls/layer each).
        {"attn_scores", Op::NoTrans, Op::Trans, S, S, kHeadSize, heads, 2 * L},
        {"attn_ctx", Op::NoTrans, Op::NoTrans, S, kHeadSize, S, heads, 2 * L},
        {"attn_dKV", Op::Trans, Op::NoTrans, S, kHeadSize, S, heads, 2 * L},
        {"mlp_fc_fwd", Op::NoTrans, Op::Trans, S, F, H, 1, L},
        {"mlp_fc_dgrad", Op::NoTrans, Op::NoTrans, S, F, H, 1, L},
        {"mlp_fc_wgrad", Op::Trans, Op::NoTrans, H, F, S, 1, L},
        {"mlp_proj_fwd", Op::NoTrans, Op::Trans, S, H, F, 1, L},
        {"mlp_proj_dgrad", Op::NoTrans, Op::NoTrans, S, H, F, 1, L},
        {"mlp_proj_wgrad", Op::Trans, Op::NoTrans, F, H, S, 1, L},
        {"lm_head_fwd", Op::NoTrans, Op::Trans, S, kVocab, H, 1, 1},
        {"lm_head_dgrad", Op::NoTrans, Op::NoTrans, S, H, kVocab, 1, 1},
        {"lm_head_wgrad", Op::Trans, Op::NoTrans, kVocab, H, S, 1, 1},
    };
}

#define BENCH_CUDA_CHECK(call)                                                \
    do                                                                        \
    {                                                                         \
        cudaError_t const err_ = (call);                                      \
        if (err_ != cudaSuccess)                                              \
        {                                                                     \
            std::fprintf(stderr,                                              \
                "%s:%d: %s: %s\n",                                            \
                __FILE__,                                                     \
                __LINE__,                                                     \
                #call,                                                        \
                cudaGetErrorString(err_));                                    \
            std::exit(2);                                                     \
        }                                                                     \
    } while (false)

#define BENCH_CUBLAS_CHECK(call)                                              \
    do                                                                        \
    {                                                                         \
        cublasStatus_t const st_ = (call);                                    \
        if (st_ != CUBLAS_STATUS_SUCCESS)                                     \
        {                                                                     \
            std::fprintf(stderr,                                              \
                "%s:%d: %s: cublas status %d\n",                              \
                __FILE__,                                                     \
                __LINE__,                                                     \
                #call,                                                        \
                int(st_));                                                    \
            std::exit(2);                                                     \
        }                                                                     \
    } while (false)

std::uint32_t lcg(std::uint32_t &s)
{
    s = s * 1664525u + 1013904223u;
    return s >> 8;
}

//! A GEMM operand as the engine lays a chunked tile out.
struct Buf
{
    std::size_t elems = 0; //!< total elements over all batch items
    std::size_t cls = 0;   //!< chunk class in bytes
    int nchunks = 0;
    std::vector<float *> chunks;   //!< device bases, nchunks entries
    float *const *table = nullptr; //!< device chunk table (nchunks > 1)
    int shift = 0; //!< log2(class floats) when the table is set, else 0
};

Buf make_buf(std::size_t elems, std::size_t cls)
{
    Buf b;
    b.elems = elems;
    b.cls = cls;
    std::size_t const cls_elems = cls / sizeof(float);
    b.nchunks =
        int(std::max<std::size_t>(1, (elems + cls_elems - 1) / cls_elems));
    for (int c = 0; c < b.nchunks; ++c)
    {
        std::size_t const off = std::size_t(c) * cls_elems;
        std::size_t const len = std::min(cls_elems, elems - off);
        float *p = nullptr;
        BENCH_CUDA_CHECK(cudaMalloc(&p, len * sizeof(float)));
        b.chunks.push_back(p);
    }
    if (b.nchunks > 1)
    {
        BENCH_CUDA_CHECK(
            cudaMalloc((void **) &b.table, b.nchunks * sizeof(float *)));
        BENCH_CUDA_CHECK(cudaMemcpy((void *) b.table,
            b.chunks.data(),
            b.nchunks * sizeof(float *),
            cudaMemcpyHostToDevice));
        for (std::size_t v = cls_elems; (v >>= 1) != 0;)
        {
            ++b.shift;
        }
    }
    return b;
}

//! Every host copy rides the benchmark stream: a synchronous cudaMemcpy
//! from pageable memory returns before the DMA lands, and a non-blocking
//! stream is not ordered behind the legacy stream, so a copy issued that
//! way races the next kernel (a poisoned C zeroed the GEMM's output).
void buf_fill(
    Buf const &b, std::vector<float> const &host, cudaStream_t stream)
{
    std::size_t const cls_elems = b.cls / sizeof(float);
    for (int c = 0; c < b.nchunks; ++c)
    {
        std::size_t const off = std::size_t(c) * cls_elems;
        std::size_t const len = std::min(cls_elems, b.elems - off);
        BENCH_CUDA_CHECK(cudaMemcpyAsync(b.chunks[c],
            host.data() + off,
            len * sizeof(float),
            cudaMemcpyHostToDevice,
            stream));
    }
}

void buf_read(Buf const &b, std::vector<float> &host, cudaStream_t stream)
{
    std::size_t const cls_elems = b.cls / sizeof(float);
    for (int c = 0; c < b.nchunks; ++c)
    {
        std::size_t const off = std::size_t(c) * cls_elems;
        std::size_t const len = std::min(cls_elems, b.elems - off);
        BENCH_CUDA_CHECK(cudaMemcpyAsync(host.data() + off,
            b.chunks[c],
            len * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream));
    }
    BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void buf_free(Buf &b)
{
    for (float *p : b.chunks)
    {
        BENCH_CUDA_CHECK(cudaFree(p));
    }
    if (b.table != nullptr)
    {
        BENCH_CUDA_CHECK(cudaFree((void *) b.table));
    }
}

//! The static part of the workspace-staged body for one row: the operand
//! staging flags, the chunk-aligned slot placement and the slot tables.
struct Staged
{
    bool stage[3] = {false, false, false};
    std::size_t seg_chunk[3] = {0, 0, 0};
    std::size_t seg_nchunk[3] = {0, 0, 0};
    unsigned char *ws = nullptr;
    unsigned char *const *ws_table[3] = {nullptr, nullptr, nullptr};
    kernel::copy::Segment segs[2];
    int nseg = 0;
};

void staged_free(Staged &st)
{
    if (st.ws != nullptr)
    {
        BENCH_CUDA_CHECK(cudaFree(st.ws));
    }
    for (int i = 0; i < 3; ++i)
    {
        if (st.ws_table[i] != nullptr)
        {
            BENCH_CUDA_CHECK(cudaFree((void *) st.ws_table[i]));
        }
    }
}

//! Chunk-aligned placement, identical to the wrapper's: operand i occupies
//! the whole virtual chunks [k_i, k_i + ceil(bytes_i / cls)). The gather of
//! the staged inputs is ONE fused launch (beta == 0: C is never read).
void staged_build(Staged &st, Buf *const op[3], cudaStream_t stream)
{
    std::size_t used = 0;
    for (int i = 0; i < 3; ++i)
    {
        st.stage[i] = op[i]->nchunks > 1;
        if (st.stage[i])
        {
            st.seg_chunk[i] = used;
            st.seg_nchunk[i] =
                (op[i]->elems * sizeof(float) + op[i]->cls - 1) / op[i]->cls;
            used += st.seg_nchunk[i];
        }
    }
    if (used == 0)
    {
        return;
    }
    BENCH_CUDA_CHECK(cudaMalloc((void **) &st.ws, used * op[0]->cls));
    for (int i = 0; i < 3; ++i)
    {
        if (!st.stage[i])
        {
            continue;
        }
        std::vector<unsigned char *> tab(st.seg_nchunk[i]);
        for (std::size_t j = 0; j < st.seg_nchunk[i]; ++j)
        {
            tab[j] = st.ws + (st.seg_chunk[i] + j) * op[i]->cls;
        }
        BENCH_CUDA_CHECK(cudaMalloc(
            (void **) &st.ws_table[i], tab.size() * sizeof(void *)));
        BENCH_CUDA_CHECK(cudaMemcpyAsync((void *) st.ws_table[i],
            tab.data(),
            tab.size() * sizeof(void *),
            cudaMemcpyHostToDevice,
            stream));
    }
    for (int i = 0; i < 2; ++i)
    {
        if (st.stage[i])
        {
            kernel::copy::Segment &g = st.segs[st.nseg++];
            g.src = (const unsigned char *const *) op[i]->table;
            g.nsrc = Index(op[i]->nchunks);
            g.dst = (unsigned char *const *) st.ws_table[i];
            g.ndst = Index(st.seg_nchunk[i]);
            g.nbytes = Index(op[i]->elems * sizeof(float));
            g.elems_src = Index(op[i]->cls);
            g.elems_dst = Index(op[i]->cls);
        }
    }
}

struct Stats
{
    double p10 = 0, p50 = 0, p90 = 0, min = 0;
    int n = 0;
};

Stats stats_of(std::vector<float> us)
{
    Stats s;
    s.n = int(us.size());
    if (s.n == 0)
    {
        return s;
    }
    std::sort(us.begin(), us.end());
    s.min = us.front();
    s.p10 = us[std::min(s.n - 1, s.n / 10)];
    s.p50 = us[s.n / 2];
    s.p90 = us[std::min(s.n - 1, (9 * s.n) / 10)];
    return s;
}

//! Warm up, then time `iters` back-to-back calls with an event pair each.
void run_timed(std::function<void()> const &f,
    cudaStream_t stream,
    int iters,
    std::vector<float> &out)
{
    for (int i = 0; i < kWarmup; ++i)
    {
        f();
    }
    BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<cudaEvent_t> ev(2 * std::size_t(iters));
    for (cudaEvent_t &e : ev)
    {
        BENCH_CUDA_CHECK(cudaEventCreate(&e));
    }
    for (int i = 0; i < iters; ++i)
    {
        BENCH_CUDA_CHECK(cudaEventRecord(ev[2 * std::size_t(i)], stream));
        f();
        BENCH_CUDA_CHECK(cudaEventRecord(ev[2 * std::size_t(i) + 1], stream));
    }
    BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < iters; ++i)
    {
        float ms = 0.f;
        BENCH_CUDA_CHECK(cudaEventElapsedTime(
            &ms, ev[2 * std::size_t(i)], ev[2 * std::size_t(i) + 1]));
        out.push_back(ms * 1000.f);
    }
    for (cudaEvent_t e : ev)
    {
        BENCH_CUDA_CHECK(cudaEventDestroy(e));
    }
}

std::vector<std::string> split_list(char const *s)
{
    std::vector<std::string> out;
    char const *p = s;
    while (*p != '\0')
    {
        char const *q = std::strchr(p, ',');
        if (q == nullptr)
        {
            q = p + std::strlen(p);
        }
        if (q > p)
        {
            out.emplace_back(p, q);
        }
        p = (*q == ',') ? q + 1 : q;
    }
    return out;
}

struct Context
{
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
};

//! Worst |got - ref| / max(1, max|ref|) over C; NaN maps to a huge value.
double result_error(
    std::vector<float> const &got, std::vector<float> const &ref)
{
    double scale = 1.0;
    for (float r : ref)
    {
        scale = std::max(scale, double(std::fabs(r)));
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i)
    {
        double const d = std::fabs(double(got[i]) - double(ref[i]));
        worst = std::isnan(d) ? 1e300 : std::max(worst, d / scale);
    }
    return worst;
}

int bench_row(Context const &ctx,
    Model const &model,
    long long tokens,
    std::size_t cls,
    int cls_mib,
    Row const &row)
{
    using kernel::gemm_chunked::launch_forced;
    Op const ta = row.ta;
    Op const tb = row.tb;
    int const m = int(row.m), n = int(row.n), k = int(row.k);
    int const batch = int(row.batch);
    long long const sa = row.m * row.k, sb = row.k * row.n, sc = row.m * row.n;
    std::size_t const a_elems = std::size_t(batch) * std::size_t(sa);
    std::size_t const b_elems = std::size_t(batch) * std::size_t(sb);
    std::size_t const c_elems = std::size_t(batch) * std::size_t(sc);
    // Flat operands, the chunk-table copies and the workspace all live at
    // once: skip a row that cannot fit instead of dying on cudaMalloc.
    std::size_t free_b = 0, total_b = 0;
    BENCH_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::size_t const need =
        5 * (a_elems + b_elems + c_elems) * sizeof(float) +
        (std::size_t(256) << 20);
    if (need > free_b)
    {
        std::printf("SKIP,%s,%lld,%d,%s,need_MiB=%.0f,free_MiB=%.0f\n",
            model.name,
            tokens,
            cls_mib,
            row.op,
            double(need) / 1048576.0,
            double(free_b) / 1048576.0);
        std::fflush(stdout);
        return 0;
    }
    std::vector<float> ha(a_elems), hb(b_elems), hc(c_elems);
    std::uint32_t rng = 12345u;
    for (float &x : ha)
    {
        x = (int(lcg(rng) % 2001) - 1000) / 2000.0f;
    }
    for (float &x : hb)
    {
        x = (int(lcg(rng) % 2001) - 1000) / 2000.0f;
    }
    Buf ba = make_buf(a_elems, cls), bb = make_buf(b_elems, cls),
        bc = make_buf(c_elems, cls);
    buf_fill(ba, ha, ctx.stream);
    buf_fill(bb, hb, ctx.stream);
    // The staged body's copy kernels are compiled for the engine's ONE
    // chunk class (rule 7): another class has no staged route at all.
    bool const staged_class = cls == kernel::kChunkClassBytes;
    Staged st;
    Buf *op[3] = {&ba, &bb, &bc};
    if (staged_class)
    {
        staged_build(st, op, ctx.stream);
    }
    // The flat operands are separate allocations whatever the class.
    float *fa = nullptr, *fb = nullptr, *fc = nullptr;
    BENCH_CUDA_CHECK(cudaMalloc((void **) &fa, a_elems * sizeof(float)));
    BENCH_CUDA_CHECK(cudaMalloc((void **) &fb, b_elems * sizeof(float)));
    BENCH_CUDA_CHECK(cudaMalloc((void **) &fc, c_elems * sizeof(float)));
    BENCH_CUDA_CHECK(cudaMemcpyAsync(fa,
        ha.data(),
        a_elems * sizeof(float),
        cudaMemcpyHostToDevice,
        ctx.stream));
    BENCH_CUDA_CHECK(cudaMemcpyAsync(fb,
        hb.data(),
        b_elems * sizeof(float),
        cudaMemcpyHostToDevice,
        ctx.stream));
    // One barrier behind the setup: every table and operand is in place
    // (device-wide, so the chunk-table uploads are covered as well).
    BENCH_CUDA_CHECK(cudaDeviceSynchronize());
    TransOp const ta_ =
        TransOp(ta == Op::NoTrans ? TransOp::NoTrans : TransOp::Trans);
    TransOp const tb_ =
        TransOp(tb == Op::NoTrans ? TransOp::NoTrans : TransOp::Trans);
    auto cublas_gemm = [&](const float *pa, const float *pb, float *pc)
    {
        kernel::cublas::gemm<fp32_t>(ctx.handle,
            ta_,
            tb_,
            Index(m),
            Index(n),
            Index(k),
            Index(batch),
            Scalar{1.0},
            reinterpret_cast<fp32_t const *>(pa),
            reinterpret_cast<fp32_t const *>(pb),
            Scalar{0.0},
            reinterpret_cast<fp32_t *>(pc));
    };
    float *const pa =
        st.stage[0] ? reinterpret_cast<float *>(st.ws + st.seg_chunk[0] * cls)
                    : ba.chunks[0];
    float *const pb =
        st.stage[1] ? reinterpret_cast<float *>(st.ws + st.seg_chunk[1] * cls)
                    : bb.chunks[0];
    float *const pc =
        st.stage[2] ? reinterpret_cast<float *>(st.ws + st.seg_chunk[2] * cls)
                    : bc.chunks[0];
    auto route_staged = [&](bool cutlass_leg)
    {
        if (st.nseg > 0)
        {
            kernel::copy::chunked_segments(ctx.stream, st.segs, st.nseg);
        }
        bool const engine =
            cutlass_leg &&
            kernel::gemm_chunked::flat_engine(
                ta, tb, m, n, k, 1.0f, pa, pb, 0.0f, pc, ctx.stream);
        if (!engine)
        {
            cublas_gemm(pa, pb, pc);
        }
        if (st.stage[2])
        {
            kernel::copy::chunked(ctx.stream,
                Index(bc.elems * sizeof(float)),
                (const unsigned char *const *) st.ws_table[2],
                Index(st.seg_nchunk[2]),
                (unsigned char *const *) bc.table,
                Index(bc.nchunks),
                Index(cls),
                Index(cls));
        }
        return engine;
    };
    Operand const oa{
        ba.nchunks > 1 ? (const void *) ba.table : (const void *) ba.chunks[0],
        ba.nchunks > 1 ? ba.shift : 0};
    Operand const ob{
        bb.nchunks > 1 ? (const void *) bb.table : (const void *) bb.chunks[0],
        bb.nchunks > 1 ? bb.shift : 0};
    Operand const oc{
        bc.nchunks > 1 ? (const void *) bc.table : (const void *) bc.chunks[0],
        bc.nchunks > 1 ? bc.shift : 0};
    auto route_chunked = [&]()
    {
        return launch_forced(ta,
            tb,
            m,
            n,
            k,
            batch,
            1.0f,
            oa,
            ob,
            0.0f,
            oc,
            sa,
            sb,
            sc,
            ctx.stream);
    };
    // Reference, then every route on the same data (C poisoned first, so a
    // route that skips the write is caught).
    cublas_gemm(fa, fb, fc);
    BENCH_CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    std::vector<float> ref(c_elems);
    BENCH_CUDA_CHECK(cudaMemcpyAsync(ref.data(),
        fc,
        c_elems * sizeof(float),
        cudaMemcpyDeviceToHost,
        ctx.stream));
    BENCH_CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    auto verify = [&](std::function<bool()> const &route)
    {
        std::fill(hc.begin(), hc.end(), 0.f);
        buf_fill(bc, hc, ctx.stream);
        bool const ok = route();
        BENCH_CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        double err = -1.0;
        if (ok)
        {
            buf_read(bc, hc, ctx.stream);
            err = result_error(hc, ref);
        }
        return std::make_pair(ok, err);
    };
    bool const supported =
        kernel::gemm_chunked::supports(ta, tb, m, n, k, batch);
    std::pair<bool, double> const absent{false, -1.0};
    std::pair<bool, double> st_ok = absent, cut_ok = absent;
    // Production's own native choice: launch() honors the ownership
    // policy and may pick another configuration than launch_forced.
    auto route_prod = [&]()
    {
        return kernel::gemm_chunked::launch(ta,
            tb,
            m,
            n,
            k,
            batch,
            1.0f,
            oa,
            ob,
            0.0f,
            oc,
            sa,
            sb,
            sc,
            ctx.stream);
    };
    auto const prod_ok =
        supported ? verify([&] { return route_prod(); }) : absent;
    bool const owned = prod_ok.first;
    if (staged_class)
    {
        st_ok = verify(
            [&]
            {
                route_staged(false);
                return true;
            });
        cut_ok = verify([&] { return route_staged(true); });
    }
    auto const ch_ok = verify([&] { return route_chunked(); });
    // The timed routes.
    std::vector<char const *> names = {"cublas"};
    std::vector<std::function<void()>> routes = {
        [&] { cublas_gemm(fa, fb, fc); },
    };
    if (staged_class)
    {
        names.push_back("staged");
        routes.emplace_back([&] { route_staged(false); });
        if (cut_ok.first)
        {
            names.push_back("staged_cut");
            routes.emplace_back([&] { route_staged(true); });
        }
    }
    if (owned)
    {
        names.push_back("chunked_prod");
        routes.emplace_back([&] { route_prod(); });
    }
    if (ch_ok.first)
    {
        names.push_back("chunked");
        routes.emplace_back([&] { route_chunked(); });
    }
    std::vector<std::vector<float>> samples(routes.size());
    std::vector<int> iters(routes.size(), kMinIters);
    for (std::size_t v = 0; v < routes.size(); ++v)
    {
        std::vector<float> probe;
        run_timed(routes[v], ctx.stream, 3, probe);
        Stats const ps = stats_of(probe);
        iters[v] = std::clamp(int(kTargetMs * 1000.0 / std::max(ps.p50, 1.0)),
            kMinIters,
            kMaxIters);
    }
    for (int r = 0; r < kRounds; ++r)
    {
        for (std::size_t v = 0; v < routes.size(); ++v)
        {
            run_timed(routes[v], ctx.stream, iters[v], samples[v]);
        }
    }
    char const layout[4] = {char(ba.nchunks > 1 ? 't' : 'f'),
        char(bb.nchunks > 1 ? 't' : 'f'),
        char(bc.nchunks > 1 ? 't' : 'f'),
        '\0'};
    std::printf("INFO,%s,%lld,%d,%s,%c%c,%d,%d,%d,%d,%lld,%d/%d/%d,%s,%c,%c,"
                "%s,%c,%d,%.2e,%.2e,%.2e,%.2e\n",
        model.name,
        tokens,
        cls_mib,
        row.op,
        ta == Op::NoTrans ? 'N' : 'T',
        tb == Op::NoTrans ? 'N' : 'T',
        m,
        n,
        k,
        batch,
        row.mult,
        ba.nchunks,
        bb.nchunks,
        bc.nchunks,
        layout,
        supported ? 'y' : 'n',
        owned ? 'y' : 'n',
        supported ? kernel::gemm_chunked::pick_name(ta, tb, m, n, k, batch)
                  : "",
        cut_ok.first ? 'y' : 'n',
        staged_class ? 1 : 0,
        st_ok.second,
        cut_ok.first ? cut_ok.second : -1.0,
        ch_ok.first ? ch_ok.second : -1.0,
        owned ? prod_ok.second : -1.0);
    for (std::size_t v = 0; v < routes.size(); ++v)
    {
        Stats const s = stats_of(samples[v]);
        std::printf("SAMPLES,%s,%lld,%d,%s,%s,%.2f,%.2f,%.2f,%.2f,%d\n",
            model.name,
            tokens,
            cls_mib,
            row.op,
            names[v],
            s.p10,
            s.p50,
            s.p90,
            s.min,
            s.n);
    }
    std::fflush(stdout);
    staged_free(st);
    BENCH_CUDA_CHECK(cudaFree(fa));
    BENCH_CUDA_CHECK(cudaFree(fb));
    BENCH_CUDA_CHECK(cudaFree(fc));
    buf_free(ba);
    buf_free(bb);
    buf_free(bc);
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    // The default class is the engine's: the one class of a full table.
    std::vector<std::string> classes =
        argc > 1 ? split_list(argv[1])
                 : std::vector<std::string>{
                       std::to_string(kernel::kChunkClassBytes >> 20)};
    std::vector<std::string> tokens =
        argc > 2 ? split_list(argv[2])
                 : std::vector<std::string>{"1024", "4096"};
    std::vector<std::string> names =
        argc > 3 ? split_list(argv[3]) : std::vector<std::string>{};
    if (names.empty())
    {
        for (Model const &model : kModels)
        {
            names.push_back(model.name);
        }
    }
    std::vector<std::string> rows =
        argc > 4 ? split_list(argv[4]) : std::vector<std::string>{};
    auto row_kept = [&](char const *op)
    {
        if (rows.empty())
        {
            return true;
        }
        for (std::string const &p : rows)
        {
            if (std::strncmp(op, p.c_str(), p.size()) == 0)
            {
                return true;
            }
        }
        return false;
    };
    // A pinned configuration would replace the picker's choice everywhere.
    if (std::getenv("NNHAUL_GEMM_CHUNKED_CFG") != nullptr)
    {
        std::fprintf(stderr,
            "NNHAUL_GEMM_CHUNKED_CFG is set; unset it (the "
            "benchmark times the picker's choice)\n");
        return 1;
    }
    Context ctx;
    BENCH_CUBLAS_CHECK(cublasCreate(&ctx.handle));
    BENCH_CUDA_CHECK(
        cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking));
    BENCH_CUBLAS_CHECK(cublasSetStream(ctx.handle, ctx.stream));
    BENCH_CUBLAS_CHECK(
        cublasSetPointerMode(ctx.handle, CUBLAS_POINTER_MODE_HOST));
    std::printf("INFO and SAMPLES lines follow; times are microseconds per "
                "call, p50 is the median\n");
    std::printf("ENGINE,chunk_class_bytes=%zu,staged_routes_only_at_engine_"
                "class=1\n",
        kernel::kChunkClassBytes);
    for (std::string const &cs : classes)
    {
        long long const mib = std::atoll(cs.c_str());
        if (mib < 1 || (mib & (mib - 1)) != 0)
        {
            std::fprintf(stderr,
                "skipping class %s MiB (not a power of "
                "two >= 1)\n",
                cs.c_str());
            continue;
        }
        std::size_t const cls = std::size_t(mib) << 20;
        for (std::string const &ts : tokens)
        {
            long long const S = std::atoll(ts.c_str());
            for (std::string const &name : names)
            {
                Model const *model = nullptr;
                for (Model const &cand : kModels)
                {
                    if (name == cand.name)
                    {
                        model = &cand;
                    }
                }
                if (model == nullptr)
                {
                    std::fprintf(stderr, "unknown model %s\n", name.c_str());
                    return 1;
                }
                for (Row const &row : make_rows(*model, S))
                {
                    if (row_kept(row.op))
                    {
                        bench_row(ctx, *model, S, cls, int(mib), row);
                    }
                }
            }
        }
    }
    BENCH_CUDA_CHECK(cudaStreamDestroy(ctx.stream));
    BENCH_CUBLAS_CHECK(cublasDestroy(ctx.handle));
    return 0;
}

#else

int main() { return 0; }

#endif
