/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm_chunked/ext/large1024.cu
 * GPT-2 large at 1024 tokens, all-table (every operand multi-chunk): the
 * two projection GEMMs, NN and NT 1024 x 1280 x 5120. A thin C over a deep
 * k: 80 tiles of 128 x 128 on 68 SMs leave the plain bridge two rounds
 * (1.6x cuBLAS). The piece routes them to the chunk-table bridge (CUTLASS
 * 2.11 SIMT mainloop, gemm_chunked_bridge.cuh) with a DETERMINISTIC split-K
 * of a fixed slice count per shape and its own fixed-order reduction
 * (large1024_bridge.cuh).
 *
 * The piece also carries a persistent kernel of its own
 * (large1024_kernels.cuh): one CTA per SM, whole tiles first, then the
 * tiles a whole-per-CTA split leaves over as load-balanced k ranges with a
 * fixed-order fixup kernel. It is correct on every transpose combination
 * and chunk class but not yet competitive (about 1.3x cuBLAS on these
 * shapes: the address and load phase at the top of every panel is a burst
 * of integer work the FFMA pipe waits for), so the picker does not route
 * to it; its configurations stay pinnable.
 *
 * @version 1.0.0
 * */

// Corresponding header
#include "../gemm_chunked_ext.hh"

// Standard libraries
#include <cstdio>
#include <type_traits>

#ifdef NNTILE_USE_CUDA

// The piece's own kernels
#include "large1024_kernels.cuh"

// The CUTLASS bridge with this piece's split-K reduction
#include "large1024_bridge.cuh"

namespace nntile::kernel::gemm_chunked
{

namespace ext
{

namespace
{

//! This piece's configuration ids, local to it (the dispatcher adds its
//! base before handing the id out).
enum
{
    kCfgPlain = 0,       //!< one CTA per tile
    kCfgPersist = 1,     //!< one CTA per SM, whole tiles + k-split ranges
    kCfgPersistWhole = 2 //!< one CTA per SM, whole tiles only
};
constexpr int kNumCfg = 3;
//! One CTA per tile at two CTAs per SM (at most 128 registers).
constexpr int kCfgPlain2 = 7;
//! The CUTLASS bridge with the deterministic split-K: cfg = kBridgeSFirst +
//! slices (2..16).
constexpr int kBridgeSFirst = 60;
constexpr int kBridgeSMax = 16;
//! The slice count the two projection rows are routed with (measured on
//! sm_75: 5 beats 2, 4, 6, 8, 10, 12 and 16 on both rows).
constexpr int kProjSlices = 5;

//! The single-chunk sentinel (shift 62) and anything above read as 62.
constexpr int shift(int s) noexcept { return s < 62 ? s : 62; }

//! The current device's SM count (cached per device; 0 when unknown).
int sm_count() noexcept
{
    constexpr int kMaxDevices = 64;
    static int cache[kMaxDevices] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices)
    {
        return 0;
    }
    if (cache[dev] == 0)
    {
        int v = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) !=
                cudaSuccess ||
            v <= 0)
        {
            return 0;
        }
        cache[dev] = v;
    }
    return cache[dev];
}

//! The piece's own kernel, one of its kCfg ids.
template <int TA, int TB, int ACA, int ACB>
bool run(int mode, const Call &call, cudaStream_t stream) noexcept
{
    const int ns = sm_count();
    if (ns <= 0)
    {
        return false;
    }
    large1024::Params p{};
    p.a = call.a;
    p.b = call.b;
    p.c = call.c;
    p.sa = shift(call.a_shift);
    p.sb = shift(call.b_shift);
    p.sc = shift(call.c_shift);
    // Batch 1: the packed leading dimensions (C-order row-major A is
    // [m, k] or [k, m]; B is [k, n] or [n, k]).
    p.lda = TA == 0 ? call.k : call.m;
    p.ldb = TB == 0 ? call.n : call.k;
    p.ldc = call.n;
    p.mt = call.m / 128;
    p.nt = p.mt * (call.n / 128);
    p.P = call.k / 8;
    p.alpha = call.alpha;
    p.beta = call.beta;
    int grid = 0;
    if (mode == kCfgPlain || mode == kCfgPlain2)
    {
        p.whole = p.nt;
        p.wpc = 1;
        p.np = p.nt;
        grid = p.nt;
    }
    else if (mode == kCfgPersist)
    {
        p.rem = p.nt % ns;
        p.whole = p.nt - p.rem;
        p.wpc = p.whole / ns;
        p.np = ns;
        grid = ns;
    }
    else
    {
        p.whole = p.nt;
        p.np = ns;
        grid = ns < p.nt ? ns : p.nt;
    }
    if (p.rem > 0)
    {
        p.ws = static_cast<float *>(
            scratch(size_t(2) * size_t(ns) * 16384 * sizeof(float)));
        if (p.ws == nullptr)
        {
            return false;
        }
    }
    switch (mode)
    {
    case kCfgPlain:
        large1024::gemm<TA, TB, ACA, ACB, 0, 1><<<grid, 256, 0, stream>>>(p);
        break;
    case kCfgPersist:
        large1024::gemm<TA, TB, ACA, ACB, 1, 1><<<grid, 256, 0, stream>>>(p);
        break;
    case kCfgPlain2:
        large1024::gemm<TA, TB, ACA, ACB, 0, 2><<<grid, 256, 0, stream>>>(p);
        break;
    default:
        large1024::gemm<TA, TB, ACA, ACB, 2, 1><<<grid, 256, 0, stream>>>(p);
        break;
    }
    if (cudaGetLastError() != cudaSuccess)
    {
        return false;
    }
    if (p.rem > 0)
    {
        large1024::fixup<<<16 * p.rem, 256, 0, stream>>>(p);
    }
    return cudaGetLastError() == cudaSuccess;
}

//! The CUTLASS bridge's split-K kernel with the piece's fixed-order
//! reduction: `slices` slices, chunk-table operands, batch 1.
template <int TA, int TB>
bool run_bridge_sk(int slices, const Call &call, cudaStream_t stream) noexcept
{
    if (call.a_shift == 0 || call.b_shift == 0 || call.k % 128 != 0)
    {
        return false;
    }
    float *ws = static_cast<float *>(scratch(
        size_t(slices) * size_t(call.m) * size_t(call.n) * sizeof(float)));
    if (ws == nullptr)
    {
        return false;
    }
    using LA = std::conditional_t<TA == 0, large1024::RM, large1024::CM>;
    using LB = std::conditional_t<TB == 0, large1024::RM, large1024::CM>;
    return large1024::launch_sk<LA, LB>(call.a,
        shift(call.a_shift),
        call.b,
        shift(call.b_shift),
        call.c,
        shift(call.c_shift),
        call.m,
        call.n,
        call.k,
        call.alpha,
        call.beta,
        ws,
        slices,
        stream);
}

} // namespace

int large1024_pick(int base, const Call &call) noexcept
{
    // Claim exactly the calls this piece serves: the two projection GEMMs
    // of GPT-2 large at 1024 tokens, batch 1, every operand a chunk table
    // (a shift of 0 is a flat operand).
    if (call.batch != 1 || call.a_shift == 0 || call.b_shift == 0 ||
        call.c_shift == 0)
    {
        return -1;
    }
    int const combo =
        (static_cast<int>(call.ta) << 1) | static_cast<int>(call.tb);
    struct Shape
    {
        int m, n, k, combo, cfg;
    };
    static Shape const served[] = {
        {1024,
            1280,
            5120,
            0,
            kBridgeSFirst + kProjSlices}, //!< mlp_proj_dgrad NN
        {1024,
            1280,
            5120,
            1,
            kBridgeSFirst + kProjSlices}, //!< mlp_proj_fwd NT
    };
    for (Shape const &s : served)
    {
        if (s.m == call.m && s.n == call.n && s.k == call.k &&
            s.combo == combo)
        {
            return base + s.cfg;
        }
    }
    return -1;
}

const char *large1024_name(int cfg) noexcept
{
    switch (cfg)
    {
    case kCfgPlain:
        return "chunked-ext-large1024-plain-128x128x8-p4";
    case kCfgPersist:
        return "chunked-ext-large1024-persist-128x128x8-p4-ksplit";
    case kCfgPersistWhole:
        return "chunked-ext-large1024-persist-128x128x8-p4";
    case kCfgPlain2:
        return "chunked-ext-large1024-plain-128x128x8-p4-2cta";
    }
    if (cfg >= kBridgeSFirst && cfg <= kBridgeSFirst + kBridgeSMax)
    {
        // Diagnostics only (the benchmark's INFO line): the slice count is
        // part of the name. Not thread-safe, like the family's trace hook.
        static char name[64];
        std::snprintf(name,
            sizeof(name),
            "chunked-ext-large1024-bridge-128x128x8-quad-tdl-sk%d",
            cfg - kBridgeSFirst);
        return name;
    }
    return "chunked-ext-large1024-?";
}

bool large1024_launch(int cfg, const Call &call, cudaStream_t stream) noexcept
{
    if (call.batch != 1 || call.m % 128 != 0 || call.n % 128 != 0 ||
        call.n % 4 != 0 || call.k % 8 != 0)
    {
        return false;
    }
    int const ci =
        (static_cast<int>(call.ta) << 1) | static_cast<int>(call.tb);
    if (cfg >= kBridgeSFirst)
    {
        // The bridge's split-K: cfg = kBridgeSFirst + slices.
        if (cfg > kBridgeSFirst + kBridgeSMax)
        {
            return false;
        }
        int const slices = cfg - kBridgeSFirst;
        switch (ci)
        {
        case 0:
            return run_bridge_sk<0, 0>(slices, call, stream);
        case 1:
            return run_bridge_sk<0, 1>(slices, call, stream);
        case 2:
            return run_bridge_sk<1, 0>(slices, call, stream);
        default:
            return run_bridge_sk<1, 1>(slices, call, stream);
        }
    }
    if ((cfg < 0 || cfg >= kNumCfg) && cfg != kCfgPlain2)
    {
        return false;
    }
    // A flat operand is shift 0; anything else is a chunk table. Only the
    // matched pairs are instantiated.
    bool const flat = call.a_shift == 0;
    if (flat != (call.b_shift == 0))
    {
        return false;
    }
    int const ac = flat ? 0 : 1;
#define L1024_RUN4(TA, TB)                                                    \
    if (ac == 0)                                                              \
    {                                                                         \
        return run<TA, TB, 0, 0>(cfg, call, stream);                          \
    }                                                                         \
    return run<TA, TB, 1, 1>(cfg, call, stream);
    switch (ci)
    {
    case 0:
        L1024_RUN4(0, 0)
    case 1:
        L1024_RUN4(0, 1)
    case 2:
        L1024_RUN4(1, 0)
    default:
        L1024_RUN4(1, 1)
    }
#undef L1024_RUN4
    return false;
}

} // namespace ext

} // namespace nntile::kernel::gemm_chunked

#endif // NNTILE_USE_CUDA
