/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm_chunked/ext/proj.cu
 * The GPT-2 MLP projection GEMMs at 4096 tokens with every operand on
 * chunk tables: forward (NT) and input gradient (NN), m = 4096, n = H,
 * k = 4H for H = 768, 1024 and 1280. The bridge's 128x128x8 threadblock
 * tile puts 192, 256 and 320 threadblocks on 68 SMs: 2.82, 3.76 and 4.71
 * waves, so the last wave idles 6% of the machine. The piece keeps the
 * bridge's kernel for the full waves and runs the last wave's tiles on
 * all SMs: each cut once along k, the parts reduced in fixed order
 * (proj_bridge.cuh).
 *
 * @version 1.0.0
 * */

// Corresponding header
#include "../gemm_chunked_ext.hh"

#ifdef NNTILE_USE_CUDA

// Third-party headers: the bridge header relies on these (as cuda.cu
// includes them); CUTLASS 2.11 is on this directory's include path.
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>

// The chunk-table-aware SIMT mainloop
#include "../gemm_chunked_bridge.cuh"

// The full-wave and last-wave kernels on the bridge's mainloop
#include "proj_bridge.cuh"

namespace nntile::kernel::gemm_chunked
{

namespace ext
{

namespace
{

//! This piece's configuration ids, local to it (the dispatcher adds its
//! base before handing the id out).
constexpr int kCfgWave = 0;  //!< full waves, then the cut last wave
constexpr int kCfgPlain = 1; //!< the plain grid through the same launcher

using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
using Epi =
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

//! The bridge's assembly (CUTLASS 2.11 SIMT fp32, threadblock 128x128x8,
//! warp 32x64x8, 2 stages, 256 threads, chunk-table addressing for A and
//! B) under the piece's kernels.
template <typename LA, typename LB> struct Wave
{
    using Base = cutlass::gemm::device::GemmBridge<float,
        LA,
        float,
        LB,
        float,
        RM,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        cutlass::gemm::GemmShape<128, 128, 8>,
        cutlass::gemm::GemmShape<32, 64, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        Epi,
        Swizzle,
        2,
        1,
        1>;
    using Kernel = cutlass::gemm::kernel::
        GemmWaveFull<typename Base::Mma, typename Base::Epilogue, Swizzle>;
    using Tail = cutlass::gemm::kernel::GemmWaveTail<typename Base::Mma,
        typename Base::Epilogue>;
};

//! The SM count of the current device: the wave size (one 256-thread
//! threadblock resides per SM). A device property, read once.
int wave_units() noexcept
{
    static int const cached = []
    {
        int dev = 0;
        int num = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(
                &num, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            num <= 0)
        {
            return 1;
        }
        return num;
    }();
    return cached;
}

//! One call: `tail` selects the cut last wave, else the plain grid.
template <typename W>
bool run_wave(const Call &call, cudaStream_t stream, bool tail) noexcept
{
    using Full = typename W::Kernel;
    using Tail = typename W::Tail;
    using LA = typename W::Base::LayoutA;
    using LB = typename W::Base::LayoutB;
    using TileIterator = typename W::Base::Epilogue::OutputTileIterator;
    constexpr int kM = Full::Mma::Shape::kM;
    constexpr int kN = Full::Mma::Shape::kN;
    constexpr int kK = Full::Mma::Shape::kK;
    constexpr int kFixupThreads = 256;
    static_assert(kM * kN % (4 * kFixupThreads) == 0,
        "the fixup hands each thread whole float4 vectors");

    int const m = call.m;
    int const n = call.n;
    int const k = call.k;
    int const grid_m = m / kM;
    int const grid_n = n / kN;
    int const tiles = grid_m * grid_n;
    int const kit = k / kK;

    // The schedule: whole threadblocks for the full waves, the last
    // wave's tiles cut once over all `units` threadblocks (heads and
    // the spare threadblocks' tails). A tail too thin to cut (its
    // tails would be shorter than a panel) keeps the plain grid.
    int const units = wave_units();
    int const waves = (tiles + units - 1) / units;
    int full = (waves - 1) * units;
    int rest = tiles - full;
    int extras = units - rest;
    // The tail as a block of the output, as wide along n as it can be
    // (so the fewest m panels are read twice: by a head and by a spare
    // threadblock's tail). No such block, or a tail too thin to cut
    // (tails shorter than a panel): the plain grid.
    int tail_m = 0;
    int tail_n0 = grid_n;
    for (int rows = grid_n; rows > 0 && tail_m == 0; --rows)
    {
        if (rest > 0 && rest % rows == 0 && rest / rows <= grid_m)
        {
            tail_m = rest / rows;
            tail_n0 = grid_n - rows;
        }
    }
    int per_extra = rest > 0 ? (rest + extras - 1) / extras : 0;
    int tail_len =
        rest > 0 ? int(double(rest) * kit / double(units) / per_extra) : 0;
    if (!tail || tail_m == 0 || tail_len <= 0 || tail_len >= kit)
    {
        full = tiles;
        rest = 0;
        tail_m = 0;
        tail_n0 = grid_n;
    }

    // The tail's partial slots (one tile each, for the heads and for
    // every tail slot of the spare threadblocks). Asked for before
    // anything runs: out of memory, the caller's fallback owns the call.
    float *ws = nullptr;
    if (rest > 0)
    {
        long long const slots =
            static_cast<long long>(rest) + extras * per_extra;
        ws = static_cast<float *>(scratch(
            static_cast<std::size_t>(slots) * kM * kN * sizeof(float)));
        if (ws == nullptr)
        {
            return false;
        }
    }

    // 1. The full waves: the bridge's own kernel
    typename Full::Params fp;
    cutlass::gemm::GemmCoord const tiled(grid_m, grid_n, 1);
    fp.problem_size = cutlass::gemm::GemmCoord(m, n, k);
    fp.grid_tiled_shape = tiled;
    fp.swizzle_log_tile = Swizzle().get_log_tile(tiled);
    fp.params_A = typename Full::Mma::IteratorA::Params(
        LA::packed({m, k}), call.a_shift);
    fp.ref_A = typename Full::Mma::IteratorA::TensorRef(
        static_cast<float const *>(call.a), LA::packed({m, k}));
    fp.params_B = typename Full::Mma::IteratorB::Params(
        LB::packed({k, n}), call.b_shift);
    fp.ref_B = typename Full::Mma::IteratorB::TensorRef(
        static_cast<float const *>(call.b), LB::packed({k, n}));
    fp.params_C =
        typename TileIterator::Params(RM::packed({m, n}), call.c_shift);
    fp.ref_C = typename TileIterator::TensorRef(
        static_cast<float *>(call.c), RM::packed({m, n}));
    fp.params_D =
        typename TileIterator::Params(RM::packed({m, n}), call.c_shift);
    fp.ref_D = typename TileIterator::TensorRef(
        static_cast<float *>(call.c), RM::packed({m, n}));
    fp.output_op = typename Full::OutputOp::Params(call.alpha, call.beta);
    fp.gemm_k_size = kit * kK;
    fp.tail_m = tail_m;
    fp.tail_n0 = tail_n0;

    static bool const configured = []
    {
        int const smem = int(sizeof(typename Full::SharedStorage));
        return cudaFuncSetAttribute(cutlass::Kernel<Full>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem) == cudaSuccess &&
               cudaFuncSetAttribute(cutlass::Kernel<Tail>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   int(sizeof(typename Tail::SharedStorage))) == cudaSuccess;
    }();
    if (!configured)
    {
        return false;
    }
    dim3 const grid = Swizzle().get_grid_shape(tiled);
    cutlass::Kernel<Full><<<grid,
        Full::kThreadCount,
        sizeof(typename Full::SharedStorage),
        stream>>>(fp);
    if (cudaGetLastError() != cudaSuccess)
    {
        return false;
    }
    if (rest <= 0)
    {
        return true;
    }

    // 2. The last wave's heads and tails, as partials
    typename Tail::Params tp;
    tp.problem_size = cutlass::gemm::GemmCoord(m, n, k);
    tp.params_A = fp.params_A;
    tp.ref_A = fp.ref_A;
    tp.params_B = fp.params_B;
    tp.ref_B = fp.ref_B;
    tp.params_S = typename TileIterator::Params(RM::packed({kM, kN}), 0);
    tp.partials = ws;
    tp.tail_m = tail_m;
    tp.tail_n0 = tail_n0;
    tp.tiles = rest;
    tp.extras = extras;
    tp.slot_stride = per_extra;
    tp.split = kit - tail_len;
    tp.kit = kit;
    cutlass::Kernel<Tail><<<rest + extras,
        Tail::kThreadCount,
        sizeof(typename Tail::SharedStorage),
        stream>>>(tp);
    if (cudaGetLastError() != cudaSuccess)
    {
        return false;
    }

    // 3. The fixed-order reduction of every tail tile
    cutlass::gemm::kernel::GemmWaveFixup<kM, kN, kFixupThreads>
        <<<rest, kFixupThreads, 0, stream>>>(ws,
            rest,
            extras,
            per_extra,
            tail_m,
            tail_n0,
            n,
            call.c,
            call.c_shift,
            call.alpha,
            call.beta);
    return cudaGetLastError() == cudaSuccess;
}

//! The calls this piece serves: shape (m, n, k, transposes), batch 1,
//! every operand on a chunk table. Sorted by (m, n, k, combo).
struct Claim
{
    int m;
    int n;
    int k;
    char combo; //!< (TransA?1:0)<<1 | (TransB?1:0)
    char cfg;   //!< the piece's configuration for the call
};

Claim const kClaims[] = {
    {4096, 768, 3072, 0, kCfgWave},  // mlp_proj_dgrad, GPT-2 small
    {4096, 768, 3072, 1, kCfgWave},  // mlp_proj_fwd, GPT-2 small
    {4096, 1024, 4096, 0, kCfgWave}, // mlp_proj_dgrad, GPT-2 medium
    {4096, 1024, 4096, 1, kCfgWave}, // mlp_proj_fwd, GPT-2 medium
    {4096, 1280, 5120, 0, kCfgWave}, // mlp_proj_dgrad, GPT-2 large
    {4096, 1280, 5120, 1, kCfgWave}, // mlp_proj_fwd, GPT-2 large
};

} // namespace

int proj_pick(int base, const Call &call) noexcept
{
    // A flat operand is shift 0; anything else is a chunk table.
    if (call.batch != 1 || call.a_shift == 0 || call.b_shift == 0)
    {
        return -1;
    }
    int const combo =
        (static_cast<int>(call.ta) << 1) | static_cast<int>(call.tb);
    for (Claim const &c : kClaims)
    {
        if (c.m == call.m && c.n == call.n && c.k == call.k &&
            c.combo == combo)
        {
            return base + c.cfg;
        }
    }
    return -1;
}

const char *proj_name(int cfg) noexcept
{
    switch (cfg)
    {
    case kCfgWave:
        return "chunked-ext-proj-wave-cut-128x128x8-quad-tdl";
    case kCfgPlain:
        return "chunked-ext-proj-plain-128x128x8-quad-tdl";
    }
    return "chunked-ext-proj-?";
}

bool proj_launch(int cfg, const Call &call, cudaStream_t stream) noexcept
{
    // Only the claimed calls (any chunk class): the schedule is measured
    // on those shapes alone. A flat A or B is not this piece's layout.
    if (cfg != kCfgWave && cfg != kCfgPlain)
    {
        return false;
    }
    if (proj_pick(0, call) < 0 || call.a_shift == 0 || call.b_shift == 0)
    {
        return false;
    }
    bool const tail = cfg == kCfgWave;
    int const ci =
        (static_cast<int>(call.ta) << 1) | static_cast<int>(call.tb);
    switch (ci)
    {
    case 0:
        return run_wave<Wave<RM, RM>>(call, stream, tail);
    case 1:
        return run_wave<Wave<RM, CM>>(call, stream, tail);
    default:
        return false; //!< transposed A: not this piece's rows
    }
}

} // namespace ext

} // namespace nntile::kernel::gemm_chunked

#endif // NNTILE_USE_CUDA
