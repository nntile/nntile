/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm_chunked/ext/flat.cu
 * Flat-operand GEMMs of the GPT-2 MLP (CUTLASS SIMT fp32 engines, routed
 * per exact shape).
 *
 * Serves batch-1 calls whose three operands are flat (a single chunk each:
 * shift 0) on the classic shapes of the GPT-2 small and medium MLP at 1024
 * tokens. A flat operand needs no chunk-table resolve, so the engines are
 * CUTLASS 2.11's own SIMT fp32 mainloop on plain pointers (the native
 * flat engines measured against the chunk-table bridge: 1-8 % faster on
 * the transposed operands' loaders).
 *
 * Two engines, chosen per shape by a constant table:
 *
 *   - plain: threadblock 128x128x8 (or 64x128x8), warp 32x64x8, 2 stages;
 *   - serial split-K: the same mainloop with a fixed slice count, each
 *     slice adding into D in ascending k order under CUTLASS's Semaphore
 *     (acquire / release, no atomics: bitwise run-to-run deterministic).
 *     It fills the machine where one output tile per threadblock leaves
 *     SMs idle (48, 64 or 144 tiles on 68 SMs).
 *
 * The launchers skip the library's per-call semaphore memset: every
 * semaphore is back at 0 when the kernel retires (the last slice of a
 * tile releases 0), so the buffer is zeroed once, when it is allocated.
 *
 * @version 1.0.0
 * */

// Corresponding header
#include "../gemm_chunked_ext.hh"

#ifdef NNTILE_USE_CUDA

// Third-party headers (CUTLASS 2.11 is on this directory's include path)
#include <cutlass/gemm/device/gemm.h>

// Standard libraries
#include <cstdio>

namespace nntile::kernel::gemm_chunked
{

namespace ext
{

namespace
{

using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
using Epi =
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
using Swz = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using Ins = cutlass::gemm::GemmShape<1, 1, 1>;

//! CUTLASS 2.11 SIMT fp32 GEMM on flat pointers; Serial selects the
//! split-K-serial kernel (per-tile semaphores, fixed ascending order).
template <typename LA, typename LB, int BM, int BN, bool Serial>
using Flat = cutlass::gemm::device::Gemm<float,
    LA,
    float,
    LB,
    float,
    RM,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<BM, BN, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    Ins,
    Epi,
    Swz,
    2,
    1,
    1,
    Serial>;

//! The tile layouts the piece instantiates.
enum Tile : int
{
    kTile128x128 = 0,
    kTile64x128
};

//! One claimed call: the GEMM (m, n, k, transposes: (A?1:0)<<1 | (B?1:0))
//! and its engine (tile, slice count; 1 = the plain GEMM). Measured on the
//! RTX 2080 Ti rig (sweep of the stock and split engines per shape).
struct Route
{
    int m;
    int n;
    int k;
    char combo;
    char tile;
    char slices;
};

Route const kRoutes[] = {
    // GPT-2 medium, 1024 tokens (H = 1024, 4H = 4096)
    {1024, 4096, 1024, 0, kTile128x128, 1}, // mlp_fc_dgrad
    {1024, 4096, 1024, 1, kTile128x128, 1}, // mlp_fc_fwd
    {1024, 4096, 1024, 2, kTile128x128, 1}, // mlp_fc_wgrad
    {1024, 1024, 4096, 0, kTile128x128, 5}, // mlp_proj_dgrad
    {1024, 1024, 4096, 1, kTile128x128, 3}, // mlp_proj_fwd
    {4096, 1024, 1024, 2, kTile128x128, 1}, // mlp_proj_wgrad
    // GPT-2 small, 1024 tokens (H = 768, 4H = 3072)
    {1024, 3072, 768, 0, kTile128x128, 1}, // mlp_fc_dgrad
    {1024, 3072, 768, 1, kTile128x128, 1}, // mlp_fc_fwd
    {768, 3072, 1024, 2, kTile64x128, 3},  // mlp_fc_wgrad
    {1024, 768, 3072, 0, kTile128x128, 7}, // mlp_proj_dgrad
    {1024, 768, 3072, 1, kTile128x128, 4}, // mlp_proj_fwd
    {3072, 768, 1024, 2, kTile64x128, 3},  // mlp_proj_wgrad
};
constexpr int kNumRoutes = int(sizeof(kRoutes) / sizeof(kRoutes[0]));

//! The per-tile semaphores of the serial split-K kernels: one int per
//! output tile, in a private grow-only buffer (never freed, like the core
//! family's own buffers; single-threaded launch discipline). It is zeroed
//! once per allocation, synchronously: the kernel leaves every semaphore
//! it used at 0 (the last slice of a tile releases 0), so no call needs to
//! reset anything.
int *semaphores(int tiles) noexcept
{
    static int *buffer = nullptr;
    static int capacity = 0;
    if (tiles <= capacity)
    {
        return buffer;
    }
    int const grown = tiles + tiles / 2;
    int *fresh = nullptr;
    if (cudaMalloc(&fresh, grown * sizeof(int)) != cudaSuccess)
    {
        return nullptr;
    }
    if (cudaMemset(fresh, 0, grown * sizeof(int)) != cudaSuccess)
    {
        cudaFree(fresh);
        return nullptr;
    }
    // Kernels already enqueued on other streams may still hold the old
    // buffer: it is not freed.
    buffer = fresh;
    capacity = grown;
    return buffer;
}

template <typename G>
typename G::Arguments make_args(const Call &call, int slices) noexcept
{
    int const ta = static_cast<int>(call.ta);
    int const tb = static_cast<int>(call.tb);
    return typename G::Arguments({call.m, call.n, call.k},
        {static_cast<float const *>(call.a), ta ? call.m : call.k},
        {static_cast<float const *>(call.b), tb ? call.k : call.n},
        {static_cast<float *>(call.c), call.n},
        {static_cast<float *>(call.c), call.n},
        {call.alpha, call.beta},
        slices);
}

//! The plain GEMM through the library's launcher.
template <typename G>
bool run_plain(const Call &call, cudaStream_t stream) noexcept
{
    G gemm;
    return gemm.initialize(make_args<G>(call, 1), nullptr, stream) ==
               cutlass::Status::kSuccess &&
           gemm.run(stream) == cutlass::Status::kSuccess;
}

//! The serial split-K GEMM: the library kernel launched directly, without
//! the per-call semaphore memset of Gemm::initialize().
template <typename G>
bool run_serial(const Call &call, int slices, cudaStream_t stream) noexcept
{
    using Kernel = typename G::GemmKernel;
    int const tiles =
        ((call.m + G::ThreadblockShape::kM - 1) / G::ThreadblockShape::kM) *
        ((call.n + G::ThreadblockShape::kN - 1) / G::ThreadblockShape::kN);
    int *const locks = semaphores(tiles);
    if (locks == nullptr)
    {
        return false;
    }
    typename G::Arguments const args = make_args<G>(call, slices);
    typename G::ThreadblockSwizzle const swizzle;
    cutlass::gemm::GemmCoord const grid =
        swizzle.get_tiled_shape(args.problem_size,
            {G::ThreadblockShape::kM,
                G::ThreadblockShape::kN,
                G::ThreadblockShape::kK},
            args.split_k_slices);
    typename Kernel::Params const params(args.problem_size,
        grid,
        args.ref_A.non_const_ref(),
        args.ref_B.non_const_ref(),
        args.ref_C.non_const_ref(),
        args.ref_D,
        args.epilogue,
        locks);
    cutlass::Kernel<Kernel><<<swizzle.get_grid_shape(grid),
        dim3(Kernel::kThreadCount, 1, 1),
        sizeof(typename Kernel::SharedStorage),
        stream>>>(params);
    return cudaGetLastError() == cudaSuccess;
}

//! One engine over the transpose combos (LA / LB = storage layouts).
template <template <typename, typename, int, int, bool> class Eng,
    int BM,
    int BN>
bool run_tile(const Call &call, int slices, cudaStream_t stream) noexcept
{
    int const ci =
        (static_cast<int>(call.ta) << 1) | static_cast<int>(call.tb);
    if (call.batch != 1 || call.a_shift != 0 || call.b_shift != 0 ||
        call.c_shift != 0 || call.m % BM != 0 || call.n % BN != 0 ||
        call.k % 8 != 0)
    {
        return false;
    }
    if (slices < 2)
    {
        switch (ci)
        {
        case 0:
            return run_plain<Eng<RM, RM, BM, BN, false>>(call, stream);
        case 1:
            return run_plain<Eng<RM, CM, BM, BN, false>>(call, stream);
        case 2:
            return run_plain<Eng<CM, RM, BM, BN, false>>(call, stream);
        default:
            return run_plain<Eng<CM, CM, BM, BN, false>>(call, stream);
        }
    }
    switch (ci)
    {
    case 0:
        return run_serial<Eng<RM, RM, BM, BN, true>>(call, slices, stream);
    case 1:
        return run_serial<Eng<RM, CM, BM, BN, true>>(call, slices, stream);
    case 2:
        return run_serial<Eng<CM, RM, BM, BN, true>>(call, slices, stream);
    default:
        return run_serial<Eng<CM, CM, BM, BN, true>>(call, slices, stream);
    }
}

bool run_route(int route, const Call &call, cudaStream_t stream) noexcept
{
    Route const &r = kRoutes[route];
    if (r.tile == kTile128x128)
    {
        return run_tile<Flat, 128, 128>(call, r.slices, stream);
    }
    return run_tile<Flat, 64, 128>(call, r.slices, stream);
}

} // namespace

int flat_pick(int base, const Call &call) noexcept
{
    // Claim exactly the routed GEMMs on flat operands (shift 0).
    if (call.batch != 1 || call.a_shift != 0 || call.b_shift != 0 ||
        call.c_shift != 0)
    {
        return -1;
    }
    int const combo =
        (static_cast<int>(call.ta) << 1) | static_cast<int>(call.tb);
    for (int i = 0; i < kNumRoutes; ++i)
    {
        Route const &r = kRoutes[i];
        if (r.m == call.m && r.n == call.n && r.k == call.k &&
            r.combo == combo)
        {
            return base + i;
        }
    }
    return -1;
}

const char *flat_name(int cfg) noexcept
{
    if (cfg < 0 || cfg >= kNumRoutes)
    {
        return "chunked-ext-flat-?";
    }
    Route const &r = kRoutes[cfg];
    static char names[kNumRoutes][64];
    if (names[cfg][0] == '\0')
    {
        std::snprintf(names[cfg],
            sizeof names[cfg],
            "chunked-ext-flat-%dx128x8%s-%c%c-%d-%d-%d",
            r.tile == kTile128x128 ? 128 : 64,
            r.slices > 1 ? "-sks" : "",
            "NT"[r.combo >> 1],
            "NT"[r.combo & 1],
            r.m,
            r.n,
            r.k);
    }
    return names[cfg];
}

bool flat_launch(int cfg, const Call &call, cudaStream_t stream) noexcept
{
    if (cfg < 0 || cfg >= kNumRoutes || flat_pick(0, call) != cfg)
    {
        return false; //!< not one of this piece's claims
    }
    return run_route(cfg, call, stream);
}

} // namespace ext

} // namespace nntile::kernel::gemm_chunked

#endif // NNTILE_USE_CUDA
