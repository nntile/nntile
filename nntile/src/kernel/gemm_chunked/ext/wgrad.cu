/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm_chunked/ext/wgrad.cu
 * The weight gradients of the MLP at 4096 tokens (TN; A k x m, B k x n
 * and C all chunk-table operands, batch 1; the six measured shapes of
 * GPT-2 small, medium and large, claimed exactly): the plain bridge maps
 * one 128x128 output tile to one threadblock and keeps one threadblock
 * resident per SM, so a weight gradient loses its last wave (GPT-2
 * small, k = S = 4096: 144 tiles on 68 SMs run as 3 layers, 1.40x
 * cuBLAS). This piece runs the same CUTLASS SIMT mainloop and epilogue
 * on BALANCED WORKERS (ext/wgrad_bridge.cuh): one CTA per SM, each with
 * the same number of k-iterations over whole tiles plus a slice of the
 * leftover tiles; the shared tiles are finished by a fixed-order
 * reduction kernel. Deterministic: no atomics, the split is a pure
 * function of the shape.
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

// The piece's own kernel stack (after the CUTLASS device headers)
#include "wgrad_bridge.cuh"

namespace nntile::kernel::gemm_chunked
{

namespace ext
{

namespace
{

//! This piece's configuration ids, local to it (the dispatcher adds its
//! base before handing the id out): the balanced-worker GEMM
//! (ext/wgrad_bridge.cuh) as threadblock 128x128x8, warp 32x64x8, 68
//! workers, numbered over the tiles in one of two orders -- a strip
//! along m (the bridge's order) or blocks of tiles that keep one layer
//! of workers on a compact part of the output.
enum Cfg
{
    kCfgStrip = 0,
    kCfgBlocked = 1
};

//! The shapes the piece measured (TN, batch 1, the three operands
//! through chunk tables): the MLP weight gradients at 4096 tokens,
//! GPT-2 small, medium and large. (m, n, k.)
constexpr int kShapes[][3] = {
    {768, 3072, 4096},
    {3072, 768, 4096},
    {1024, 4096, 4096},
    {4096, 1024, 4096},
    {1280, 5120, 4096},
    {5120, 1280, 4096},
};

//! The configuration of a measured shape, by its index in kShapes.
//! Measured (4 MiB engine, sweep): the strip order wins on the wide
//! grids (fc_wgrad: 6-10 tile-rows for 24-40 tile-columns), the blocked
//! order on the tall grids (proj_wgrad: 24-40 tile-rows for 3-10
//! tile-columns), where a strip layer reads every A tile-row at once.
constexpr Cfg kShapeCfg[] = {
    kCfgStrip,   //!< small fc_wgrad
    kCfgBlocked, //!< small proj_wgrad
    kCfgStrip,   //!< medium fc_wgrad
    kCfgBlocked, //!< medium proj_wgrad
    kCfgStrip,   //!< large fc_wgrad
    kCfgBlocked, //!< large proj_wgrad
};

//! True when the call is one of the measured shapes with the layout the
//! piece serves.
bool serves(const Call &call, int &shape) noexcept
{
    shape = -1;
    if (call.batch != 1 || call.ta != Op::Trans || call.tb != Op::NoTrans ||
        call.a_shift == 0 || call.b_shift == 0 || call.c_shift == 0)
    {
        return false;
    }
    for (int i = 0; i < int(sizeof(kShapes) / sizeof(kShapes[0])); ++i)
    {
        if (call.m == kShapes[i][0] && call.n == kShapes[i][1] &&
            call.k == kShapes[i][2])
        {
            shape = i;
            return true;
        }
    }
    return false;
}

} // namespace

int wgrad_pick(int base, const Call &call) noexcept
{
    // Claim exactly the calls this piece measured: shape (call.m,
    // call.n, call.k, call.ta, call.tb) and layout (a_shift, b_shift,
    // c_shift: 0 is a flat operand, anything else a chunk table).
    int shape;
    if (!serves(call, shape))
    {
        return -1;
    }
    return base + kShapeCfg[shape];
}

const char *wgrad_name(int cfg) noexcept
{
    switch (cfg)
    {
    case kCfgStrip:
        return "chunked-ext-wgrad-bal-128x128x8-w68";
    case kCfgBlocked:
        return "chunked-ext-wgrad-bal-128x128x8-w68-blocked";
    }
    return "chunked-ext-wgrad-?";
}

bool wgrad_launch(int cfg, const Call &call, cudaStream_t stream) noexcept
{
    int shape;
    if (!serves(call, shape))
    {
        return false;
    }
    switch (cfg)
    {
    case kCfgStrip:
        return wgradbt::run<128, 128, 8, 32, 64, 68, false>(call, stream);
    case kCfgBlocked:
        return wgradbt::run<128, 128, 8, 32, 64, 68, true>(call, stream);
    }
    return false;
}

} // namespace ext

} // namespace nntile::kernel::gemm_chunked

#endif // NNTILE_USE_CUDA
