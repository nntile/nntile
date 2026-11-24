/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm_chunked/ext/tails.cu
 * Piece `tails`: the chunked bridge for the shapes the family's law
 * leaves out.
 *
 * The core's shape law (every dimension 64 or a multiple of 128) keeps
 * GPT-2 XL (H=1600 = 12 x 128 + 64) and the 50257-wide lm head off the
 * native family. Here the bridge's mainloop runs with its operand
 * loaders in tail mode (gemm_chunked_bridge.cuh, BridgeQuadA/B with
 * Tails = true): a quad outside the operand is zero-filled, a quad that
 * is only partly inside or not 16-byte aligned is loaded element by
 * element. The epilogue already masks its rows and columns and stores
 * single elements, so it needs nothing.
 *
 * The piece claims two kinds of call:
 *   - any batch-1 shape outside the law (all flat/table pairs of A, B);
 *   - a law-abiding batch-1 shape with exactly one single-chunk input
 *     (the core bridge only exists with both inputs flat or both on
 *     tables).
 * The core's other configurations keep every law-abiding shape with
 * matched inputs: they are tuned, this kernel is general.
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

// The chunk-table-aware SIMT mainloop and epilogue iterators
#include "../gemm_chunked_bridge.cuh"

namespace nntile::kernel::gemm_chunked
{

namespace ext
{

namespace
{

//! This piece's configuration ids, local to it (the dispatcher adds its
//! base before handing the id out).
constexpr int kCfgTails = 0;

using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
using OutOp =
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
using Swz = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using Tile = cutlass::gemm::GemmShape<128, 128, 8>;

//! The bridge's kernel with tail-mode loaders. ACA and ACB select flat
//! (0) or chunk-table (1) addressing per input.
template <typename LA, typename LB, int ACA, int ACB> struct Tails
{
    static bool const kAQuadAlongK = cutlass::platform::is_same<LA, RM>::value;
    static bool const kBQuadAlongN = cutlass::platform::is_same<LB, RM>::value;
    using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<Tile,
        cutlass::gemm::GemmShape<32, 64, 8>,
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
    using Mma =
        cutlass::gemm::threadblock::MmaPipelined<typename MmaCore::Shape,
            cutlass::transform::threadblock::BridgeQuadA<LA, ACA, true>,
            cutlass::transform::threadblock::BridgeStoreA<kAQuadAlongK>,
            cutlass::transform::threadblock::BridgeQuadB<LB, ACB, true>,
            cutlass::transform::threadblock::BridgeStoreB<kBQuadAlongN>,
            float,
            RM,
            typename MmaCore::MmaPolicy>;
    using EpiParts = cutlass::epilogue::threadblock::
        DefaultEpilogueSimt<Tile, typename Mma::Operator, OutOp, 1>;
    using OutputTileIterator = cutlass::epilogue::threadblock::
        ChunkedEpilogueIterator<typename EpiParts::OutputTileThreadMap, float>;
    using Epilogue = cutlass::epilogue::threadblock::Epilogue<Tile,
        typename Mma::Operator,
        EpiParts::kPartitionsK,
        OutputTileIterator,
        typename EpiParts::AccumulatorFragmentIterator,
        typename EpiParts::WarpTileIterator,
        typename EpiParts::SharedLoadIterator,
        OutOp,
        typename EpiParts::Padding>;
    using Kernel =
        cutlass::gemm::kernel::GemmChunked<Mma, Epilogue, Swz, false>;
};

//! One launch. The operands are packed matrices (C row-major, A and B
//! in the layout their transpose flag says); a table operand's pointer
//! is its chunk table. C is both source and destination (beta).
template <typename LA, typename LB, int ACA, int ACB>
bool run(const Call &call, cudaStream_t stream) noexcept
{
    using T = Tails<LA, LB, ACA, ACB>;
    using Kernel = typename T::Kernel;
    Swz swz;
    cutlass::gemm::GemmCoord const problem(call.m, call.n, call.k);
    cutlass::gemm::GemmCoord const grid =
        swz.get_tiled_shape(problem, {128, 128, 8}, 1);
    typename T::Mma::IteratorA::TensorRef const ref_a(
        static_cast<float const *>(call.a), LA::packed({call.m, call.k}));
    typename T::Mma::IteratorB::TensorRef const ref_b(
        static_cast<float const *>(call.b), LB::packed({call.k, call.n}));
    // The epilogue's source reference is non-const too (C is read
    // through it when beta != 0).
    typename T::Epilogue::OutputTileIterator::TensorRef const ref_c(
        static_cast<float *>(call.c), RM::packed({call.m, call.n}));
    typename T::Epilogue::OutputTileIterator::TensorRef const ref_d(
        static_cast<float *>(call.c), RM::packed({call.m, call.n}));
    typename Kernel::Params params(problem,
        grid,
        ref_a,
        ref_b,
        ref_c,
        ref_d,
        typename T::Epilogue::OutputOp::Params(call.alpha, call.beta),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        call.a_shift,
        call.b_shift,
        call.c_shift,
        call.c_shift);
    cutlass::Kernel<Kernel><<<swz.get_grid_shape(grid),
        dim3(Kernel::kThreadCount, 1, 1),
        int(sizeof(typename Kernel::SharedStorage)),
        stream>>>(params);
    return cudaGetLastError() == cudaSuccess;
}

template <typename LA, typename LB>
bool run_pair(const Call &call, int aca, int acb, cudaStream_t stream) noexcept
{
    if (aca == 0)
    {
        return acb == 0 ? run<LA, LB, 0, 0>(call, stream)
                        : run<LA, LB, 0, 1>(call, stream);
    }
    return acb == 0 ? run<LA, LB, 1, 0>(call, stream)
                    : run<LA, LB, 1, 1>(call, stream);
}

bool law_dim(int d) noexcept { return d % 128 == 0 || d == 64; }

} // namespace

int tails_pick(int base, const Call &call) noexcept
{
    if (call.batch != 1 || call.m < 1 || call.n < 1 || call.k < 1)
    {
        return -1;
    }
    int const single = int(call.a_shift == 0) + int(call.b_shift == 0);
    if (law_dim(call.m) && law_dim(call.n) && law_dim(call.k))
    {
        // A law-abiding shape: only the mixed layouts the core bridge
        // declines (exactly one input single-chunk). The shape query
        // (no operand, all shifts 0) declines here.
        return single == 1 ? base + kCfgTails : -1;
    }
    return base + kCfgTails;
}

const char *tails_name(int cfg) noexcept
{
    return cfg == kCfgTails ? "chunked-ext-tails-128x128x8-quad" : "";
}

bool tails_launch(int cfg, const Call &call, cudaStream_t stream) noexcept
{
    if (cfg != kCfgTails || call.a == nullptr || call.b == nullptr ||
        call.c == nullptr)
    {
        return false;
    }
    int const aca = call.a_shift == 0 ? 0 : 1;
    int const acb = call.b_shift == 0 ? 0 : 1;
    int const ci =
        (static_cast<int>(call.ta) << 1) | static_cast<int>(call.tb);
    switch (ci)
    {
    case 0:
        return run_pair<RM, RM>(call, aca, acb, stream);
    case 1:
        return run_pair<RM, CM>(call, aca, acb, stream);
    case 2:
        return run_pair<CM, RM>(call, aca, acb, stream);
    default:
        return run_pair<CM, CM>(call, aca, acb, stream);
    }
}

} // namespace ext

} // namespace nntile::kernel::gemm_chunked

#endif // NNTILE_USE_CUDA
