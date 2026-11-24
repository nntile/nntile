/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm_chunked/stub.cc
 * The staged-only-capable build (USE_CHUNKED_GEMM=OFF, root CMakeLists):
 * the native chunked-GEMM family is not compiled and these
 * constant-decline stand-ins keep the public contract of cuda.hh. Every
 * GEMM then stays on cuBLAS -- the workspace-staged path (gather into
 * the gemm workspace, cuBLAS, scatter back) for chunked operands, the
 * classic contiguous call otherwise -- the routing the GPT-2
 * forced-run protocol always uses. The one behavioral change: the
 * workspace-overflow rescue has no native escape hatch, so a call whose
 * staged demand exceeds the gemm workspace throws the staged path's
 * rule-11 error instead of being rescued.
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/gemm_chunked/cuda.hh"

namespace nntile::kernel::gemm_chunked
{

bool supports(Op, Op, int, int, int, int) noexcept
{
    return false;
}

bool supports_forced(Op, Op, int, int, int, int) noexcept
{
    return false;
}

const char *pick_name(Op, Op, int, int, int, int) noexcept
{
    return "";
}

bool launch_forced(Op, Op, int, int, int, int, float, const Operand &,
    const Operand &, float, const Operand &, long long, long long,
    long long, cudaStream_t) noexcept
{
    return false;
}

bool launch(Op, Op, int, int, int, int, float, const Operand &,
    const Operand &, float, const Operand &, long long, long long,
    long long, cudaStream_t) noexcept
{
    return false;
}

bool flat_engine(Op, Op, int, int, int, float, const void *, const void *,
    float, void *, cudaStream_t) noexcept
{
    return false;
}

} // namespace nntile::kernel::gemm_chunked
