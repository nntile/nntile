/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm_chunked/ext/large1024_bridge.cuh
 * The CUTLASS chunk-table bridge (gemm_chunked_bridge.cuh: mainloop and
 * split-K kernel, included unmodified) with this piece's own deterministic
 * reduction: a fixed slice count per shape, summed in ascending order.
 *
 * @version 1.0.0
 * */

#pragma once

// Third-party headers: the bridge header relies on these (as cuda.cu
// includes them); CUTLASS 2.11 is on this directory's include path.
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>

// The chunk-table-aware SIMT mainloop
#include "../gemm_chunked_bridge.cuh"

namespace nntile::kernel::gemm_chunked::ext::large1024
{

using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
using Epi =
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;

//! Fixed-order reduction of the split-K partials (slice-major workspace of
//! GemmChunkedSplitKParallel): C[e] = alpha * sum_s ws[s][e] + beta * C[e],
//! slices summed in ascending order by every thread; beta == 0 never reads
//! C. One float4 per thread; m * n is a multiple of 4.
__global__ void __launch_bounds__(256) sk_reduce(const float *__restrict__ ws,
    void *c,
    int c_shift,
    float alpha,
    float beta,
    long long total4,
    int slices)
{
    long long const e4 =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e4 >= total4)
    {
        return;
    }
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    long long const base = e4 * 4;
    long long const stride = total4 * 4;
    for (int s = 0; s < slices; ++s)
    {
        float4 const w =
            *reinterpret_cast<const float4 *>(ws + s * stride + base);
        acc.x += w.x;
        acc.y += w.y;
        acc.z += w.z;
        acc.w += w.w;
    }
    const int sh = c_shift < 62 ? c_shift : 62;
    const long long mask = (1LL << sh) - 1;
    float *out[4];
    float v[4] = {alpha * acc.x, alpha * acc.y, alpha * acc.z, alpha * acc.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        long long const e = base + i;
        out[i] = sh == 0
                     ? static_cast<float *>(c) + e
                     : static_cast<float *const *>(c)[e >> sh] + (e & mask);
        if (beta != 0.f)
        {
            v[i] += beta * out[i][0];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        out[i][0] = v[i];
    }
}

//! The bridge's split-K kernel on `slices` fixed slices plus sk_reduce.
//! Table operands only (A and B chunk tables, C flat or table), batch 1,
//! every dimension a multiple of 128.
template <typename LA, typename LB>
bool launch_sk(const void *a,
    int sa,
    const void *b,
    int sb,
    void *c,
    int sc,
    int m,
    int n,
    int k,
    float alpha,
    float beta,
    float *ws,
    int slices,
    cudaStream_t stream) noexcept
{
    using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<128, 128, 8>,
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
    using IteratorA = cutlass::transform::threadblock::BridgeQuadA<LA, 1>;
    using IteratorB = cutlass::transform::threadblock::BridgeQuadB<LB, 1>;
    static bool const kAQuadAlongK =
        cutlass::platform::is_same<LA, cutlass::layout::RowMajor>::value;
    static bool const kBQuadAlongN =
        cutlass::platform::is_same<LB, cutlass::layout::RowMajor>::value;
    using Mma =
        cutlass::gemm::threadblock::MmaPipelined<typename MmaCore::Shape,
            IteratorA,
            cutlass::transform::threadblock::BridgeStoreA<kAQuadAlongK>,
            IteratorB,
            cutlass::transform::threadblock::BridgeStoreB<kBQuadAlongN>,
            float,
            RM,
            typename MmaCore::MmaPolicy>;
    using EpiParts = cutlass::epilogue::threadblock::DefaultEpilogueSimt<
        cutlass::gemm::GemmShape<128, 128, 8>,
        typename Mma::Operator,
        Epi,
        1>;
    using OutputTileIterator = cutlass::epilogue::threadblock::
        ChunkedEpilogueIterator<typename EpiParts::OutputTileThreadMap, float>;
    using Epilogue = cutlass::epilogue::threadblock::Epilogue<
        cutlass::gemm::GemmShape<128, 128, 8>,
        typename Mma::Operator,
        EpiParts::kPartitionsK,
        OutputTileIterator,
        typename EpiParts::AccumulatorFragmentIterator,
        typename EpiParts::WarpTileIterator,
        typename EpiParts::SharedLoadIterator,
        Epi,
        typename EpiParts::Padding>;
    using Kernel = cutlass::gemm::kernel::GemmChunkedSplitKParallel<Mma,
        Epilogue,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>;
    using Swz = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    Swz swz;
    cutlass::gemm::GemmCoord const grid = swz.get_tiled_shape(
        cutlass::gemm::GemmCoord(m, n, k), {128, 128, 8}, slices);
    typename Kernel::Params kp(cutlass::gemm::GemmCoord(m, n, k),
        grid,
        {static_cast<float const *>(a), LA::packed({m, k})},
        {static_cast<float const *>(b), LB::packed({k, n})},
        {ws, RM::packed({m, n})},
        typename Kernel::OutputOp::Params(1.f, 0.f),
        sa,
        sb,
        0);
    static bool configured = []
    {
        return cudaFuncSetAttribute(cutlass::Kernel<Kernel>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   48 * 1024) == cudaSuccess;
    }();
    if (!configured)
    {
        return false;
    }
    int const smem = int(sizeof(typename Kernel::SharedStorage));
    cutlass::Kernel<Kernel>
        <<<swz.get_grid_shape(grid), dim3(256, 1, 1), smem, stream>>>(kp);
    if (cudaGetLastError() != cudaSuccess)
    {
        return false;
    }
    long long const total4 = static_cast<long long>(m) * n / 4;
    int const blocks = int((total4 + 255) / 256);
    sk_reduce<<<blocks, 256, 0, stream>>>(
        ws, c, sc, alpha, beta, total4, slices);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace nntile::kernel::gemm_chunked::ext::large1024
