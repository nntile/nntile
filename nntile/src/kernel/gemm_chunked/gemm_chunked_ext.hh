/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm_chunked/gemm_chunked_ext.hh
 * Extension pieces of the native chunked fp32 GEMM family: new
 * configurations that live in their own translation units.
 *
 * A piece is one file kernel/gemm_chunked/ext/<name>.cu (see ext/README.md
 * for the contract and a template). The build registers every such file by
 * its name: the piece defines <name>_pick, <name>_name and <name>_launch
 * in nntile::kernel::gemm_chunked::ext, and this dispatcher hands the
 * family's launcher the piece's configuration ids. The core translation
 * unit (cuda.cu) only calls pick(), name() and launch() here, so a piece
 * changes nothing until its file exists.
 *
 * @version 1.0.0
 * */

#pragma once

// Compile-time definitions (NNTILE_USE_CUDA comes from here: it must
// be known before the guard below)
#include <nntile/defs.h>

#ifdef NNTILE_USE_CUDA

// Standard libraries
#include <cstddef>

// Third-party headers
#include <cuda_runtime.h>

// Corresponding header
#include "nntile/kernel/gemm_chunked/cuda.hh"

namespace nntile::kernel::gemm_chunked::ext
{

//! Configuration ids at and above this belong to the pieces; the core
//! family (cuda.cu) keeps 0..99. Piece i (file-name order) owns
//! [kFirstCfg + i * kCfgSpan, + kCfgSpan).
constexpr int kFirstCfg = 100;
constexpr int kCfgSpan = 100;

//! One call, as the family's entry points received it (the operand
//! addressing modes are those of Operand in cuda.hh).
struct Call
{
    Op ta;
    Op tb;
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
    long long sb; //!< B batch stride in elements
    long long sc; //!< C batch stride in elements
    float alpha;
    float beta;
};

//! The piece's choice for a call: one of its own configuration ids,
//! `base + local id`, or -1 when the piece does not claim it. A piece
//! claims only the shapes AND layouts it serves (the call carries the
//! operand addressing modes); pieces are asked in file-name order and
//! the first claim wins.
using PickFn = int (*)(int base, const Call &call) noexcept;

//! Name of one of the piece's configuration ids (its own id, not `base`).
using NameFn = const char *(*) (int cfg) noexcept;

//! Launch one of the piece's configuration ids (its own id). Returns false
//! without launching when the piece cannot run the call: the caller's
//! fallback owns it, exactly as for a core configuration.
using LaunchFn = bool (*)(
    int cfg, const Call &call, cudaStream_t stream) noexcept;

//! The pieces' pick for the call: a configuration id >= kFirstCfg, or -1.
int pick(const Call &call) noexcept;

//! Name of a piece configuration id.
const char *name(int cfg) noexcept;

//! Launch a piece configuration id.
bool launch(int cfg, const Call &call, cudaStream_t stream) noexcept;

//! Grow-only device scratch (split-K partials and the like), shared by
//! the pieces; null when the allocation failed. Never freed, like the
//! core family's own buffers. Single-threaded launch discipline.
void *scratch(std::size_t bytes) noexcept;

} // namespace nntile::kernel::gemm_chunked::ext

#endif // NNTILE_USE_CUDA
