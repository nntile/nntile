/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/mask_scalar.cc
 * NNHaul wrappers for mask_scalar operation
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/mask_scalar.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/mask_scalar.hh"
#include "nntile/nnhaul/chunk_units.hh"

#ifdef NNTILE_USE_CUDA
#include <algorithm>
#include <chrono>
#include <vector>
#endif

namespace nntile::nnhaul
{

#ifdef NNTILE_USE_CUDA
namespace
{
/*! In-trainer host-section instrument for one representative wrapper,
 * following the NNHAUL_BODY_DIAG approach: with NNHAUL_RING_DIAG=1 in the
 * environment, every chunked mask instance records steady-clock section
 * times of the merged-table host work -- (1) staging build (clear + the
 * operand adds) and (2) the table upload (begin + upload_tables, i.e. the
 * single H2D copy) -- and the process-exit summary prints medians/tails.
 * This is the instrument used to compare pinned and pageable host memory
 * (an in-trainer-population variant; the rejected pinned regime has since
 * been removed from the transport). Zero cost when off (one branch on a
 * cached bool per instance). */
struct RingDiag
{
    bool on = std::getenv("NNHAUL_RING_DIAG") != nullptr;
    std::vector<double> build_ns, up_ns;

    void sample(double b, double u)
    {
        if(build_ns.size() < 200000)
        {
            build_ns.push_back(b);
            up_ns.push_back(u);
        }
    }

    static void dump(char const *tag, std::vector<double> &v)
    {
        if(v.empty())
        {
            std::printf("ringdiag %-9s n=0\n", tag);
            return;
        }
        std::sort(v.begin(), v.end());
        std::size_t const n = v.size();
        auto q = [&](double p) {
            return v[std::size_t(p * double(n - 1))];
        };
        std::printf("ringdiag %-9s n=%zu med=%.2fus p90=%.2fus p99=%.2fus "
            "max=%.2fus\n",
            tag, n, q(0.5) / 1e3, q(0.9) / 1e3, q(0.99) / 1e3,
            v.back() / 1e3);
    }

    ~RingDiag()
    {
        if(!on || build_ns.empty())
        {
            return;
        }
        std::printf("NNHAUL_RING_DIAG mask_scalar host sections "
            "(build = staging clear+adds; up = table begin + ONE merged "
            "upload):\n");
        dump("build", build_ns);
        dump("up", up_ns);
        std::fflush(stdout);
    }
};

RingDiag &ring_diag() noexcept
{
    static RingDiag d;
    return d;
}
} // namespace
#endif

// ---------------------------------------------------------------------------
// Native chunked mask_scalar driver (chunk_units.hh scheme).
//
// Kernel index map (derived from kernel::mask_scalar::cpu/cuda, NOT
// assumed): data is nrows-by-ncols Fortran with element (i, j) at flat
// i + j*nrows - the CONTIGUOUS unit is the nrows-element COLUMN, and the
// mask is the small operand indexed by the row only (constant along the
// contiguous axis), exactly the slice-class geometry. A run of whole
// columns [j0, j0+L) inside one data chunk is therefore the plain kernel
// sub-call (nrows, L, mask, val, data + j0*nrows). A column straddling a
// data chunk boundary is staged through a bounded nrows temp (gathered -
// the op writes only masked-out rows - kernel (nrows, 1) on the temp,
// scattered back). The mask itself is one nrows-element segment: used in
// place when its range lies in one chunk, gathered ONCE into a bounded
// temp otherwise (every sub-call reads the whole mask). Writes are
// idempotent val assignments, so the piecewise result is BITWISE the
// plain call under any factorization. Zero new kernels; all-unchunked
// buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void mask_scalar_chunked(Index nrows, Index ncols,
    ::nnhaul::Buffer *buf_mask, Scalar val, ::nnhaul::Buffer *buf_data,
    Arena &arena, Sub &&sub)
{
    if(nrows <= 0 || ncols <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_data(buf_data);
    chunkunits::ChunkView<nntile::bool_t> const cv_mask(buf_mask);
    bool const chunked_data = cv_data.nchunks > 1;
    T *temp_col = chunked_data ? arena.alloc(std::size_t(nrows)) : nullptr;
    nntile::bool_t const *mask = nullptr;
    nntile::bool_t *temp_mask = nullptr;
    if(cv_mask.nchunks > 1
        && !chunkunits::in_one_chunk(cv_mask, 0, std::size_t(nrows)))
    {
        // The arena hands out T-sized blocks; the mask needs nrows bytes,
        // so round up in T units and view the block as bool_t (1-byte
        // alignment inside a T alignment).
        std::size_t const t_elems = (std::size_t(nrows) * sizeof(nntile::bool_t)
            + sizeof(T) - 1) / sizeof(T);
        temp_mask = reinterpret_cast<nntile::bool_t *>(arena.alloc(t_elems));
        chunkunits::gather_range(cv_mask, 0, std::size_t(nrows), temp_mask,
            arena);
        mask = temp_mask;
    }
    else
    {
        mask = cv_mask.ptr(0, 0);
    }
    if(!arena.ok_state() || mask == nullptr)
    {
        return;
    }
    Index j = 0;
    while(j < ncols && arena.ok_state())
    {
        std::size_t const f = std::size_t(j) * nrows;
        if(!chunkunits::in_one_chunk(cv_data, f, std::size_t(nrows)))
        {
            // Straddling column: stage it whole through the bounded temp.
            chunkunits::gather_range(cv_data, f, std::size_t(nrows),
                temp_col, arena);
            if(arena.ok_state())
            {
                sub(nrows, Index{1}, mask, val, temp_col);
                chunkunits::scatter_range(cv_data, f, std::size_t(nrows),
                    temp_col, arena);
            }
            ++j;
            continue;
        }
        std::size_t const c = cv_data.chunk_of(f);
        std::size_t const end = cv_data.chunk_end(c);
        Index L = static_cast<Index>((end - f) / nrows);
        if(L > ncols - j)
        {
            L = ncols - j;
        }
        if(L < 1)
        {
            L = 1;
        }
        sub(nrows, L, mask, val, cv_data.ptr(c, f - cv_data.chunk_begin(c)));
        j += L;
    }
}

} // namespace

//! Constructor
template<typename T>
MaskScalar<std::tuple<T>>::MaskScalar():
    codelet(
        "nntile_mask_scalar",
        &MaskScalar<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &MaskScalar<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &MaskScalar<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Mask scalar operation for NNHaul buffer on CPU
template<typename T>
void MaskScalar<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_data = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_mask = static_cast<::nnhaul::Buffer *>(buffers[1]);
    if((buf_data->chunk_count() == 1) && (buf_mask->chunk_count() == 1)
       )
    {
        kernel::mask_scalar::cpu<T>(args->nrows, args->ncols,
            buf_mask->get_ptr<nntile::bool_t>(), args->val,
            buf_data->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        mask_scalar_chunked<T>(
            args->nrows,
            args->ncols,
            buf_mask,
            args->val,
            buf_data,
            arena,
            [](Index r, Index c, nntile::bool_t const *pm, Scalar v, T *pd) {
                kernel::mask_scalar::cpu<T>(r, c, pm, v, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result partially unwritten (same failure class as a
        // failed kernel launch).
        std::fprintf(stderr,
            "nntile_mask_scalar: cpu chunked driver aborted after an "
            "allocation failure; data left unwritten\n");
    }
}
// Specializations of CPU wrapper for accelerated types
template<>
void MaskScalar<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MaskScalar<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MaskScalar<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Mask scalar NNHaul buffer on CUDA
template<typename T>
void MaskScalar<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_data = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_mask = static_cast<::nnhaul::Buffer *>(buffers[1]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if((buf_data->chunk_count() == 1) && (buf_mask->chunk_count() == 1)
       )
    {
        kernel::mask_scalar::cuda<T>(stream, args->nrows, args->ncols,
            buf_mask->get_ptr<nntile::bool_t>(), args->val,
            buf_data->get_ptr<T>());
        return;
    }
    // Chunked single-launch path: the WHOLE region is issued as ONE
    // chunk-aware kernel, elementwise and bitwise the whole-buffer call.
    // Transport: the device chunk-pointer table is the only regime.
    {
        chunkunits::ChunkView<T> const cv_data(buf_data);
        chunkunits::ChunkView<nntile::bool_t> const cv_mask(buf_mask);
        chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
        RingDiag &rdiag = ring_diag();
        std::chrono::steady_clock::time_point dt0, dt1;
        if(rdiag.on)
        {
            dt0 = std::chrono::steady_clock::now();
        }
        tabs.clear();
        std::size_t const o_data_ptrs = tabs.add(cv_data);
        std::size_t const o_mask_ptrs = tabs.add(cv_mask);
        chunkunits::ChunkPtrTable &arena = chunkunits::chunk_ptr_table();
        if(arena.begin(stream))
        {
            if(rdiag.on)
            {
                dt1 = std::chrono::steady_clock::now();
            }
            void const *table_base = arena.upload_tables(tabs);
            if(table_base != nullptr)
            {
                if(rdiag.on)
                {
                    std::chrono::steady_clock::time_point const dt2 =
                        std::chrono::steady_clock::now();
                    rdiag.sample(
                        std::chrono::duration<double, std::nano>(dt1 - dt0)
                            .count(),
                        std::chrono::duration<double, std::nano>(dt2 - dt1)
                            .count());
                }
                T *const *data_tab = arena.table_view<T *>(
                    table_base, o_data_ptrs);
                const nntile::bool_t *const *mask_tab = arena.table_view<const nntile::bool_t *>(
                    table_base, o_mask_ptrs);
                kernel::mask_scalar::chunked<T>(
                    stream,
                    args->nrows,
                    args->ncols,
                    args->val,
                    data_tab,
                    Index(cv_data.nchunks),
                    Index(cv_data.chunk_elems),
                    mask_tab,
                    Index(cv_mask.nchunks),
                    Index(cv_mask.chunk_elems));
                arena.end(stream);
                return;
            }
        }
    }
        // There is no per-piece
    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot table
    // serve any launch up to a 16 TiB operand, so a failed upload is an
    // invariant violation, not a mode to stage around.
    std::fprintf(stderr,
        "nntile_mask_scalar: chunk-ptr table cannot serve this launch (invariant "
        "violation; no per-piece staging fallback exists)\n");
    std::abort();
}

// Specializations of CUDA wrapper for accelerated types
template<>
void MaskScalar<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void MaskScalar<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void MaskScalar<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for mask_scalar tasks
template<typename T>
std::uint64_t MaskScalar<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::nnhaul::fnv1a(&args->nrows, sizeof(args->nrows), hash);
    hash = ::nntile::nnhaul::fnv1a(&args->ncols, sizeof(args->ncols), hash);
    return hash;
}

template<typename T>
void MaskScalar<std::tuple<T>>::submit(int worker_hint,
        Index nrows, Index ncols, ::nnhaul::Handle & mask, Scalar val, ::nnhaul::Handle & data)
//! Insert mask_scalar task into NNHaul task pool
/*! No argument checking is performed. All the inputs are packed and passed to
 * submit() call. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nrows = nrows;
    args->ncols = ncols;
    args->val = val;
    // Indicate maximal possible amount of writes as flops count
    double nflops = sizeof(T) * nrows * (ncols+1);
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked column driver; mask_scalar never stages.
    (void)nflops;
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_RW, &data }, { ACCESS_R, &mask } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class MaskScalar<std::tuple<nntile::fp64_t>>;
template class MaskScalar<std::tuple<nntile::fp32_t>>;
template class MaskScalar<std::tuple<nntile::fp32_fast_tf32_t>>;
template class MaskScalar<std::tuple<nntile::fp32_fast_fp16_t>>;
template class MaskScalar<std::tuple<nntile::fp32_fast_bf16_t>>;
template class MaskScalar<std::tuple<nntile::bf16_t>>;
template class MaskScalar<std::tuple<nntile::fp16_t>>;

//! Pack of mask_scalar operations for different types
mask_scalar_pack_t mask_scalar;

} // namespace nntile::nnhaul
