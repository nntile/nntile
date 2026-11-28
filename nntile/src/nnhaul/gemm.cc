/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/gemm.cc
 * GEMM operation for NNHaul buffers
 *
 * @version 2.0.0
 * */

// Corresponding headers
#include "nntile/nnhaul/ops/gemm.hh"

// Standard libraries
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Third-party headers

// Other NNTile headers
#include "nntile/kernel/cublas.hh"
#include "nntile/kernel/copy/cuda.hh"
#include "nntile/nnhaul/chunk_units.hh"

#ifdef NNTILE_USE_CUDA
#include "nntile/kernel/gemm/cuda.hh"
#endif

#include <string>

namespace nntile::nnhaul
{

//! Constructor
template<typename T>
Gemm<std::tuple<T>>::Gemm():
    codelet(
        "nntile_gemm",
        nullptr,
#ifdef NNTILE_USE_CUDA
        &Gemm<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Gemm<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default.
    // Unsupported types no-op inside the CPU kernel.
}

// ---------------------------------------------------------------------------
// GEMM: cuBLAS on contiguous operands, a workspace copy for the rest.
//
// The kernel is deliberately minimal. For each operand decide whether it
// is already contiguous: one chunk, or a chain of chunks that sit at
// consecutive class-stride addresses. Contiguous inputs go straight to
// cuBLAS. A scattered input is copied once into the process-wide GEMM
// workspace; a scattered output is computed in the workspace and copied
// back. The actual copy is the shared kernel::copy::chunked driver.
//
// The single in-order compute stream makes each task's
// copy-in -> cuBLAS -> copy-out chain stream-ordered, so consecutive
// tasks cannot overlap their workspace use and every task reuses the
// workspace from offset 0. Staged bytes must fit the workspace or the
// driver throws nnhaul::Error (no fallback). NNHAUL_GEMM_WORKSPACE_BYTES
// shrinks the workspace for tests.
//
// The CPU body mirrors this over host scratch (chunkunits::ArenaCpu).
// NNHAUL_GEMM_STATS=1 prints one line per chunked body;
// NNHAUL_GEMM_SUMMARY=1 prints the process totals at exit, including the
// C2W (chunked -> workspace) and W2C (workspace -> chunked) chunk counts.
// ---------------------------------------------------------------------------

namespace
{

//! Process-wide GEMM staging meters. Relaxed atomics: the gemm body is
//! device-lane code and these counters are diagnostics only. The snapshot
//! API (gemm_staging_stats) and the NNHAUL_GEMM_SUMMARY=1 exit line expose
//! them.
struct GemmStagingMeters
{
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> contiguous{0};
    std::atomic<std::uint64_t> staged{0};
    std::atomic<std::uint64_t> staged_a{0};
    std::atomic<std::uint64_t> staged_b{0};
    std::atomic<std::uint64_t> staged_c{0};
    std::atomic<std::uint64_t> c2w_chunks{0};
    std::atomic<std::uint64_t> w2c_chunks{0};
    std::atomic<std::uint64_t> c2w_bytes{0};
    std::atomic<std::uint64_t> w2c_bytes{0};
};

GemmStagingMeters &gemm_meters() noexcept
{
    static GemmStagingMeters m;
    return m;
}

//! The exit-time summary: one key=value line after the run's own output.
struct GemmStagingReport
{
    ~GemmStagingReport()
    {
        char const *e = std::getenv("NNHAUL_GEMM_SUMMARY");
        if(e == nullptr || std::atoi(e) == 0)
        {
            return;
        }
        GemmStagingStats const m = gemm_staging_stats();
        std::fprintf(stderr,
            "[gemm-stats] calls=%llu contiguous=%llu staged=%llu "
            "staged_A=%llu staged_B=%llu staged_C=%llu "
            "c2w_chunks=%llu w2c_chunks=%llu "
            "c2w_MiB=%.1f w2c_MiB=%.1f\n",
            static_cast<unsigned long long>(m.calls),
            static_cast<unsigned long long>(m.contiguous),
            static_cast<unsigned long long>(m.staged),
            static_cast<unsigned long long>(m.staged_a),
            static_cast<unsigned long long>(m.staged_b),
            static_cast<unsigned long long>(m.staged_c),
            static_cast<unsigned long long>(m.c2w_chunks),
            static_cast<unsigned long long>(m.w2c_chunks),
            static_cast<double>(m.c2w_bytes) / (1024.0 * 1024.0),
            static_cast<double>(m.w2c_bytes) / (1024.0 * 1024.0));
    }
};

GemmStagingReport const gemm_staging_report;

//! Test/diagnostics hook: one stderr line per chunked body (path,
//! chunk counts, sub-call count) - the launch-count evidence for the
//! chunked checks.
bool gemm_stats() noexcept
{
    static bool const cached = []() noexcept {
        char const *e = std::getenv("NNHAUL_GEMM_STATS");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return cached;
}

//! Whether an operand is already contiguous: one chunk, or a chain of
//! chunks at consecutive class-stride addresses (pointer adjacency is
//! checked, not chunk-id arithmetic, so the fact holds however the pool
//! carved the arena). O(chunks), tiny.
template<typename T>
bool chain_contiguous(::nnhaul::Buffer *buf) noexcept
{
    std::size_t const n = buf->chunk_count();
    if (n <= 1)
    {
        return true;
    }
    char const *base =
        reinterpret_cast<char const *>(buf->get_chunk_ptr<T>(0));
    for (std::size_t i = 1; i < n; ++i)
    {
        char const *p =
            reinterpret_cast<char const *>(buf->get_chunk_ptr<T>(i));
        if (p != base + i * chunkunits::chunk_class_bytes())
        {
            return false;
        }
    }
    return true;
}

#ifdef NNTILE_USE_CUDA
//! The workspace-staged chunked driver (see the policy block above).
//! Throws nnhaul::Error when the staged bytes exceed the workspace
//! (nominal: the codelet is noexcept, so the throw terminates the
//! process with the message -- the rule-11 error surface).
template<typename T>
void gemm_workspace_staged(cublasHandle_t handle, TransOp transA,
    TransOp transB, Index m, Index n, Index k, Index batch, Scalar alpha,
    Scalar beta, bool broadcast_a, bool broadcast_b,
    ::nnhaul::Buffer *bufA, ::nnhaul::Buffer *bufB, ::nnhaul::Buffer *bufC)
{
    cudaStream_t stream = ::nnhaul::cuda_stream();
    chunkunits::ChunkView<T> const cvA(bufA);
    chunkunits::ChunkView<T> const cvB(bufB);
    chunkunits::ChunkView<T> const cvC(bufC);
    // The staging decision is per operand: only a scattered operand is
    // copied. A scattered C always gets a workspace slot for the output
    // and is copied back; it is copied IN only when beta != 0 reads it.
    bool const stage_a = !chain_contiguous<T>(bufA);
    bool const stage_b = !chain_contiguous<T>(bufB);
    bool const stage_c = !chain_contiguous<T>(bufC);
    {
        GemmStagingMeters &m = gemm_meters();
        m.staged.fetch_add(1, std::memory_order_relaxed);
        if(stage_a)
        {
            m.staged_a.fetch_add(1, std::memory_order_relaxed);
            m.c2w_chunks.fetch_add(bufA->chunk_count(),
                std::memory_order_relaxed);
            m.c2w_bytes.fetch_add(bufA->nbytes(),
                std::memory_order_relaxed);
        }
        if(stage_b)
        {
            m.staged_b.fetch_add(1, std::memory_order_relaxed);
            m.c2w_chunks.fetch_add(bufB->chunk_count(),
                std::memory_order_relaxed);
            m.c2w_bytes.fetch_add(bufB->nbytes(),
                std::memory_order_relaxed);
        }
        if(stage_c)
        {
            m.staged_c.fetch_add(1, std::memory_order_relaxed);
            // The output always travels workspace -> C (W2C).
            m.w2c_chunks.fetch_add(bufC->chunk_count(),
                std::memory_order_relaxed);
            m.w2c_bytes.fetch_add(bufC->nbytes(),
                std::memory_order_relaxed);
            if(beta != Scalar{0})
            {
                // beta != 0 makes cuBLAS read C, so gather it too (C2W).
                m.c2w_chunks.fetch_add(bufC->chunk_count(),
                    std::memory_order_relaxed);
                m.c2w_bytes.fetch_add(bufC->nbytes(),
                    std::memory_order_relaxed);
            }
        }
    }
    // The scatter rewrites C's WHOLE flat
    // byte range with the gemm's logical output; a staged C larger than
    // m*n*batch elements would have its excess bytes overwritten with
    // uninitialized workspace bytes (the classic leaves them untouched).
    // Production tiles are exact-size; violations abort loudly instead
    // of corrupting silently.
    if(stage_c
        && bufC->nbytes() != std::size_t(m)*std::size_t(n)
            *std::size_t(batch)*sizeof(T))
    {
        std::fprintf(stderr,
            "nntile_gemm: staged C is %zu bytes but the gemm writes "
            "only %zu -- oversized C handles are unsupported "
            "(bitwise-classic precondition)\n",
            bufC->nbytes(),
            std::size_t(m)*std::size_t(n)*std::size_t(batch)*sizeof(T));
        std::abort();
    }
    // The workspace is virtually chunked at the class size: operand i
    // occupies whole virtual chunks [k_i, k_i + ceil(nbytes_i/class)),
    // so the workspace side of every copy is an ordinary chunk table of
    // the same class as the operand. Padding in an operand's last virtual
    // chunk stays inside the workspace and is never written past nbytes.
    kernel::gemm::Workspace &ws = kernel::gemm::workspace();
    std::size_t const cls = chunkunits::chunk_class_bytes();
    std::size_t used_chunks = 0;
    std::size_t seg_chunk[3] = {0, 0, 0};
    std::size_t seg_nchunk[3] = {0, 0, 0};
    auto place = [&](bool staged, ::nnhaul::Buffer *buf, int i) {
        if(staged)
        {
            seg_chunk[i] = used_chunks;
            seg_nchunk[i] = (buf->nbytes() + cls - 1) / cls;
            used_chunks += seg_nchunk[i];
        }
    };
    place(stage_a, bufA, 0);
    place(stage_b, bufB, 1);
    place(stage_c, bufC, 2);
    if(used_chunks * cls > ws.peak_bytes)
    {
        ws.peak_bytes = used_chunks * cls;
    }
    if(used_chunks * cls > ws.bytes)
    {
        std::string need;
        auto name = [&](bool staged, char const *n, ::nnhaul::Buffer *buf,
                        int i) {
            if(!staged)
            {
                return;
            }
            if(!need.empty())
            {
                need += " + ";
            }
            need += n;
            need += " (";
            need += std::to_string(buf->nbytes());
            need += " bytes = ";
            need += std::to_string(seg_nchunk[i]);
            need += " chunks)";
        };
        name(stage_a, "A", bufA, 0);
        name(stage_b, "B", bufB, 1);
        name(stage_c, "C", bufC, 2);
        throw ::nnhaul::Error(
            "gemm workspace (" + std::to_string(ws.bytes) + " bytes ~ "
            + std::to_string(ws.bytes / cls) + " whole chunks) exceeded: needs "
            + std::to_string(used_chunks) + " chunks ("
            + std::to_string(used_chunks * cls)
            + " bytes) for " + need);
    }
    // One merged chunk-ptr table for every staged operand. Only table
    // bytes cross the control stream; workspace memory never does.
    chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
    tabs.clear();
    std::size_t slot_off[3] = {0, 0, 0};
    if(stage_a)
    {
        slot_off[0] = tabs.add(cvA);
    }
    if(stage_b)
    {
        slot_off[1] = tabs.add(cvB);
    }
    if(stage_c)
    {
        slot_off[2] = tabs.add(cvC);
    }
    chunkunits::ChunkPtrTable &table = chunkunits::chunk_ptr_table();
    if(!table.begin(stream))
    {
        std::fprintf(stderr,
            "nntile_gemm: chunk-ptr table init failed (no fallback exists)\n");
        std::abort();
    }
    // The workspace side of each staged operand is its whole virtual
    // chunk table [ws + k_i*cls, ws + (k_i+1)*cls, ...], appended to the
    // same merged upload. The copy itself is kernel::copy::chunked.
    std::size_t ws_off[3] = {0, 0, 0};
    auto ws_slots = [&](int i) {
        std::vector<unsigned char *> tab(seg_nchunk[i]);
        for(std::size_t j = 0; j < seg_nchunk[i]; ++j)
        {
            tab[j] = ws.base + (seg_chunk[i] + j) * cls;
        }
        return tabs.add(tab);
    };
    if(stage_a)
    {
        ws_off[0] = ws_slots(0);
    }
    if(stage_b)
    {
        ws_off[1] = ws_slots(1);
    }
    if(stage_c)
    {
        ws_off[2] = ws_slots(2);
    }
    // ONE merged upload: operand tables + workspace slots.
    void const *table_base = table.upload_tables(tabs);
    if(table_base == nullptr)
    {
        std::fprintf(stderr,
            "nntile_gemm: chunk-ptr table cannot serve the launch "
            "(no fallback exists)\n");
        std::abort();
    }
    // ONE gather call for ALL staged input operands: the segments ride
    // distinct chunk-table slices but a single kernel launch (grid.y
    // selects the segment, so no flat-range stitching and no per-operand
    // launch).
    kernel::copy::Segment segs[3];
    int nseg = 0;
    auto add_segment = [&](int i, ::nnhaul::Buffer *buf) {
        kernel::copy::Segment &g = segs[nseg++];
        g.src = table.table_view<const unsigned char *>(table_base,
            slot_off[i]);
        g.nsrc = Index(buf->chunk_count());
        g.dst = table.table_view<unsigned char *>(table_base, ws_off[i]);
        g.ndst = Index(seg_nchunk[i]);
        g.nbytes = Index(buf->nbytes());
        g.elems_src = Index(chunkunits::chunk_class_bytes());
        g.elems_dst = Index(chunkunits::chunk_class_bytes());
    };
    if(stage_a)
    {
        add_segment(0, bufA);
    }
    if(stage_b)
    {
        add_segment(1, bufB);
    }
    if(stage_c && beta != Scalar{0})
    {
        add_segment(2, bufC);
    }
    if(nseg > 0)
    {
        kernel::copy::chunked_segments(stream, segs, nseg);
    }
    // The one cuBLAS call, on whatever is contiguous now. The gather and
    // scatter are exact copies, so the result is bitwise the classic
    // call's.
    T const *pa = stage_a
        ? reinterpret_cast<T const *>(ws.base + seg_chunk[0] * cls)
        : bufA->get_chunk_ptr<T>(0);
    T const *pb = stage_b
        ? reinterpret_cast<T const *>(ws.base + seg_chunk[1] * cls)
        : bufB->get_chunk_ptr<T>(0);
    T *pc = stage_c
        ? reinterpret_cast<T *>(ws.base + seg_chunk[2] * cls)
        : bufC->get_chunk_ptr<T>(0);
    kernel::cublas::gemm<T>(handle, transA, transB, m, n, k, batch, alpha,
        pa, pb, beta, pc, broadcast_a, broadcast_b);
    if(stage_c)
    {
        kernel::copy::chunked(
            stream,
            Index(bufC->nbytes()),
            table.table_view<const unsigned char *>(table_base, ws_off[2]),
            Index(seg_nchunk[2]),
            table.table_view<unsigned char *>(table_base, slot_off[2]),
            Index(bufC->chunk_count()),
            Index(chunkunits::chunk_class_bytes()),
            Index(chunkunits::chunk_class_bytes()));
    }
    table.end(stream);
    if(gemm_stats())
    {
        std::size_t const c2w = (stage_a ? cvA.nchunks : 0)
            + (stage_b ? cvB.nchunks : 0)
            + (stage_c && beta != Scalar{0} ? cvC.nchunks : 0);
        std::size_t const w2c = stage_c ? cvC.nchunks : 0;
        std::fprintf(stderr,
            "nntile_gemm: path=W tA=%c tB=%c m=%lld n=%lld k=%lld "
            "batch=%lld chunks=%zu/%zu/%zu staged=%c%c%c "
            "c2w_chunks=%zu w2c_chunks=%zu staged_bytes=%zu calls=1\n",
            transA.value == TransOp::Trans ? 'T' : 'N',
            transB.value == TransOp::Trans ? 'T' : 'N',
            static_cast<long long>(m), static_cast<long long>(n),
            static_cast<long long>(k), static_cast<long long>(batch),
            cvA.nchunks, cvB.nchunks, cvC.nchunks,
            stage_a ? 'A' : '-', stage_b ? 'B' : '-',
            stage_c ? 'C' : '-', c2w, w2c,
            used_chunks * cls);
    }
}
#endif // NNTILE_USE_CUDA

//! One stderr stats line for the direct path: the all-1-chunk classic
//! dispatch and the contiguous-chain fast path share it.
void log_direct(TransOp transA, TransOp transB, Index m, Index n,
    Index k, Index batch, std::size_t ca, std::size_t cb,
    std::size_t cc) noexcept
{
    if(gemm_stats())
    {
        std::fprintf(stderr,
            "nntile_gemm: path=F tA=%c tB=%c m=%lld n=%lld k=%lld "
            "batch=%lld chunks=%zu/%zu/%zu calls=1\n",
            transA.value == TransOp::Trans ? 'T' : 'N',
            transB.value == TransOp::Trans ? 'T' : 'N',
            static_cast<long long>(m), static_cast<long long>(n),
            static_cast<long long>(k), static_cast<long long>(batch),
            ca, cb, cc);
    }
}

} // namespace

GemmStagingStats gemm_staging_stats() noexcept
{
    GemmStagingMeters &m = gemm_meters();
    GemmStagingStats s;
    s.calls = m.calls.load(std::memory_order_relaxed);
    s.contiguous = m.contiguous.load(std::memory_order_relaxed);
    s.staged = m.staged.load(std::memory_order_relaxed);
    s.staged_a = m.staged_a.load(std::memory_order_relaxed);
    s.staged_b = m.staged_b.load(std::memory_order_relaxed);
    s.staged_c = m.staged_c.load(std::memory_order_relaxed);
    s.c2w_chunks = m.c2w_chunks.load(std::memory_order_relaxed);
    s.w2c_chunks = m.w2c_chunks.load(std::memory_order_relaxed);
    s.c2w_bytes = m.c2w_bytes.load(std::memory_order_relaxed);
    s.w2c_bytes = m.w2c_bytes.load(std::memory_order_relaxed);
    return s;
}

#ifdef NNTILE_USE_CUDA // CUDA implementation requires cuBLAS
//! GEMM for contiguous matrices without padding through NNHaul buffers
template<typename T>
void Gemm<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_a = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_b = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_c = static_cast<::nnhaul::Buffer *>(buffers[2]);
    // Get cuBLAS handle and CUDA stream
    cublasHandle_t handle = ::nnhaul::cublas_handle();
    cudaStream_t stream = ::nnhaul::cuda_stream();
    cublasSetStream(handle, stream);
    // alpha and beta parameters of GEMM operation are on CPU host
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    gemm_meters().calls.fetch_add(1, std::memory_order_relaxed);
    if((buf_a->chunk_count() == 1) && (buf_b->chunk_count() == 1)
        && (buf_c->chunk_count() == 1))
    {
        gemm_meters().contiguous.fetch_add(1, std::memory_order_relaxed);
        const T *A = buf_a->get_ptr<T>();
        const T *B = buf_b->get_ptr<T>();
        T *C = buf_c->get_ptr<T>();
        // Call corresponding cuBLAS routine
        kernel::cublas::gemm<T>(
            handle,
            args->transA,
            args->transB,
            args->m,
            args->n,
            args->k,
            args->batch,
            args->alpha,
            A,
            B,
            args->beta,
            C,
            args->broadcast_a,
            args->broadcast_b
        );
        return;
    }
    if(chain_contiguous<T>(buf_a) && chain_contiguous<T>(buf_b)
        && chain_contiguous<T>(buf_c))
    {
        // Every operand is contiguous (one chunk, or a contiguous chain
        // whose chunk 0 is the classic base). Hand the pointers straight
        // to cuBLAS; nothing is copied.
        gemm_meters().contiguous.fetch_add(1, std::memory_order_relaxed);
        kernel::cublas::gemm<T>(
            handle,
            args->transA,
            args->transB,
            args->m,
            args->n,
            args->k,
            args->batch,
            args->alpha,
            buf_a->get_chunk_ptr<T>(0),
            buf_b->get_chunk_ptr<T>(0),
            args->beta,
            buf_c->get_chunk_ptr<T>(0),
            args->broadcast_a,
            args->broadcast_b
        );
        log_direct(args->transA, args->transB, args->m, args->n,
            args->k, args->batch, buf_a->chunk_count(),
            buf_b->chunk_count(), buf_c->chunk_count());
        return;
    }
    gemm_workspace_staged<T>(handle, args->transA, args->transB, args->m,
        args->n, args->k, args->batch, args->alpha, args->beta,
        args->broadcast_a, args->broadcast_b, buf_a, buf_b, buf_c);
}
#endif //NNTILE_USE_CUDA

//! Footprint for GEMM tasks that depends on transA, transB, M, N, K, batch and alpha
template<typename T>
std::uint64_t Gemm<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // In case alpha is zero, entire gemm is unnecessary so it is better to
    // give it a different footprint since gemm time will be totally different
    uint32_t hash = args->alpha == Scalar{0} ? -1 : 0;
    // Single codelet is used for all combinations of transA and transB
    hash = ::nntile::nnhaul::fnv1a(&args->transA, sizeof(args->transA), hash);
    hash = ::nntile::nnhaul::fnv1a(&args->transB, sizeof(args->transB), hash);
    // Apply hash over parameters M, N and K. This way if we swap values of M,
    // N and K total size of buffers will remain the same, but the footprint
    // will be different
    hash = ::nntile::nnhaul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::nnhaul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::nnhaul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::nnhaul::fnv1a(&args->batch, sizeof(args->batch), hash);
    return hash;
}

template<typename T>
void Gemm<std::tuple<T>>::submit(int worker_hint, const TransOp &transA, const TransOp &transB, Index m, Index n,
        Index k, Index batch, Scalar alpha, ::nnhaul::Handle & A, ::nnhaul::Handle & B, Scalar beta,
        ::nnhaul::Handle & C, int redux, bool broadcast_a, bool broadcast_b)
{
    if(redux != 0)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
    // Check that matrix sizes fit proper types for the underlying cuBLAS call
    constexpr Scalar zero = 0, one = 1;
    enum data_access_mode C_mode;
    if(beta == zero)
    {
        C_mode = ACCESS_W;
    }
    else if(beta == one)
    {
        (void)redux;
        C_mode = static_cast<data_access_mode>(
                ACCESS_RW | ACCESS_COMMUTE);
    }
    else
    {
        C_mode = ACCESS_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->transA = transA;
    args->transB = transB;
    args->m = m;
    args->n = n;
    args->k = k;
    args->batch = batch;
    args->alpha = alpha;
    args->beta = beta;
    args->broadcast_a = broadcast_a;
    args->broadcast_b = broadcast_b;
    // FLOPs calculation
    double nflops = 2 * m * n * k * batch;
    // Submit task: the body decides per operand between direct cuBLAS
    // and a workspace copy.
    (void)nflops;
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_R, &A }, { ACCESS_R, &B }, { C_mode, &C } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Gemm<std::tuple<nntile::fp64_t>>;
template class Gemm<std::tuple<nntile::fp32_t>>;
template class Gemm<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Gemm<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Gemm<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Gemm<std::tuple<nntile::bf16_t>>;
template class Gemm<std::tuple<nntile::fp16_t>>;

//! Pack of gemm operations for different types
gemm_pack_t gemm;

} // namespace nntile::nnhaul
