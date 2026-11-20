/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gemm.cc
 * GEMM operation for StarPU buffers
 *
 * @version 1.2.0
 * */

// Corresponding headers
#include "nntile/nnhaul/ops/gemm.hh"

// Standard libraries
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

// Third-party headers

// Other NNTile headers
#include "nntile/kernel/cblas.hh"
#include "nntile/kernel/cublas.hh"
#include "nntile/kernel/gemm_chunked/cuda.hh"
#include "nntile/kernel/copy/cuda.hh"
#include "nntile/nnhaul/chunk_units.hh"

#ifdef NNTILE_USE_CUDA
#include "nntile/kernel/gemm/cuda.hh"
#endif

#include <string>

namespace nntile::haul
{

//! Constructor
template<typename T>
Gemm<std::tuple<T>>::Gemm():
    codelet(
        "nntile_gemm",
#ifdef NNTILE_USE_CBLAS
        &Gemm<std::tuple<T>>::cpu,
#else
        nullptr,
#endif
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
// GEMM staging policy (user directives, 2026-10-19 "GEMM POLICY" and
// 2026-10-28 "per-operand staging"):
//
// A gemm task whose operands are chunked stages through the ONE gemm
// WORKSPACE (kernel::gemm::workspace(), default 1 GiB, plain cudaMalloc,
// allocated once per process, never freed, per-GPU, touched only by the
// compute stream -- see the kernel's invariant block). The staging
// decision is PER OPERAND (2026-10-28): an operand chains straight
// through its leading chunk when its chunk chain is contiguous —
// 1-chunk or multi-chunk — and only a SCATTERED operand pays a copy:
//
//   (i)   every SCATTERED input operand (A, B, and C when beta != 0)
//         is gathered from its chunk chain into one contiguous
//         workspace slot, the WHOLE gather being ONE fused kernel
//         launch (segments [A bytes][B bytes][C bytes?], the chunk
//         tables riding the device chunk-ptr table as one merged
//         launch table); contiguous operands pass to cuBLAS directly
//         (their leading chunk IS the classic base) and are never
//         staged; a scattered C always takes a workspace slot for the
//         OUTPUT and scatters back after the call;
//   (ii)  the GEMM on the workspace/contiguous pointers, on the
//         compute stream: [campaign2/p6-staged] flat beta=0 batch-1
//         classic calls try the p5 embedded CUTLASS flat engine FIRST
//         (gemm_chunked::flat_engine, the measured per-shape winner
//         table); every other call -- and every decline -- is the
//         EXISTING kernel::cublas::gemm entry, the exact classic
//         call;
//   (iii) the multi-chunk output C is scattered back into its chunks,
//         again ONE fused kernel launch.
//
// The gather/scatter legs are exact copies, and the cuBLAS leg sees
// the same values and layouts as the classic contiguous call, so a
// cuBLAS-legged staged call is BITWISE the classic call's; the engine
// leg (beta=0 classic shapes only) is fp32 FMA-accumulated -- the
// native family's accuracy class, carrying the p5 campaign battery
// (BETA0 poison gate, ALPHA=2 probe, bitwise run-to-run determinism).
// The single in-order compute stream makes each task's
// gather -> GEMM -> scatter chain stream-ordered, so consecutive
// tasks cannot overlap their workspace use and every task reuses the
// workspace from OFFSET 0 -- no allocator, no free list.
//
// The staged bytes (sum over multi-chunk operands of their sizes,
// padded to the fused kernel's 16-byte unit) must fit the workspace or
// the driver THROWS nnhaul::Error -- no fallback exists (rule 11). The
// NNHAUL_GEMM_WORKSPACE_BYTES env hook (read once per process, like
// the table's hooks) shrinks the workspace so a check can exercise the
// throw cheaply.
//
// What this mechanism REPLACED (deleted 2026-10-19, ~690 lines): the
// piece-2 no-k-split driver -- path B (per-batch pointer-array
// batching), path I (i-panels with straddle-row staging and the
// I-vs-S traffic model), path S (whole-operand arena staging) and the
// batch-residency walkers. The degenerate all-1-chunk case stays the
// plain cuBLAS call (the tripwire-sanctioned degenerate), and the
// flag-off classic dispatch is byte-identical to the historical one.
// (review 540 #2) The retired path-B entry cublas::gemm_array, the
// m-split probe instrument cublas::gemm_ld, and the unwired lab kernel
// gemm_native/ are DELETED from this change -- zero callers remained.
//
// The CPU body mirrors the same ONE mechanism over host scratch
// (chunkunits::ArenaCpu -- the CPU worker path, not a transport): no
// workspace, no size check, host staging of the whole operands.
// NNHAUL_GEMM_STATS=1 prints one line per chunked body (the
// launch-count evidence for the chunked-tile checks).
// ---------------------------------------------------------------------------

namespace
{

//! (owner order, 2026-10-31) The per-run GEMM staging meters: how the
//! gemm calls split between the CONTIGUOUS direct paths (the all-1-
//! chunk classic dispatch and the contiguous-chain fast-forward -- no
//! byte staged, plain cuBLAS) and the CHUNKED/STAGED workspace path,
//! with the per-operand staged counts (A/B/C) and the IN-NODE staging
//! volume the staged path moves (the gathers of scattered INPUT
//! operands into the workspace plus the scatter of a staged OUTPUT C
//! back into its chunks -- separate from the engine's cross-node COPY
//! ops the [nnhaul-counters] line reports). Relaxed atomics (the gemm
//! body is device-lane code); the summary prints once at process exit
//! under NNHAUL_GEMM_SUMMARY=1 (the [nnhaul-counters] pattern: one
//! greppable key=value line, diagnostics only, never control flow).
struct GemmStagingMeters
{
    std::atomic<std::uint64_t> calls{0};      // every CUDA gemm body
    std::atomic<std::uint64_t> contiguous{0}; // classic + fast-forward
    std::atomic<std::uint64_t> native{0};     // native chunked kernels
    std::atomic<std::uint64_t> staged{0};     // workspace-staged path
    std::atomic<std::uint64_t> staged_a{0};   // per-operand staging
    std::atomic<std::uint64_t> staged_b{0};
    std::atomic<std::uint64_t> staged_c{0};
    std::atomic<std::uint64_t> gather_bytes{0};  // inputs -> workspace
    std::atomic<std::uint64_t> scatter_bytes{0}; // workspace -> C
    //! [campaign2/p6-staged] staged calls whose GEMM leg ran the p5
    //! CUTLASS flat engine instead of cuBLAS (the rest of the staged
    //! calls keep the cuBLAS leg: beta != 0, batch > 1, foreign shape,
    //! engine-less build).
    std::atomic<std::uint64_t> staged_engine{0};
};

GemmStagingMeters &gemm_meters() noexcept
{
    static GemmStagingMeters m;
    return m;
}

//! The exit-time summary: one line, key=value, after the run's own
//! output (a static destructor runs at process exit; the driver's
//! counters are final by then).
struct GemmStagingReport
{
    ~GemmStagingReport()
    {
        char const *e = std::getenv("NNHAUL_GEMM_SUMMARY");
        if(e == nullptr || std::atoi(e) == 0)
        {
            return;
        }
        GemmStagingMeters const &m = gemm_meters();
        std::uint64_t const calls = m.calls.load(
            std::memory_order_relaxed);
        std::uint64_t const contig = m.contiguous.load(
            std::memory_order_relaxed);
        std::uint64_t const staged = m.staged.load(
            std::memory_order_relaxed);
        std::uint64_t const sa = m.staged_a.load(
            std::memory_order_relaxed);
        std::uint64_t const sb = m.staged_b.load(
            std::memory_order_relaxed);
        std::uint64_t const sc = m.staged_c.load(
            std::memory_order_relaxed);
        std::uint64_t const gb = m.gather_bytes.load(
            std::memory_order_relaxed);
        std::uint64_t const nat = m.native.load(std::memory_order_relaxed);
        std::uint64_t const sbt = m.scatter_bytes.load(
            std::memory_order_relaxed);
        std::uint64_t const eng = m.staged_engine.load(
            std::memory_order_relaxed);
        std::fprintf(stderr,
            "[gemm-stats] calls=%llu contiguous=%llu native=%llu "
            "staged=%llu "
            "staged_A=%llu staged_B=%llu staged_C=%llu "
            "gather_MiB=%.1f scatter_MiB=%.1f staged_engine=%llu\n",
            static_cast<unsigned long long>(calls),
            static_cast<unsigned long long>(contig),
            static_cast<unsigned long long>(nat),
            static_cast<unsigned long long>(staged),
            static_cast<unsigned long long>(sa),
            static_cast<unsigned long long>(sb),
            static_cast<unsigned long long>(sc),
            static_cast<double>(gb) / (1024.0 * 1024.0),
            static_cast<double>(sbt) / (1024.0 * 1024.0),
            static_cast<unsigned long long>(eng));
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

//! The contiguous-chain fact (user directives, 2026-10-28): a
//! multi-chunk tile whose chunks sit at consecutive class-stride
//! addresses IS the classic contiguous buffer — its chunk chain is a
//! slice chain of one flat array. Pointer adjacency is checked, never
//! chunk-id arithmetic (the fact must hold however the pool carved
//! the arena). One compare per chunk, O(chunks) and tiny. The staging
//! decision is PER OPERAND: an input chains straight through its
//! leading chunk when contiguous and only a SCATTERED operand pays
//! the workspace copy.
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
        if (p != base + i * chunkunits::kChunkClassBytes)
        {
            return false;
        }
    }
    return true;
}

#ifdef NNTILE_USE_CUDA
//! Kill switch for the native kernel family (default on): read once per
//! process, the table-hook pattern. The staged path is the fallback.
bool gemm_native_enabled() noexcept
{
    static bool const cached = [] {
        char const *e = std::getenv("NNHAUL_GEMM_NATIVE");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return cached;
}

//! The NATIVE chunked-kernel driver: kernel::gemm_chunked consumes the
//! operands' chunk tables IN PLACE (or a flat base for a contiguous
//! chain) -- zero gather/scatter bytes, one kernel launch whose tables
//! ride the chunk-ptr table as one merged upload. fp32 only (the
//! campaign's dtype; every other repr type keeps the staged path).
//! Returns false without side effects when the picker does not own the
//! shape (the caller falls through to the workspace-staged path).
template<typename T>
bool gemm_kernel_native(TransOp transA, TransOp transB, Index m, Index n,
    Index k, Index batch, Scalar alpha, Scalar beta, bool broadcast_a,
    bool broadcast_b, ::nnhaul::Buffer *bufA, ::nnhaul::Buffer *bufB,
    ::nnhaul::Buffer *bufC, cudaStream_t stream, bool force = false)
{
    if constexpr(std::is_same_v<T, fp32_t>)
    {
        if(!gemm_native_enabled())
        {
            return false;
        }
        using kernel::gemm_chunked::Operand;
        using kernel::gemm_chunked::Op;
        Op const ta = transA.value == TransOp::NoTrans ? Op::NoTrans
                                                       : Op::Trans;
        Op const tb = transB.value == TransOp::NoTrans ? Op::NoTrans
                                                       : Op::Trans;
        if(!kernel::gemm_chunked::supports(ta, tb, m, n, k, batch))
        {
            return false;
        }
        chunkunits::ChunkView<T> const cv_a(bufA);
        chunkunits::ChunkView<T> const cv_b(bufB);
        chunkunits::ChunkView<T> const cv_c(bufC);
        //! A contiguous chain IS the flat buffer: chunk 0 is the base
        ///  and the kernel adds element offsets with no indirection.
        ///  [rescue, 2026-11-12] The force arg (the workspace-overflow
        ///  rescue's escape hatch) ALWAYS runs in table mode -- a
        ///  contiguous multi-chunk chain uploads its chunk-pointer
        ///  table like a scattered chain, so the rescue launch never
        ///  depends on how the chunks are laid. A 1-chunk buffer has
        ///  no chain at all (it IS one chunk, the sentinel degenerate)
        ///  -- its chunk-0 base is the only form it has.
        bool const flat_a = !cv_a.real
            || (!force && chain_contiguous<T>(bufA));
        bool const flat_b = !cv_b.real
            || (!force && chain_contiguous<T>(bufB));
        bool const flat_c = !cv_c.real
            || (!force && chain_contiguous<T>(bufC));
        chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
        tabs.clear();
        std::size_t off_a = 0, off_b = 0, off_c = 0;
        if(!flat_a)
        {
            off_a = tabs.add(cv_a);
        }
        if(!flat_b)
        {
            off_b = tabs.add(cv_b);
        }
        if(!flat_c)
        {
            off_c = tabs.add(cv_c);
        }
        chunkunits::ChunkPtrTable &table = chunkunits::chunk_ptr_table();
        //! [owner directive, 2026-11-11] An ALL-FLAT call (every operand
        ///  contiguous or single-chunk — the forced regime's normal
        ///  case) has ZERO tables to upload: the table serves nothing,
        ///  upload_tables would return its empty-service failure and a
        ///  servable native launch would be refused. Skip the table
        ///  entirely — the Operands below use chunk-0 bases with
        ///  shift=0 and need no table view. table.end is likewise
        ///  conditional below (a never-begun table must not be ended).
        bool const any_table = !flat_a || !flat_b || !flat_c;
        void const *table_base = nullptr;
        if(any_table)
        {
            if(!table.begin(stream))
            {
                return false;
            }
            table_base = table.upload_tables(tabs);
            if(table_base == nullptr)
            {
                return false;
            }
        }
        Operand const oa{flat_a ? reinterpret_cast<void const *>(
                                     bufA->get_chunk_ptr<T>(0))
                                : reinterpret_cast<void const *>(
                                     const_cast<float const **>(
                                         chunkunits::ChunkPtrTable::
                                             table_view<float const *>(
                                                 table_base, off_a))),
            flat_a ? 0 : cv_a.shift};
        Operand const ob{flat_b ? reinterpret_cast<void const *>(
                                     bufB->get_chunk_ptr<T>(0))
                                : reinterpret_cast<void const *>(
                                     const_cast<float const **>(
                                         chunkunits::ChunkPtrTable::
                                             table_view<float const *>(
                                                 table_base, off_b))),
            flat_b ? 0 : cv_b.shift};
        Operand const oc{flat_c ? reinterpret_cast<void *>(
                                     bufC->get_chunk_ptr<T>(0))
                                : reinterpret_cast<void *>(
                                     const_cast<float **>(
                                         chunkunits::ChunkPtrTable::
                                             table_view<float *>(
                                                 table_base, off_c))),
            flat_c ? 0 : cv_c.shift};
        long long const sa =
            broadcast_a ? 0LL : static_cast<long long>(m) * k;
        long long const sb =
            broadcast_b ? 0LL : static_cast<long long>(k) * n;
        long long const sc = static_cast<long long>(m) * n;
        bool const launched = force
            ? kernel::gemm_chunked::launch_forced(ta, tb,
                static_cast<int>(m), static_cast<int>(n),
                static_cast<int>(k), static_cast<int>(batch),
                static_cast<float>(alpha), oa, ob, static_cast<float>(beta),
                oc, sa, sb, sc, stream)
            : kernel::gemm_chunked::launch(ta, tb, static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k),
                static_cast<int>(batch), static_cast<float>(alpha), oa, ob,
                static_cast<float>(beta), oc, sa, sb, sc, stream);
        if(any_table)
        {
            table.end(stream);
        }
        if(launched)
        {
            gemm_meters().native.fetch_add(1, std::memory_order_relaxed);
            note_gemm_kernel_native();
        }
        return launched;
    }
    else
    {
        (void)stream;
        return false;
    }
}

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
    // PER-OPERAND staging (user directive, 2026-10-28): an operand
    // stages iff its OWN chain is scattered; a contiguous chain —
    // 1-chunk or multi-chunk — passes through its leading chunk with
    // no copy. A scattered C still takes a workspace slot (the output
    // needs a contiguous home; it is gathered first only when
    // beta != 0 reads it).
    //! [owner directive, 2026-11-11, vocabulary corrected] Under
    ///  FORCE_CHUNKED this body owns EVERY gemm: all inputs are copied
    ///  into the workspace, cuBLAS runs on the copies, and the output
    ///  is copied back to the chunks -- a contiguous chain stages
    ///  exactly like a scattered one (the same gather->cuBLAS->
    ///  scatter sequence no matter how the chunks are laid). The
    ///  flag-off per-operand rule stays: an operand stages iff its
    ///  own chain is scattered.
    bool const forced = chunkunits::force_chunked_kernels();
    bool const stage_a = forced || !chain_contiguous<T>(bufA);
    bool const stage_b = forced || !chain_contiguous<T>(bufB);
    bool const stage_c = forced || !chain_contiguous<T>(bufC);
    // The staging meters (owner order, 2026-10-31): one staged call,
    // the per-operand staging flags, and the in-node staging volume --
    // the gathers of every staged INPUT operand (C reads its slot only
    // when beta != 0) plus the scatter of the staged OUTPUT back into
    // its chunks.
    {
        GemmStagingMeters &m = gemm_meters();
        m.staged.fetch_add(1, std::memory_order_relaxed);
        if(stage_a)
        {
            m.staged_a.fetch_add(1, std::memory_order_relaxed);
            m.gather_bytes.fetch_add(bufA->nbytes(),
                std::memory_order_relaxed);
        }
        if(stage_b)
        {
            m.staged_b.fetch_add(1, std::memory_order_relaxed);
            m.gather_bytes.fetch_add(bufB->nbytes(),
                std::memory_order_relaxed);
        }
        if(stage_c)
        {
            m.staged_c.fetch_add(1, std::memory_order_relaxed);
            m.scatter_bytes.fetch_add(bufC->nbytes(),
                std::memory_order_relaxed);
            if(beta != Scalar{0})
            {
                m.gather_bytes.fetch_add(bufC->nbytes(),
                    std::memory_order_relaxed);
            }
        }
    }
    // (gemm critic C1, 2026-10-19) The scatter rewrites C's WHOLE flat
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
    // (1, user directive 2026-10-19) The workspace is VIRTUALLY
    // CHUNKED at the class size and every staged operand is placed
    // CHUNK-ALIGNED: operand i occupies whole virtual chunks
    // [k_i, k_i + ceil(nbytes_i/class)), so the workspace side of every
    // copy is an ordinary chunk table of the same class as the operand
    // -- kernel::copy::chunked runs AS IS on both sides (no sentinel, no
    // workspace special-casing; the padding inside each operand's last
    // virtual chunk is workspace-internal and never written past
    // nbytes). The overflow check is chunk-granular.
    kernel::gemm::Workspace &ws = kernel::gemm::workspace();
    std::size_t const cls = chunkunits::kChunkClassBytes;
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
    // (2) one merged chunk-ptr table for every staged operand, on the
    // table (the event-ordered control-stream upload lives inside
    // upload_tables; only table bytes ride it -- never workspace
    // memory).
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
    //! [owner directive, 2026-11-11] An ALL-CONTIGUOUS call (nothing
    ///  stages — the forced regime's lm_head class: contiguous operands
    ///  the native family cannot serve) has ZERO tables to upload and
    ///  needs neither table nor workspace: the classic call below runs
    ///  directly on the operands' chunk-0 bases. This path was
    ///  unreachable before FORCE_CHUNKED gated the contiguous
    ///  fast-forward; the empty upload's rule-11 abort is not a
    ///  servable answer for it.
    bool const any_staged = stage_a || stage_b || stage_c;
    void const *table_base = nullptr;
    if(any_staged)
    {
        if(!table.begin(stream))
        {
            std::fprintf(stderr,
                "nntile_gemm: chunk-ptr table init failed (rule 11 -- no "
                "fallback exists)\n");
            std::abort();
        }
    }
    // (3, user directive 2026-10-19: reuse the SCALABLE chunk-table
    // kernel) The staging IS kernel::copy::chunked -- the proven
    // one-launch chunked->chunked flat copy every other op uses. Each
    // staged operand's workspace slot rides the SAME merged table upload
    // as a 1-entry table (the starting point + element count interface;
    // the 1-chunk sentinel shift makes the whole workspace one chunk).
    // The earlier gemm-local fused kernel (and its StagePlan parameter
    // that once spilled to local memory) is deleted.
    // each staged operand's workspace side: its whole virtual-chunk
    // table [ws + k_i*cls, ws + (k_i+1)*cls, ...] -- ordinary chunk
    // pointers of the class, appended to the same merged upload
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
    // ONE merged upload: operand tables + workspace slots (staged
    // operands only; the all-contiguous degenerate skips the table).
    if(any_staged)
    {
        table_base = table.upload_tables(tabs);
        if(table_base == nullptr)
        {
            std::fprintf(stderr,
                "nntile_gemm: chunk-ptr table cannot serve the launch "
                "(rule 11 -- no fallback exists)\n");
            std::abort();
        }
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
        g.elems_src = Index(chunkunits::kChunkClassBytes);
        g.elems_dst = Index(chunkunits::kChunkClassBytes);
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
    // (ii) the classic call itself, verbatim shape, on the
    // staged/contiguous pointers: same values, same layouts as the
    // flag-off dispatch.
    T const *pa = stage_a
        ? reinterpret_cast<T const *>(ws.base + seg_chunk[0] * cls)
        : bufA->get_chunk_ptr<T>(0);
    T const *pb = stage_b
        ? reinterpret_cast<T const *>(ws.base + seg_chunk[1] * cls)
        : bufB->get_chunk_ptr<T>(0);
    T *pc = stage_c
        ? reinterpret_cast<T *>(ws.base + seg_chunk[2] * cls)
        : bufC->get_chunk_ptr<T>(0);
    // [campaign2/p6-staged] The GEMM leg first tries
    // the p5 embedded CUTLASS flat engine (gemm_chunked::flat_engine —
    // the p5-cutlass round-1 winner table, ids 30-38, WINNERS ONLY:
    // the 77 of 288 classic rows whose measured p5 flat ratio is at
    // or under 0.995; the wash/loss rows keep cuBLAS so the leg never
    // regresses) on EXACTLY the flat beta=0 batch-1 classic calls.
    // Everything else falls through UNTOUCHED
    // to the cuBLAS call: beta != 0 (the table is beta=0-measured),
    // batch > 1 (broadcast included), non-classic shapes, and the
    // engine-less build (flat_engine is a constant false without the
    // CUTLASS headers -- no __has_include drift). The gather/scatter
    // legs are unchanged: the engine reads the same gathered A/B and
    // overwrites all m*n elements of C's workspace slot without
    // reading it (the p5 BETA0 poison-gate property), so the scatter
    // still rewrites C's whole flat byte range. Numerics: the leg is
    // fp32 FMA-accumulated (the native family's accuracy class), not
    // bitwise cuBLAS; the engines carry the campaign battery (BETA0
    // poison, ALPHA=2, bitwise run-to-run determinism -- p5 r1).
    bool engine_leg = false;
    if constexpr(std::is_same_v<T, fp32_t>)
    {
        //! [owner directive, 2026-11-12, FINAL] Under FORCE the staged
        ///  body is gather -> cuBLAS -> scatter for EVERY call: the
        ///  winners-table CUTLASS leg is a flag-off optimization only
        ///  (staged means cuBLAS by definition).
        if(!forced && batch == 1)
        {
            using kernel::gemm_chunked::Op;
            Op const ta = transA.value == TransOp::NoTrans
                ? Op::NoTrans
                : Op::Trans;
            Op const tb = transB.value == TransOp::NoTrans
                ? Op::NoTrans
                : Op::Trans;
            engine_leg = kernel::gemm_chunked::flat_engine(ta, tb,
                static_cast<int>(m), static_cast<int>(n),
                static_cast<int>(k), static_cast<float>(alpha), pa, pb,
                static_cast<float>(beta), pc, stream);
        }
    }
    if(engine_leg)
    {
        gemm_meters().staged_engine.fetch_add(1,
            std::memory_order_relaxed);
    }
    else
    {
        kernel::cublas::gemm<T>(handle, transA, transB, m, n, k, batch,
            alpha, pa, pb, beta, pc, broadcast_a, broadcast_b);
    }
    if(stage_c)
    {
        kernel::copy::chunked(
            stream,
            Index(bufC->nbytes()),
            table.table_view<const unsigned char *>(table_base, ws_off[2]),
            Index(seg_nchunk[2]),
            table.table_view<unsigned char *>(table_base, slot_off[2]),
            Index(bufC->chunk_count()),
            Index(chunkunits::kChunkClassBytes),
            Index(chunkunits::kChunkClassBytes));
    }
    if(any_staged)
    {
        table.end(stream);
    }
    if(gemm_stats())
    {
        std::fprintf(stderr,
            "nntile_gemm: path=W tA=%c tB=%c m=%lld n=%lld k=%lld "
            "batch=%lld chunks=%zu/%zu/%zu staged=%c%c%c "
            "staged_bytes=%zu engine_leg=%c calls=1\n",
            transA.value == TransOp::Trans ? 'T' : 'N',
            transB.value == TransOp::Trans ? 'T' : 'N',
            static_cast<long long>(m), static_cast<long long>(n),
            static_cast<long long>(k), static_cast<long long>(batch),
            cvA.nchunks, cvB.nchunks, cvC.nchunks,
            stage_a ? 'A' : '-', stage_b ? 'B' : '-',
            stage_c ? 'C' : '-',
            used_chunks * cls, engine_leg ? 'y' : 'n');
    }
}
#endif // NNTILE_USE_CUDA

//! One stderr stats line for the direct (fast-forward) body: the
//! all-1-chunk degenerate and the contiguous-chain fast path share it.
void log_degenerate(TransOp transA, TransOp transB, Index m, Index n,
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

#ifdef NNTILE_USE_CBLAS // CPU implementation requires CBLAS
//! GEMM for contiguous matrices without padding through StarPU buffers
template<typename T>
void Gemm<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_a = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_b = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_c = static_cast<::nnhaul::Buffer *>(buffers[2]);
    // Launch kernel
    if constexpr (kernel::cblas::gemm_is_supported<T>)
    {
        if((buf_a->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_b->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
            && (buf_c->chunk_count() == 1 && !chunkunits::force_chunked_kernels()))
        {
        // Classic contiguous dispatch: illegal under the force flag
        // (the tripwire proves it never happens in a flag-on run).
        chunkunits::classic_dispatch_tripwire("nntile_gemm", false);
            const T *A = buf_a->get_ptr<T>();
            const T *B = buf_b->get_ptr<T>();
            T *C = buf_c->get_ptr<T>();
            kernel::cblas::gemm<T>(
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
        try
        {
            if(buf_a->chunk_count() == 1 && buf_b->chunk_count() == 1
                && buf_c->chunk_count() == 1)
            {
                // Degenerate all-1-chunk case: the plain classic call.
                kernel::cblas::gemm<T>(
                    args->transA,
                    args->transB,
                    args->m,
                    args->n,
                    args->k,
                    args->batch,
                    args->alpha,
                    buf_a->get_ptr<T>(),
                    buf_b->get_ptr<T>(),
                    args->beta,
                    buf_c->get_ptr<T>(),
                    args->broadcast_a,
                    args->broadcast_b
                );
            }
            else
            {
                // The CPU mirror of the staging policy: whole
                // multi-chunk operands into host scratch, ONE
                // classic-shaped call, C scattered back.
                chunkunits::ChunkView<T> const cv_a(buf_a);
                chunkunits::ChunkView<T> const cv_b(buf_b);
                chunkunits::ChunkView<T> const cv_c(buf_c);
                bool const stage_a = cv_a.nchunks > 1;
                bool const stage_b = cv_b.nchunks > 1;
                bool const stage_c = cv_c.nchunks > 1;
                chunkunits::ArenaCpu<T> arena;
                T *pa = stage_a ? arena.alloc(cv_a.total_elems) : nullptr;
                T *pb = stage_b ? arena.alloc(cv_b.total_elems) : nullptr;
                // C's slot is only READ when beta != 0 (BLAS leaves C
                // unread when beta is zero), but the scatter needs it
                // whenever C is chunked.
                T *pc = stage_c ? arena.alloc(cv_c.total_elems) : nullptr;
                auto gather_whole =
                    [&arena](chunkunits::ChunkView<T> const &cv, T *dst) {
                        for(std::size_t c = 0; c < cv.nchunks; ++c)
                        {
                            std::size_t const len =
                                cv.chunk_end(c) - cv.chunk_begin(c);
                            if(len > 0)
                            {
                                arena.copy(dst + cv.chunk_begin(c),
                                    cv.ptr(c, 0), len);
                            }
                        }
                    };
                if(stage_a)
                {
                    gather_whole(cv_a, pa);
                }
                if(stage_b)
                {
                    gather_whole(cv_b, pb);
                }
                if(stage_c && args->beta != Scalar{0})
                {
                    gather_whole(cv_c, pc);
                }
                kernel::cblas::gemm<T>(
                    args->transA,
                    args->transB,
                    args->m,
                    args->n,
                    args->k,
                    args->batch,
                    args->alpha,
                    stage_a ? pa : buf_a->get_ptr<T>(),
                    stage_b ? pb : buf_b->get_ptr<T>(),
                    args->beta,
                    stage_c ? pc : buf_c->get_ptr<T>(),
                    args->broadcast_a,
                    args->broadcast_b
                );
                if(stage_c)
                {
                    for(std::size_t c = 0; c < cv_c.nchunks; ++c)
                    {
                        std::size_t const len =
                            cv_c.chunk_end(c) - cv_c.chunk_begin(c);
                        if(len > 0)
                        {
                            arena.copy(cv_c.ptr(c, 0),
                                pc + cv_c.chunk_begin(c), len);
                        }
                    }
                }
            }
            note_gemm_native_chunked();
        }
        catch(...)
        {
            // A noexcept body must not throw; a failed scratch allocation
            // leaves the result unwritten (same failure class as a failed
            // kernel launch).
            std::fprintf(stderr,
                    "nntile_gemm: cpu chunked driver aborted after an "
                    "allocation failure; C left unwritten\n");
        }
    }
}
#endif // NNTILE_USE_CBLAS

#ifdef NNTILE_USE_CUDA // CUDA implementation requires cuBLAS
//! GEMM for contiguous matrices without padding through StarPU buffers
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
    if((buf_a->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_b->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
        && (buf_c->chunk_count() == 1 && !chunkunits::force_chunked_kernels()))
    {
        // Classic contiguous dispatch: illegal under the force flag
        // (the tripwire proves it never happens in a flag-on run).
        chunkunits::classic_dispatch_tripwire("nntile_gemm", false);
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
    //! [owner directive, 2026-11-11] FORCE_CHUNKED: the contiguous-chain
    ///  fast-forward is OFF under the flag — the call falls through to
    ///  the native chunked family (gemm_kernel_native with force=true,
    ///  the union router) instead of cuBLAS-on-chunk-0.
    if(!chunkunits::force_chunked_kernels()
        && chain_contiguous<T>(buf_a) && chain_contiguous<T>(buf_b)
        && chain_contiguous<T>(buf_c))
    {
        // The fast-forward body (user directive, 2026-10-28): every
        // operand is EITHER the all-1-chunk degenerate OR a
        // contiguous multi-chunk chain — chunk 0's pointer IS the
        // classic contiguous base, the argument-identical cuBLAS call
        // runs on it, and no byte is staged. A flag-on run of a
        // contiguous graph stays bitwise the flag-off one.
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
        log_degenerate(args->transA, args->transB, args->m, args->n,
            args->k, args->batch, buf_a->chunk_count(),
            buf_b->chunk_count(), buf_c->chunk_count());
        note_gemm_native_chunked();
        return;
    }
    //! [owner directive, 2026-11-12, FINAL] FORCE means a GEMM is
    ///  ALWAYS staged, no matter if the data is contiguous or not:
    ///  every input is copied into the workspace and the output is
    ///  copied back to the chunks. The native attempt is therefore
    ///  skipped entirely under the flag (it stays the flag-off fast
    ///  path, and the workspace-overflow rescue's escape hatch below).
    if(!chunkunits::force_chunked_kernels()
        && gemm_kernel_native<T>(args->transA, args->transB, args->m,
            args->n, args->k, args->batch, args->alpha, args->beta,
            args->broadcast_a, args->broadcast_b, buf_a, buf_b, buf_c,
            stream))
    {
        note_gemm_native_chunked();
        return;
    }
    // Workspace-overflow rescue: a call whose staged bytes exceed the
    // gemm workspace would THROW (the staged path's rule-11 error, no
    // fallback) — the native kernel has no such cap, so it owns the
    // call whatever the ownership policy says.
    {
        // Chunk-class rounding, IDENTICAL to gemm_workspace_staged's
        // demand ((nbytes+cls-1)/cls*cls per staged operand): a 16-byte
        // pad under-approximates, and a forced call in the gap would
        // hit the staged path's rule-11 throw instead of the rescue.
        std::size_t const cls = chunkunits::kChunkClassBytes;
        auto const padded = [cls](std::size_t nbytes) {
            return (nbytes + cls - 1) / cls * cls;
        };
        //! Under FORCE_CHUNKED every operand stages, so every operand
        ///  counts toward the workspace demand -- the rescue gate must
        ///  see the same need the staged path will compute, or a
        ///  too-large forced call would throw instead of rescuing.
        bool const forced_rescue = chunkunits::force_chunked_kernels();
        std::size_t need = 0;
        if(forced_rescue || !chain_contiguous<T>(buf_a))
        {
            need += padded(buf_a->nbytes());
        }
        if(forced_rescue || !chain_contiguous<T>(buf_b))
        {
            need += padded(buf_b->nbytes());
        }
        if(forced_rescue || !chain_contiguous<T>(buf_c))
        {
            need += padded(buf_c->nbytes());
        }
        if(need > 0 && need > kernel::gemm::workspace().bytes
            && gemm_kernel_native<T>(args->transA, args->transB, args->m,
                args->n, args->k, args->batch, args->alpha, args->beta,
                args->broadcast_a, args->broadcast_b, buf_a, buf_b, buf_c,
                stream, true))
        {
            note_gemm_kernel_native();
            note_gemm_native_chunked();
            return;
        }
    }
    gemm_workspace_staged<T>(handle, args->transA, args->transB, args->m,
        args->n, args->k, args->batch, args->alpha, args->beta,
        args->broadcast_a, args->broadcast_b, buf_a, buf_b, buf_c);
    note_gemm_native_chunked();
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
    hash = ::nntile::haul::fnv1a(&args->transA, sizeof(args->transA), hash);
    hash = ::nntile::haul::fnv1a(&args->transB, sizeof(args->transB), hash);
    // Apply hash over parameters M, N and K. This way if we swap values of M,
    // N and K total size of buffers will remain the same, but the footprint
    // will be different
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->batch, sizeof(args->batch), hash);
    return hash;
}

template<typename T>
void Gemm<std::tuple<T>>::submit(int starpu_worker_hint, const TransOp &transA, const TransOp &transB, Index m, Index n,
        Index k, Index batch, Scalar alpha, ::nnhaul::Handle & A, ::nnhaul::Handle & B, Scalar beta,
        ::nnhaul::Handle & C, int redux, bool broadcast_a, bool broadcast_b)
{
    if(redux != 0)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
    // Check that matrix sizes fit proper types for underlying CBLAS
#ifdef NNTILE_USE_CBLAS
    if(static_cast<CBLAS_INT>(m) != m)
    {
        throw std::runtime_error("GEMM size M does not fit CBLAS_INT");
    }
    if(static_cast<CBLAS_INT>(n) != n)
    {
        throw std::runtime_error("GEMM size N does not fit CBLAS_INT");
    }
    if(static_cast<CBLAS_INT>(k) != k)
    {
        throw std::runtime_error("GEMM size K does not fit CBLAS_INT");
    }
#endif // NNTILE_USE_CBLAS
    constexpr Scalar zero = 0, one = 1;
    enum starpu_data_access_mode C_mode;
    if(beta == zero)
    {
        C_mode = STARPU_W;
    }
    else if(beta == one)
    {
        if(redux != 0)
        {
            C_mode = STARPU_REDUX;
        }
        else
        {
            C_mode = static_cast<starpu_data_access_mode>(
                STARPU_RW | STARPU_COMMUTE);
        }
    }
    else
    {
        C_mode = STARPU_RW;
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
    // Submit task: the BODY picks the classic contiguous call, the
    // all-1-chunk degenerate, or the workspace-staging mechanism.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &A }, { STARPU_R, &B }, { C_mode, &C } }, args, sizeof(*args));
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

} // namespace nntile::haul
