/*! @copyright (c) 2026-present Skolkovo Institute of Science and
 *                              Technology (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file include/nnhaul/nnhaul.hh
 * Public API: handles, codelets, task insertion, host acquires, and
 * runtime init/shutdown.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <cstddef>
#include <cstdint>

#include <nnhaul/defs.h>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

//! @namespace nnhaul
/*! NNHaul: a multi-device rolling runtime for task-based execution.
 * */
namespace nnhaul
{

/*! Access mode a task declares for one of its buffers.
 * */
enum class Access
{
    Read,     //!< Task only reads the buffer's current version.
    Write,    //!< Task produces the next version (reads nothing).
    ReadWrite //!< Task reads the current version, then produces the
              //!< next one in the same buffer.
};

/*! Compute kind of a worker or task — a DIAGNOSTIC label only.
 *
 * CPU kernels are first-class TILE_OPs on the host node's execution
 * lane; a multi-implementation codelet's TILE_OP node selects the body
 * that runs (host node -> the CPU body, a device node -> that
 * device's CUDA body). The kind never gates execution; it labels task
 * views and worker diagnostics.
 * */
enum class Kind
{
    Cpu, //!< The host node's body class.
    Cuda //!< A CUDA device node's body class.
};

/*! Memory-node identifier: devices 0..k-1 in ascending CUDA ordinal,
 * host k. Public because the planner interface names nodes. */
using NodeId = std::uint8_t;

/*! Worker identifier. A WorkerId IS a NodeId by identity: valid ids at submit are 0..k — devices 0..k-1 (ascending
 * CUDA ordinal), host k. */
using WorkerId = int;
inline constexpr WorkerId worker_unspecified = -1; //!< "No worker".
inline constexpr WorkerId cuda_worker = 0; //!< Device node 0.

class Handle;
class Codelet;
class Acquired;
struct HandleImpl;

/*! One named buffer use inside an insert() call: the handle plus the
 * access mode the submitted task declares for it.
 * */
struct BufferUse
{
    Handle *handle; //!< The data handle.
    Access access;  //!< How the task uses it.
};

/*! Runtime error: bad API use, dead handles, cap violations, or a
 * failed runtime component (transfer, launch).
 * */
class Error : public std::runtime_error
{
  public:
    /*! Wraps a human-readable message. */
    explicit Error(std::string const &message)
        : std::runtime_error(message)
    {
    }
};

/*! A device buffer handed to a codelet kernel.
 *
 * Kernel bodies receive an array of Buffer pointers and cast each to
 * their element type via get_ptr<T>() or get_chunk_ptr<T>(i).
 * capacity() is what the pool carved; nbytes() is the payload the
 * handle owns.
 *
 * Every handle's tile decomposes, by the one chunk class, into
 * chunk_count() whole-class blocks: n = ceil(nbytes /
 * chunk_class_bytes()), every block exactly the class, the last block's
 * logical payload possibly shorter (the intra-chunk tail — wasted
 * bytes inside a full class block). get_ptr<T>() works
 * exactly when chunk_count() == 1 (a tile at or below the class) — it
 * throws on a multi-chunk buffer so a kernel can never silently read
 * the first chunk as if it were the whole tile; get_chunk_ptr<T>(i)
 * returns chunk i and chunk_bytes(i) its logical payload.
 * */
struct Buffer
{
    /*! Typed view of the buffer's device memory (single-chunk tiles
     * only: chunk_count() == 1). */
    template <class T> T *get_ptr() const
    {
        if (chunk_count_ > 1)
        {
            throw Error("get_ptr on a chunked buffer: use get_chunk_ptr");
        }
        return static_cast<T *>(ptr_);
    }

    /*! Typed view of chunk i's device memory (chunk 0 is the whole
     * tile for a single-chunk buffer). */
    template <class T> T *get_chunk_ptr(std::size_t i) const
    {
        if (i >= chunk_count_)
        {
            throw Error("chunk index out of range");
        }
        return static_cast<T *>(
            chunk_ptrs_ != nullptr ? chunk_ptrs_[i] : ptr_);
    }

    /*! True when the runtime holds a current host-node replica of
     * this buffer's tile chunk (the binding table's host segment;
     * single-chunk buffers, which is everything a codelet can address
     * with one pointer). Present means actual: a codelet may read it
     * on the host with no D2H and no stream synchronization. */
    bool has_host_copy() const { return host_ptr_ != nullptr; }
    /*! The host replica as a typed pointer (valid only with
     * has_host_copy()). */
    template <class T> T const *get_host_ptr() const
    {
        if (host_ptr_ == nullptr)
        {
            throw Error("get_host_ptr without a host copy");
        }
        return static_cast<T const *>(host_ptr_);
    }

    /*! Bytes the pool carved (sum over chunks). */
    std::size_t capacity() const { return capacity_; }
    /*! Bytes the handle owns (the logical payload size). */
    std::size_t nbytes() const { return nbytes_; }
    /*! Number of chunks: ceil(nbytes / chunk_class_bytes()), at least 1
     * for a nonzero tile. */
    std::size_t chunk_count() const { return chunk_count_; }
    /*! Logical payload bytes of chunk i: the full chunk class, except
     * the last chunk's payload, which may be shorter (the
     * intra-chunk tail). */
    std::size_t chunk_bytes(std::size_t i) const
    {
        if (i >= chunk_count_)
        {
            throw Error("chunk index out of range");
        }
        if (chunk_bytes_ == 0)
        {
            return nbytes_; // single-chunk: one exact-size chunk
        }
        std::size_t const off = i * chunk_bytes_;
        return nbytes_ - off < chunk_bytes_ ? nbytes_ - off : chunk_bytes_;
    }

    /*! The contiguous-chain fact (carried as engine surface): true when
     * the tile's whole chunk chain sits at consecutive class-stride
     * addresses — the chain IS one flat classic buffer (trivially
     * true at one chunk). Pointer adjacency is checked, never chunk-id
     * arithmetic, so the fact holds however the pool carved the
     * arena. O(chunks), one compare each; a kernel taking a plain
     * pointer over the whole tile may do so exactly when this holds
     * (the classic call on chunk 0's pointer is then bitwise the
     * chunked execution, no byte staged). */
    bool chain_contiguous() const
    {
        if (chunk_count_ <= 1)
        {
            return true;
        }
        void const *const *ptrs =
            chunk_ptrs_ != nullptr ? chunk_ptrs_ : nullptr;
        if (ptrs == nullptr)
        {
            return false; // no per-chunk table: never claim the fact
        }
        unsigned char const *base =
            static_cast<unsigned char const *>(ptrs[0]);
        std::size_t const cls = chunk_bytes(0);
        for (std::size_t i = 1; i < chunk_count_; ++i)
        {
            if (static_cast<unsigned char const *>(ptrs[i]) !=
                base + i * cls)
            {
                return false;
            }
        }
        return true;
    }

    // Internal: runtime implementation detail (opaque outside src/).
    void *ptr_ = nullptr;  // chunk 0's block (the whole tile when
                           // chunk_count() == 1)
    std::size_t capacity_ = 0; // sum of the chunk block capacities
    std::size_t nbytes_ = 0;
    // Internal: chunk table snapshot filled by the worker at launch
    // (null for single-chunk buffers — ptr_ is the one chunk).
    void *const *chunk_ptrs_ = nullptr;
    std::size_t chunk_count_ = 1;
    std::size_t chunk_bytes_ = 0; // 0 = single exact-size chunk
    // Internal: the current host-node replica (single-chunk buffers
    // only); null when there is none.
    void const *host_ptr_ = nullptr;
};

/*! A codelet: a named task type carrying its kernel bodies.
 *
 * A Codelet object outlives every task inserted with it (register it
 * once next to your other statics, like any codelet). The TWO
 * bodies are ONE task symbol's two implementations: a task whose codelet has a CUDA body is runnable on
 * every device node, one whose codelet has a CPU body is runnable on
 * the host node, and the TILE_OP's node selects the body — the host
 * node dispatches cpu_, a device node dispatches cuda_ loaded in that
 * device's context at init. A codelet with neither body is runnable
 * nowhere; every TILE_OP naming it is the check-time fault
 * task-not-runnable.
 *
 * The scalar-argument blob (cl_args) of a task is a REGISTRATION-TIME
 * fact: it is copied at submit and travels with the task symbol;
 * a workload whose scalars vary per op must either multiply task
 * symbols (one Codelet each) or route the varying values through
 * operand chunk bytes as planned data.
 * */
class Codelet
{
  public:
    /*! Kernel body: receives the buffer array and the cl_args copy.
     * Must not throw. Runs on the node's execution lane inside the
     * codelet window, with cuda_stream()/cublas_handle()/
     * cudnn_handle()/device_index() naming the RUNNING node. */
    using Kernel = void (*)(void **buffers, void *cl_args) noexcept;
    /*! Footprint key function: hashes cl_args into the diagnostic
     * history key. May be null. */
    using FootprintFn = std::uint64_t (*)(
        void const *cl_args, std::size_t cl_arg_size) noexcept;

    /*! Registers the codelet and assigns its identity id. */
    Codelet(std::string name, Kernel cpu, Kernel cuda,
        FootprintFn footprint);

    Codelet(Codelet const &) = delete;
    Codelet &operator=(Codelet const &) = delete;

    /*! Unique id assigned at construction (the task symbol's TaskId
     * and a footprint key input). */
    std::uint32_t id() const;
    /*! True when a CUDA kernel body is present (runnable on every
     * device node). */
    bool has_cuda() const;
    /*! True when a CPU kernel body is present (runnable on the host
     * node). */
    bool has_cpu() const;
    /*! The codelet's diagnostic name. */
    std::string const &name() const;

    // Internal: runtime implementation detail (opaque outside src/).
    std::string name_;
    Kernel cpu_ = nullptr;
    Kernel cuda_ = nullptr;
    FootprintFn footprint_ = nullptr;
    std::uint32_t id_ = 0;
};

/*! RAII pin of a handle's host-node bytes, returned by acquire().
 *
 * Valid only on the acquiring thread; the destructor unpins the bytes
 * and, for Write/ReadWrite acquires, publishes the new version (the
 * tile's host-node binding is the actual one; the next plan re-seeds
 * from it).
 * */
class Acquired
{
  public:
    /*! Takes ownership of an implementation pin (internal). */
    Acquired(Acquired &&other) noexcept;
    Acquired &operator=(Acquired &&) = delete;
    /*! Unpins the host bytes and retires the host write, if any. */
    ~Acquired();

    /*! Typed view of the pinned host bytes. */
    template <class T> T *get_ptr() const { return static_cast<T *>(ptr_); }

    /*! Raw host pointer to the acquired bytes. */
    void *get() const;

    // Internal: runtime implementation detail.
    Acquired(void *impl, void *ptr, std::uint64_t generation = 0);

    // HandleImpl (opaque here); the destructor unpins the host-node
    // bytes and wakes the engine.
    void *impl_ = nullptr;
    void *ptr_ = nullptr;
    std::uint64_t generation_ = 0;
};

/*! A data handle: one logical buffer (tile) of nbytes bytes.
 *
 * A tile's residency is the BINDING TABLE's business (one binding per
 * node the tile's chunks occupy); the handle carries only the
 * tile-side facts — nbytes, its chunk count, its diagnostic name.
 * The runtime moves a tile's chunks node to node as planned COPYs;
 * the user only sees versions (every Write bumps the version by one
 * at the dataflow level). Handles are not copyable and must outlive
 * wait() of the work that uses them.
 *
 * Requirement: an Acquired MUST be destroyed before the Handle it came
 * from. ~Handle requires the handle's pins to be zero; otherwise it
 * records the runtime error and aborts the run (an Acquired would
 * otherwise use freed handle state).
 *
 * ONE CHUNK CLASS: every tile decomposes into n =
 * ceil(nbytes / chunk_class_bytes()) whole-class chunks of exactly the
 * runtime class initialized from NNHAUL_CHUNK_SIZE_MB — no second class and no
 * tail chunk of any other size; the last chunk's remainder is the
 * intra-chunk tail, wasted bytes inside a full class block. There is no per-handle chunk geometry.
 * */
class Handle
{
  public:
    /*! Creates a handle of nbytes bytes (nonzero). Needs no runtime;
     * the handle registers itself if init() already ran. */
    explicit Handle(std::size_t nbytes);
    /*! Creates a handle of nbytes bytes. The declared chunk_bytes is
     * ACCEPTED FOR SOURCE COMPATIBILITY and IGNORED: the one chunk
     * class rule fixes every tile's decomposition at
     * ceil(nbytes / chunk_class_bytes()). Needs no runtime. */
    Handle(std::size_t nbytes, std::size_t chunk_bytes);
    Handle(Handle const &) = delete;
    Handle &operator=(Handle const &) = delete;
    /*! Marks the tile dead. If a pending or unjoined task still names
     * it, the destructor closes and joins that work before the handle
     * dies; otherwise the bindings retire at the next close. */
    ~Handle();

    /*! Sets the diagnostic name (logs, error messages). */
    void set_name(std::string name);
    /*! The diagnostic name. */
    std::string const &name() const;
    /*! The logical payload size in bytes. */
    std::size_t nbytes() const;

    /*! Host access to the current bytes: Read pins the tile's
     * host-node bytes (moving them there first if only a device
     * holds the actual version); Write/ReadWrite hand the host bytes
     * out for a host-side rewrite that becomes the tile's actual
     * version. Blocks until consistent. */
    Acquired acquire(Access access);

    /*! Mark dead and order the drop of the handle's bindings after
     * every task already submitted. The handle must stay alive until
     * wait() joins the marker. */
    void invalidate();

    // Internal: runtime implementation detail (opaque outside src/).
    HandleImpl *impl_ = nullptr;
};

/*! The ONE chunk class: a RUNTIME constant and a
 * power of two, initialized by nnhaul from the NNHAUL_CHUNK_SIZE_MB
 * environment variable (default from <nnhaul/defs.h>). Every tile decomposes as
 * ceil(nbytes / chunk_class_bytes()) whole-class chunks — every chunk
 * exactly the class, the last chunk's logical payload possibly
 * shorter (the intra-chunk tail, wasted bytes inside the block;
 * there is no tail chunk of any other size) — so there is no
 * chunk-size policy to query. The kernels' chunk-index divmod lowers
 * to shift/mask on exactly this constant. The retired chunk-policy
 * knobs (NNHAUL_CHUNK_SIZES / NNHAUL_CHUNK_POLICY /
 * NNHAUL_CHUNK_MIN_COUNT) warn once and are ignored. (NNHAUL_CHUNK_TILES
 * is NOT one of them: it was never an engine knob — it is NNTile's
 * chunked-TILE allocation gate and is honored there.)
 * */

/*! Submits one task: the codelet, its buffer uses, and the cl_args
 * blob (copied synchronously — the task's registration-time fact; the
 * caller may free it on return).
 *
 * Tasks must arrive in a topological order of the data flow — the
 * submitted order is carried verbatim into the plan's queue, the whole
 * dependency input; cross-node dataflow arrives as explicit planned
 * COPYs the fold verifies.
 *
 * The worker id IS a node id: an EXPLICIT id must
 * name a node of the inventory, 0..k (devices 0..k-1, host k), else
 * the caller-defect "unknown worker id" error. worker_unspecified
 * resolves as follows — the lowest-numbered node of the codelet's
 * runnable-node set (a CUDA-body codelet resolves to the
 * set_default_cuda_worker default, device node 0 absent the setter; a
 * CPU-only codelet resolves to the host node) — evaluated once at
 * submit and recorded with the task; a codelet with neither body has
 * no runnable node and is a loud caller defect here. Assigning a
 * codelet to a node where it cannot run is NOT rejected here — that
 * is the checker's task-not-runnable, with the census that fault
 * owns. Insert never censuses: submission is fast
 * bookkeeping onto the pending chain; nothing launches, nothing
 * decides.
 *
 * The sim flag is retired: the reactive replay queue it selected is
 * deleted machinery (the plan checker's own check is the authority),
 * and insert(sim=true) fails loudly.
 * */
void insert(Codelet &codelet,
    std::initializer_list<BufferUse> buffers,
    void const *cl_args,
    std::size_t cl_arg_size,
    WorkerId worker = worker_unspecified,
    bool sim = false);

/*! The execution lane's stream of the RUNNING codelet's node
 * (in-codelet only, valid while its codelet body runs). */
cudaStream_t cuda_stream();

/*! The RUNNING node's preallocated contiguous device WORKSPACE
 * (in-codelet only). One buffer of NNHAUL_WORKSPACE_MB (default
 * 1024) per device, allocated at init OUTSIDE the chunk pools and
 * owned by the engine. Codelets that need one contiguous
 * address range for a multi-chunk operand (notably GEMM) copy their
 * chunks in, work on the workspace, and copy back.
 * `workspace_bytes()` is the buffer's size; a body that needs more is
 * a caller defect (loud, no fallback). */
void *workspace_ptr();
std::size_t workspace_bytes();

/*! The engine's workspace REGION, readable outside a codelet as well.
 * Embedders that own the GEMM staging workspace (nntile under
 * NNHaul) borrow this one buffer instead of reserving a
 * second, so the engine and the kernels share ONE workspace. Returns
 * false when no runtime/workspace is initialized. */
bool workspace_region(void **base, std::size_t *bytes);

/*! The cuBLAS handle of the RUNNING codelet's node, bound to that
 * node's execution lane (in-codelet only). */
cublasHandle_t cublas_handle();
/*! The cuDNN handle of the RUNNING codelet's node (in-codelet only). */
cudnnHandle_t cudnn_handle();
/*! The CUDA device ordinal of the RUNNING codelet's node
 * (in-codelet only). */
int device_index();

/*! The HOST node's worker id: k, the node count minus one (devices
 * 0..k-1, host k). Loud nnhaul::Error before init — k is runtime
 * inventory truth. */
WorkerId host_worker();

/*! Sets the default node for device-class resolution — the node a
 * worker_unspecified submit of a CUDA-body codelet resolves to
 * (the default device node). MUST name a DEVICE node (0..k-1), else
 * a loud nnhaul::Error. */
void set_default_cuda_worker(WorkerId id);
/*! Sets the default node for CPU-body resolution. MUST name the HOST
 * node (host_worker()) in v1, else a loud nnhaul::Error. */
void set_default_cpu_worker(WorkerId id);

/*! The duration model's loading surface (the model is supplied by the user): sets D(T, N) — the
 * modeled duration in ns of the codelet's task symbol on the named
 * node. Run-scoped and read-only once the first plan closes (loud
 * after); loud on a foreign worker id. The automatic lane-clock
 * simulation simply does not advance on a missing entry; the
 * duration-model-miss fault belongs to the internal timing pass. */
void set_duration(Codelet const &codelet, WorkerId worker,
    std::uint64_t ns);
/*! The duration model's channel half: R(A, B), the modeled transfer
 * rate in bytes per ns over the ordered pair's channel — one COPY's
 * modeled duration is chunk_class_bytes() / R(A, B). Run-scoped and
 * read-only once the first plan closes (loud after); loud on foreign
 * ids or an unordered inventory pair. */
void set_channel_rate(WorkerId a, WorkerId b, std::uint64_t bytes_per_ns);

/*! The result of the end-of-sequence hint. `issued_seq` is the batch
 * just issued (0 when nothing was issued); `run_pending` is true
 * whenever a run was submitted. 0 alone is not a sentinel for
 * "nothing". */
struct GraphEnd
{
    std::uint64_t issued_seq;
    bool run_pending;
};

/*! The END-OF-SEQUENCE HINT: the producer inserts it once, after the
 * last task of an iteration's graph (the trainer's run()), BEFORE the
 * next iteration is pushed. NNHaul then closes this batch — builds and
 * checks its plan and issues the literal execution — WITHOUT waiting,
 * so the next iteration's host work (graph compile) overlaps this
 * batch's device work. `wait()` remains the wholesale barrier; the
 * per-batch `join_predecessor()` joins the older of two in-flight
 * batches. Idempotent: with nothing pending it returns {0, false} when
 * no batch is in flight, and {0, true} when a previous batch is still
 * un-retired. */
GraphEnd signal_graph_end();
/*! Feature guard for embedders: defined when signal_graph_end() exists.
 * nntile calls it at the end of run() only under this macro, so the
 * same recorder compiles against older nnhaul revisions. */
#define NNHAUL_SIGNAL_GRAPH_END 1

/*! Joins the OLDEST un-retired batch (the predecessor of the newest),
 * making its device work complete and host-visible without joining the
 * newest batch. Returns its batch sequence number, or 0 when fewer
 * than two batches are un-retired (no-op). Completion is signaled by
 * the queue-close barriers over all lanes; no event is queried.
 * Throws the runtime error if the runtime failed meanwhile. */
std::uint64_t join_predecessor();

/*! Blocks until every submitted task and invalidate marker completed
 * (the queue closed, all lanes joined). Throws the runtime error if
 * the runtime failed meanwhile. */
void wait();

/*! Starts the runtime over the node inventory:
 * `ncuda` IS k — the number of device nodes — with `NNHAUL_DEVICES`
 * selecting WHICH k ordinals behind NodeIds 0..k-1 (a knob listing a
 * different count than k is a loud init error); `ncpu` is a
 * compatibility parameter (the host node exists exactly once) that
 * accepts exactly 0 or 1, else a loud init error. Under the memory
 * manager's policy every pool starts EMPTY at init and grows at
 * planning time only, through the two-phase ladder (tile-wise
 * allocation acts first, single chunks per event on fallback), until
 * its SOFT limit or the non-allocable flip (a failed single-chunk
 * allocation — allocations off permanently):
 * `cpu_cap_bytes` bounds the host pool under the precedence
 * NNHAUL_LIMIT_CPU_MEM > cpu_cap_bytes > 0 (unlimited), whole class
 * blocks only; `cuda_cap_bytes_each` bounds EVERY device pool under
 * the precedence NNHAUL_LIMIT_CUDA_MEM per device wins, else a
 * positive argument honored exactly, else 0 (unlimited), a sub-class
 * limit loud. The init order is select-and-probe, contexts,
 * peer-enable, pool limits (arenas and the host pool start empty —
 * NNHAUL_PREALLOC_*_MEM names an explicit head start), event
 * provisions, lanes, per-node workspaces and library handles, and the
 * binding table's N all-free segments; codelet registration happens
 * lazily at the first submit. The run phase never
 * allocates (growth is a compilation-phase act).
 *
 * `ncuda` must be at least 1: the engine always has a device lane
 * (host-only execution is not supported). `ncpu` is accepted as 0 or 1.
 *
 * The env knobs: NNHAUL_DEVICES (default ordinals
 * 0..ncuda-1 ascending), NNHAUL_LIMIT_CUDA_MEM (per-device arena SOFT
 * LIMIT MiB), NNHAUL_LIMIT_CPU_MEM (host pool SOFT LIMIT MiB),
 * NNHAUL_PREALLOC_CUDA_MEM / NNHAUL_PREALLOC_CPU_MEM (each side's
 * preallocation head start MiB, default none),
 * NNHAUL_WORKSPACE_MB (gather/scatter workspace, per device, default
 * 1024), NNHAUL_CUBLAS_WORKSPACE_MB (cuBLAS workspace, per device,
 * default 8), NNHAUL_EVENT_POOL (exact per-device event provision),
 * NNHAUL_CHUNK_SIZE_MB (the runtime chunk class, initialized on
 * first use). RETIRED — a set knob warns once and is ignored:
 * NNHAUL_PLAN_CHECK (one check, always full) and NNHAUL_BATCH_FENCE
 * (the fence is gone with the registry). */
void init(int ncpu,
    int ncuda,
    std::size_t cpu_cap_bytes,
    std::size_t cuda_cap_bytes_each);

/*! Barrier-then-join: the queue-close barriers on ALL lanes
 * (every device lane, the host worker's drain), then the host worker
 * stops, then resources release in reverse init order. A no-op when
 * the runtime is not running. */
void shutdown();

} // namespace nnhaul
