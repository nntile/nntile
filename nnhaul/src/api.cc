/*! @copyright (c) 2026-present Skolkovo Institute of Science and
 *                              Technology (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file src/api.cc
 * Public API: init/shutdown/insert/wait/signal_graph_end/
 * join_predecessor, the Handle lifecycle and host acquires, the
 * setters, the in-codelet accessors, and the test hooks — thin
 * surfaces over the engine's acts (engine.cc). The previous engine's
 * insert_common.hh is deleted (its footprint hash lives in task.hh;
 * the cap message is a caller-defect check below); the sim insert, the
 * offloadable flag, and every reactive-manager surface are deleted
 * with it.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "engine.hh"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace nnhaul
{

// A process-unique token for the loaded engine. A plugin that statically
// linked its own nnhaul exports a different token; the planner loader
// refuses such a plugin instead of running two engines.
extern "C" void const *nnhaul_engine_identity() noexcept
{
#ifdef NNHAUL_BUILT_SHARED
    static int const engine_token = 0;
    return &engine_token;
#else
    // A static build has no process-wide engine to protect: its plugins
    // are linked to the same archive and are expected by the test suite.
    return nullptr;
#endif
}

// ---- the runtime chunk class ----

ChunkClass const &chunk_class() noexcept
{
    // Initialized once, on first use: the value is the process's page
    // size and must be identical in the engine and every kernel. The
    // env knob is the runtime source; the configure-time macro is only
    // its default. A malformed value is a configuration fault and is
    // loud — never silently rounded.
    static ChunkClass const cls = [] {
        std::size_t mib = NNHAUL_DEFAULT_CHUNK_SIZE_MB;
        char const *knob = std::getenv("NNHAUL_CHUNK_SIZE_MB");
        if (knob != nullptr && *knob != '\0')
        {
            char *end = nullptr;
            errno = 0;
            long parsed = std::strtol(knob, &end, 10);
            bool const bad = end == knob || *end != '\0' || errno == ERANGE ||
                parsed <= 0 ||
                static_cast<unsigned long>(parsed) >
                    std::numeric_limits<std::size_t>::max() >> 20;
            if (bad || (static_cast<std::size_t>(parsed) &
                    (static_cast<std::size_t>(parsed) - 1)) != 0)
            {
                std::fprintf(stderr,
                    "[nnhaul] NNHAUL_CHUNK_SIZE_MB=%s is not a positive "
                    "power-of-two number of MiB\n",
                    knob);
                std::abort();
            }
            mib = static_cast<std::size_t>(parsed);
        }
        std::size_t const bytes = mib << 20;
        int shift = 0;
        for (std::size_t v = bytes; (v >>= 1) != 0;)
        {
            ++shift;
        }
        return ChunkClass{bytes, bytes - 1, shift};
    }();
    return cls;
}

// ---- the process-global handle registry ----

namespace
{
std::mutex g_handles_mu;
// The registry is an intrusive
// doubly-linked list (reg_prev/reg_next live in HandleImpl): delist
// is pointer surgery through the handle's own links — O(1), no
// scan-and-shift under the registry mutex.
HandleImpl *g_reg_head = nullptr;
HandleImpl *g_reg_tail = nullptr;
std::size_t g_reg_count = 0;

// A legacy chunk_bytes argument that names neither the retired
// unchunked view (0) nor the one class is ignored: the plan's chunk
// geometry is ceil(nbytes / chunk_class_bytes()) and nothing else. One
// stderr warning per process, house style.
void warn_ignored_chunk_bytes(std::size_t chunk_bytes)
{
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
    {
        std::fprintf(stderr,
            "[nnhaul] Handle chunk_bytes=%zu is ignored: the chunk "
            "geometry is ceil(nbytes / %zu) whole class chunks\n",
            chunk_bytes,
            chunk_class_bytes());
    }
}
} // namespace

void register_handle(HandleImpl *impl)
{
    std::lock_guard<std::mutex> lock(g_handles_mu);
    impl->reg_prev = g_reg_tail;
    impl->reg_next = nullptr;
    if (g_reg_tail != nullptr)
    {
        g_reg_tail->reg_next = impl;
    }
    else
    {
        g_reg_head = impl;
    }
    g_reg_tail = impl;
    ++g_reg_count;
    if (rt == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> rt_lock(rt->mu);
    if (!rt->tiles.carries(impl->tile_id))
    {
        rt->tiles.add(impl->tile_id, impl->nbytes);
    }
    rt->live_tiles.emplace(impl->tile_id, impl);
}

void unregister_handle(HandleImpl *impl, bool keep_dead)
{
    {
        std::lock_guard<std::mutex> lock(g_handles_mu);
        if (impl->reg_prev != nullptr || g_reg_head == impl)
        {
            if (impl->reg_prev != nullptr)
            {
                impl->reg_prev->reg_next = impl->reg_next;
            }
            else
            {
                g_reg_head = impl->reg_next;
            }
            if (impl->reg_next != nullptr)
            {
                impl->reg_next->reg_prev = impl->reg_prev;
            }
            else
            {
                g_reg_tail = impl->reg_prev;
            }
            impl->reg_prev = nullptr;
            impl->reg_next = nullptr;
            --g_reg_count;
        }
    }
    if (rt == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> rt_lock(rt->mu);
    rt->live_tiles.erase(impl->tile_id);
    if(!keep_dead)
    {
        rt->dead_pending_unlink(impl);
    }
    rt->acquired.erase(impl);
}

std::vector<HandleImpl *> handle_registry_snapshot()
{
    std::lock_guard<std::mutex> lock(g_handles_mu);
    std::vector<HandleImpl *> out;
    out.reserve(g_reg_count);
    for (HandleImpl *impl = g_reg_head; impl != nullptr;
         impl = impl->reg_next)
    {
        out.push_back(impl);
    }
    return out;
}

// ---- Codelet ----

namespace
{
std::atomic<std::uint32_t> g_next_codelet_id{1};
std::atomic<TileId> g_next_tile_id{1};
} // namespace

Codelet::Codelet(std::string name, Kernel cpu, Kernel cuda,
    FootprintFn footprint)
    : name_(std::move(name)), cpu_(cpu), cuda_(cuda), footprint_(footprint),
      id_(g_next_codelet_id.fetch_add(1, std::memory_order_relaxed))
{
}

std::uint32_t Codelet::id() const { return id_; }
bool Codelet::has_cuda() const { return cuda_ != nullptr; }
bool Codelet::has_cpu() const { return cpu_ != nullptr; }
std::string const &Codelet::name() const { return name_; }

// ---- Handle ----

Handle::Handle(std::size_t nbytes) : Handle(nbytes, 0) {}

Handle::Handle(std::size_t nbytes, std::size_t chunk_bytes)
{
    if (nbytes == 0)
    {
        throw Error("handle size must be nonzero");
    }
    // Decompose BEFORE allocating: a decomposition fault must not leak
    // the impl the constructor would otherwise have created.
    std::size_t const chunks = tile_chunk_count(nbytes);
    impl_ = new HandleImpl;
    impl_->nbytes = nbytes;
    impl_->n = chunks;
    impl_->tile_id = g_next_tile_id.fetch_add(1, std::memory_order_relaxed);
    // The chunk geometry is the plan's, not a caller's choice: ceil(
    // nbytes / the one class) whole chunks of this tile alone, the
    // tail shorter; at or below the class the tile is exactly one
    // chunk with the contiguous get_ptr view. A legacy chunk_bytes
    // argument names no sub-chunk storage and is ignored.
    if (chunk_bytes != 0 && chunk_bytes != chunk_class_bytes())
    {
        warn_ignored_chunk_bytes(chunk_bytes);
    }
    register_handle(impl_);
}

Handle::~Handle()
{
    if (impl_ == nullptr)
    {
        return;
    }
    if (RuntimeState *R = rt.get())
    {
        // An Acquired outliving its Handle would dereference freed
        // state. Loud, never a silent use-after-free.
        if (impl_->pins != 0)
        {
            std::fprintf(stderr, "nnhaul: handle destroyed while acquired\n");
            R->set_error("handle destroyed while acquired");
            std::abort();
        }
        impl_->dead.store(true, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(R->mu);
            if (impl_->dead_prev == nullptr
                && R->dead_pending_head != impl_)
            {
                R->dead_pending_push(impl_);
            }
        }
        // A handle destroyed while tasks still name it must not leave
        // them with a dangling Handle*: close and join the work now, so
        // the retirement sweep runs before impl_ dies. The common burst
        // (temporaries dropped after wait()) has nothing pending and
        // keeps the cheap lazy marker.
        bool referenced = false;
        bool poisoned = false;
        try
        {
            referenced = engine_handle_referenced(*R, this);
        }
        catch (...)
        {
            referenced = true;
        }
        {
            std::lock_guard<std::mutex> lock(R->mu);
            poisoned = R->error_set;
        }
        if (referenced && !poisoned)
        {
            try
            {
                engine_sync_close(*R);
            }
            catch (Error const &e)
            {
                // impl_ is still named by unexecuted work; deleting it
                // would turn the latched fault into a use-after-free.
                R->set_error(e.what());
                std::fprintf(stderr,
                    "nnhaul: handle destroyed with unjoinable work: %s\n",
                    e.what());
                std::abort();
            }
        }
        else if (!poisoned)
        {
            // Nothing names this handle, but its bindings still sit in
            // the seed. The marker must survive until the next sweep, so
            // the engine takes ownership of impl_ and deletes it at that
            // sweep; deleting here would drop the marker and leak the
            // tile's pool chunks forever.
            bool already_done = false;
            {
                std::lock_guard<std::mutex> lock(R->mu);
                impl_->engine_owned = true;
                already_done = impl_->dead_done;
            }
            if (already_done)
            {
                impl_->engine_owned = false;
                unregister_handle(impl_);
                delete impl_;
            }
            else
            {
                unregister_handle(impl_, /*keep_dead=*/true);
            }
            impl_ = nullptr;
            return;
        }
        else
        {
            // Referenced-and-poisoned (or nothing referenced): the forced
            // shutdown join still drains un-retired batches, and a
            // swept_pending impl may be in one of them. Hand ownership to
            // the engine unless the sweep already finished.
            bool already_done = false;
            {
                std::lock_guard<std::mutex> lock(R->mu);
                impl_->engine_owned = true;
                already_done = impl_->dead_done;
            }
            if (already_done)
            {
                impl_->engine_owned = false;
                unregister_handle(impl_);
                delete impl_;
            }
            else
            {
                unregister_handle(impl_, /*keep_dead=*/true);
            }
            impl_ = nullptr;
            return;
        }
    }
    // ALWAYS: the process-global registry outlives the runtime, and a
    // handle destroyed after shutdown() must still leave it — else the
    // NEXT init's snapshot reads freed memory (a dangling registry entry,
    // a heap write through
    // it at the retirement sweep, the corruption surfacing in the
    // driver's free path).
    unregister_handle(impl_);
    delete impl_;
    impl_ = nullptr;
}

void Handle::set_name(std::string name)
{
    impl_->name = std::move(name);
}

std::string const &Handle::name() const
{
    return impl_->name;
}

std::size_t Handle::nbytes() const
{
    return impl_->nbytes;
}

Acquired Handle::acquire(Access access)
{
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        throw Error("nnhaul::acquire before nnhaul::init");
    }
    return Acquired(impl_, engine_acquire(*R, *impl_, access),
        R->generation);
}

void Handle::invalidate()
{
    if (impl_ == nullptr)
    {
        return;
    }
    // The non-blocking public marker: the handle's bindings drop at
    // the next close-and-plan's retirement sweep, after every task
    // already submitted (insert on a dead handle throws, so every task
    // naming it is in this batch or an earlier one).
    bool const was_dead =
        impl_->dead.exchange(true, std::memory_order_relaxed);
    RuntimeState *R = rt.get();
    if (!was_dead && R != nullptr)
    {
        std::lock_guard<std::mutex> lock(R->mu);
        if (impl_->dead_prev == nullptr && R->dead_pending_head != impl_)
        {
            R->dead_pending_push(impl_);
        }
    }
}

// ---- Acquired ----

Acquired::Acquired(void *impl, void *ptr, std::uint64_t generation)
    : impl_(impl), ptr_(ptr), generation_(generation)
{
}

Acquired::Acquired(Acquired &&other) noexcept
    : impl_(other.impl_), ptr_(other.ptr_),
      generation_(other.generation_)
{
    other.impl_ = nullptr;
    other.ptr_ = nullptr;
}

Acquired::~Acquired()
{
    if (impl_ == nullptr)
    {
        return;
    }
    if (RuntimeState *R = rt.get())
    {
        HandleImpl &impl = *static_cast<HandleImpl *>(impl_);
        try
        {
            engine_release(*R, impl, generation_);
        }
        catch (Error const &e)
        {
            R->set_error(e.what());
        }
    }
}

void *Acquired::get() const
{
    return ptr_;
}

// ---- insert ----

void insert(Codelet &codelet,
    std::initializer_list<BufferUse> buffers,
    void const *cl_args,
    std::size_t cl_arg_size,
    WorkerId worker,
    bool sim)
{
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        throw Error("nnhaul::insert before nnhaul::init");
    }
    if (R->error_set)
    {
        throw Error(R->error);
    }
    if (sim)
    {
        // The reactive replay queue is deleted machinery (the plan
        // checker's simulation is the simulation of record); the flag
        // stays in the signature for source compatibility and fails
        // loudly.
        throw Error("nnhaul::insert(sim=true) is retired: the plan "
                    "checker's simulation replaced the sim replay");
    }
    if (cl_arg_size > 0 && cl_args == nullptr)
    {
        throw Error("cl_args is null");
    }

    // The worker id IS a node id : an EXPLICIT id must
    // name a node of the inventory — devices 0..k-1, host k — else the
    // caller-defect "unknown worker id" error.
    NodeId node = 0;
    if (worker != worker_unspecified)
    {
        if (worker < 0 ||
            static_cast<std::size_t>(worker) >= R->inv->node_count())
        {
            throw Error("unknown worker id " + std::to_string(worker) +
                        " for codelet \"" + codelet.name() +
                        "\" (valid ids are devices 0.." +
                        std::to_string(R->inv->device_count() - 1) +
                        " and host " +
                        std::to_string(R->inv->host_id()) + ")");
        }
        node = static_cast<NodeId>(worker);
    }
    else
    {
        // The lowest-numbered node of the codelet's
        // runnable set, with the engine's setter state — evaluated
        // once at submit and recorded with the task. A codelet with
        // neither body has no runnable node: a loud caller defect.
        std::optional<NodeId> resolved = default_node_of(codelet,
            *R->inv, R->default_cuda_node, R->default_cpu_node);
        if (!resolved.has_value())
        {
            throw Error("codelet \"" + codelet.name() +
                        "\" has neither a CUDA nor a CPU kernel body — "
                        "runnable on no node (worker_unspecified cannot "
                        "resolve)");
        }
        node = *resolved;
    }

    // The buffer/handle checks (carried over) plus the cap rejection:
    // a tile whose chunk count exceeds EVERY node's GROWTH CEILING can
    // never bind — the caller-defect face of the whole-class pool
    // bound (the plan-time census owns everything a pool COULD cover).
    // The bound is the SOFT LIMIT, not the current
    // cardinality: pools start empty and grow on demand, so a fresh
    // run's current cardinality is 0 everywhere; a limit of 0 means
    // UNLIMITED (any tile may bind — the hard limit owns the rest).
    std::size_t best = 0;
    bool unlimited = false;
    for (std::size_t n = 0; n < R->inv->node_count(); ++n)
    {
        std::size_t const limit =
            R->inv->pool_limit(static_cast<NodeId>(n));
        if (limit == 0)
        {
            unlimited = true; // some node can grow without ceiling
        }
        else
        {
            best = std::max(best, limit);
        }
    }
    if (unlimited)
    {
        best = std::numeric_limits<std::size_t>::max();
    }
    for (BufferUse const &use : buffers)
    {
        if (use.handle == nullptr)
        {
            throw Error("null handle");
        }
        HandleImpl &impl = *use.handle->impl_;
        if (impl.dead.load(std::memory_order_relaxed))
        {
            throw Error("handle \"" + label_of(impl) + "\" is dead");
        }
        if (impl.n > best)
        {
            throw Error("tile \"" + label_of(impl) + "\" (" +
                        std::to_string(impl.nbytes) + " bytes, " +
                        std::to_string(impl.n) +
                        " chunks) is too large for every node pool "
                        "(largest cap " +
                        std::to_string(best) + " chunks of " +
                        std::to_string(chunk_class_bytes()) + " bytes)");
        }
    }

    // Submission is bookkeeping only : the task joins
    // the pending chain in submit order; nothing launches, nothing
    // decides.
    std::lock_guard<std::mutex> sm(R->submit_mu);
    Task *t = R->arena.alloc();
    t->codelet = &codelet;
    t->node = node;
    t->submit_seq = R->next_seq++;
    for (BufferUse const &use : buffers)
    {
        Task::Buf b;
        b.handle = use.handle;
        b.access = use.access;
        if (t->nbufs == Task::kInlineBufs)
        {
            t->bufs_heap.assign(
                t->bufs_inline, t->bufs_inline + Task::kInlineBufs);
        }
        if (t->nbufs >= Task::kInlineBufs)
        {
            t->bufs_heap.push_back(b);
        }
        else
        {
            t->bufs_inline[t->nbufs] = b;
        }
        t->nbufs += 1;
    }
    t->store_cl(cl_args, cl_arg_size);
    t->footprint = footprint_of(
        codelet, t->bufs(), t->nbufs, t->cl_data(), t->cl_size);
    R->queue.append_chain(t, t);
}

// ---- the in-codelet accessors (the RUNNING node's facts) ----

cudaStream_t cuda_stream()
{
    if (!tls.in_kernel || tls.stream == nullptr)
    {
        throw Error("nnhaul::cuda_stream is only valid inside a CUDA codelet");
    }
    return tls.stream;
}

void *workspace_ptr()
{
    if (!tls.in_kernel || tls.workspace == nullptr)
    {
        throw Error(
            "nnhaul::workspace_ptr is only valid inside a CUDA codelet");
    }
    return tls.workspace;
}

std::size_t workspace_bytes()
{
    return tls.workspace_bytes;
}

bool workspace_region(void **base, std::size_t *bytes)
{
    // Inside a CUDA codelet the RUNNING node's workspace is the only one
    // the lane's stream may use. Returning node 0's buffer here stages a
    // GEMM on the wrong device (or lets two devices share one buffer).
    if (tls.in_kernel && tls.workspace != nullptr)
    {
        if (base != nullptr)
        {
            *base = tls.workspace;
        }
        if (bytes != nullptr)
        {
            *bytes = tls.workspace_bytes;
        }
        return true;
    }
    RuntimeState *R = rt.get();
    if (R == nullptr || R->dev.empty() || R->dev[0].workspace == nullptr)
    {
        if (base != nullptr)
        {
            *base = nullptr;
        }
        if (bytes != nullptr)
        {
            *bytes = 0;
        }
        return false;
    }
    // Outside a codelet, node 0's workspace is the historical region.
    if (base != nullptr)
    {
        *base = R->dev[0].workspace;
    }
    if (bytes != nullptr)
    {
        *bytes = R->dev[0].workspace_bytes;
    }
    return true;
}

cublasHandle_t cublas_handle()
{
    if (!tls.in_kernel || tls.cublas == nullptr)
    {
        throw Error(
            "nnhaul::cublas_handle is only valid inside a CUDA codelet");
    }
    return tls.cublas;
}

cudnnHandle_t cudnn_handle()
{
    if (!tls.in_kernel || tls.cudnn == nullptr)
    {
        throw Error(
            "nnhaul::cudnn_handle is only valid inside a CUDA codelet");
    }
    return tls.cudnn;
}

int device_index()
{
    if (!tls.in_kernel)
    {
        throw Error(
            "nnhaul::device_index is only valid inside a codelet");
    }
    return tls.device;
}

// ---- the workers and the setters  ----

WorkerId host_worker()
{
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        throw Error("nnhaul::host_worker before nnhaul::init — k is "
                    "runtime inventory truth ");
    }
    return static_cast<WorkerId>(R->inv->host_id());
}

void set_default_cuda_worker(WorkerId id)
{
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        throw Error("nnhaul::set_default_cuda_worker before nnhaul::init");
    }
    if (id < 0 || static_cast<std::size_t>(id) >= R->inv->device_count())
    {
        throw Error("nnhaul::set_default_cuda_worker(" +
                    std::to_string(id) +
                    "): must name a DEVICE node (0.." +
                    std::to_string(R->inv->device_count() - 1) + ")");
    }
    std::lock_guard<std::mutex> lock(R->mu);
    R->default_cuda_node = static_cast<NodeId>(id);
}

void set_default_cpu_worker(WorkerId id)
{
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        throw Error("nnhaul::set_default_cpu_worker before nnhaul::init");
    }
    if (id != static_cast<WorkerId>(R->inv->host_id()))
    {
        throw Error("nnhaul::set_default_cpu_worker(" +
                    std::to_string(id) +
                    "): must name the HOST node (" +
                    std::to_string(R->inv->host_id()) +
                    ") in this version ");
    }
    std::lock_guard<std::mutex> lock(R->mu);
    R->default_cpu_node = R->inv->host_id();
}

// ---- the duration model's loading surface  ----

void set_duration(Codelet const &codelet, WorkerId worker,
    std::uint64_t ns)
{
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        throw Error("nnhaul::set_duration before nnhaul::init");
    }
    if (worker < 0 ||
        static_cast<std::size_t>(worker) >= R->inv->node_count())
    {
        throw Error("nnhaul::set_duration: unknown worker id " +
                    std::to_string(worker));
    }
    std::lock_guard<std::mutex> lock(R->mu);
    if (R->model_read_only)
    {
        throw Error("nnhaul::set_duration: the duration model is "
                    "read-only once the first plan closed");
    }
    R->duration_by_codelet[(static_cast<std::uint64_t>(codelet.id())
                                << 8) |
        static_cast<NodeId>(worker)] = ns;
}

void set_channel_rate(WorkerId a, WorkerId b, std::uint64_t bytes_per_ns)
{
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        throw Error("nnhaul::set_channel_rate before nnhaul::init");
    }
    if (a < 0 || b < 0 ||
        static_cast<std::size_t>(a) >= R->inv->node_count() ||
        static_cast<std::size_t>(b) >= R->inv->node_count())
    {
        throw Error("nnhaul::set_channel_rate: unknown worker id " +
                    std::to_string(a) + " or " + std::to_string(b));
    }
    if (bytes_per_ns == 0)
    {
        throw Error("nnhaul::set_channel_rate: a zero rate is not a "
                    "channel entry");
    }
    std::lock_guard<std::mutex> lock(R->mu);
    if (R->model_read_only)
    {
        throw Error("nnhaul::set_channel_rate: the duration model is "
                    "read-only once the first plan closed");
    }
    NodeId const na = static_cast<NodeId>(a);
    NodeId const nb = static_cast<NodeId>(b);
    if (!R->inv->has_channel(na, nb))
    {
        throw Error("nnhaul::set_channel_rate(" + std::to_string(a) +
                    ", " + std::to_string(b) +
                    "): the inventory carries no such ordered channel");
    }
    if (R->channel_rates.size() < R->inv->lane_count())
    {
        R->channel_rates.assign(R->inv->lane_count(), 0);
    }
    R->channel_rates[R->inv->lane_of_chan(na, nb)] = bytes_per_ns;
}

// ---- the pipeline's public acts ----

GraphEnd signal_graph_end()
{
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        return GraphEnd{0, false};
    }
    // The close-and-plan ISSUE act : the batch's plan is
    // built, checked, and ENQUEUED — the barriers do not run; the
    // batch joins at join_predecessor()/wait(), so the next
    // iteration's host work overlaps this batch's device work. ONE
    // discipline for traced and untraced runs alike: the stamps are
    // placed at issue and read at each plan's queue-close join,
    // wherever that join happens  — the armed tracer
    // owns no issue discipline of its own. (The old mechanism's
    // close-at-wait branch stood here and DEFERRED every batch's
    // issue to the waits, fusing the pending chain into one
    // plan-sized monster per wait — the planner's walk then starved
    // the device for seconds mid-loop; that finding removed it
    // with the fire ring it served.)
    return engine_issue_batch(*R);
}

std::uint64_t join_predecessor()
{
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        return 0;
    }
    // The join act : fewer than two un-retired batches
    // is the no-op 0; otherwise the OLDER batch's queue-close barriers
    // join here (a REAL join — the executor's close over the retained
    // slot), making its device work host-visible without joining the
    // newest batch.
    {
        std::lock_guard<std::mutex> lock(R->mu);
        if (R->error_set)
        {
            throw Error(R->error);
        }
        if (R->unretired.size() < 2)
        {
            return 0;
        }
    }
    return engine_join_oldest(*R);
}

void wait()
{
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        return;
    }
    // The wholesale barrier: join every un-retired batch, close (and
    // join) the pending chain, join again — every submitted task and
    // invalidate marker completed, all lanes joined.
    engine_sync_close(*R);
}

// ---- init / shutdown ----

namespace
{

//! Validate NNHAUL_CHUNK_SIZE_MB as a catchable init fault. The cached
//! chunk_class() aborts (it is noexcept and may be reached before init),
//! so init checks the same knob first and refuses malformed values with
//! nnhaul::Error.
void validate_chunk_class_env()
{
    char const *knob = std::getenv("NNHAUL_CHUNK_SIZE_MB");
    if (knob == nullptr || *knob == '\0')
    {
        return;
    }
    char *end = nullptr;
    errno = 0;
    long const parsed = std::strtol(knob, &end, 10);
    bool const bad = end == knob || *end != '\0' || errno == ERANGE
        || parsed <= 0
        || static_cast<unsigned long>(parsed)
            > std::numeric_limits<std::size_t>::max() >> 20
        || (static_cast<std::size_t>(parsed)
            & (static_cast<std::size_t>(parsed) - 1)) != 0;
    if (bad)
    {
        throw Error(std::string("nnhaul: NNHAUL_CHUNK_SIZE_MB=") + knob
            + " is not a positive power-of-two number of MiB");
    }
}

} // namespace

void init(int ncpu,
    int ncuda,
    std::size_t cpu_cap_bytes,
    std::size_t cuda_cap_bytes_each)
{
    std::lock_guard<std::mutex> api_lock(api_mu);
    if (rt != nullptr)
    {
        throw Error("nnhaul::init was already called");
    }
    if (ncuda < 1)
    {
        throw Error("nnhaul::init needs at least one CUDA device "
            "(ncuda >= 1): the host-only configuration is not supported");
    }
    // Validate/initialize the runtime chunk class first: one malformed
    // NNHAUL_CHUNK_SIZE_MB is a loud init fault, never a mid-run one.
    validate_chunk_class_env();
    (void)chunk_class();
    rt = std::make_unique<RuntimeState>();
    try
    {
        engine_init(ncpu, ncuda, cpu_cap_bytes, cuda_cap_bytes_each);
    }
    catch (...)
    {
        rt.reset();
        throw;
    }
}

void shutdown()
{
    std::lock_guard<std::mutex> api_lock(api_mu);
    if (rt == nullptr)
    {
        return;
    }
    engine_shutdown();
    rt.reset();
}

// ---- test hooks  ----

Stats stats()
{
    Stats s;
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        return s;
    }
    std::lock_guard<std::mutex> lock(R->mu);
    NodeInventory const &inv = *R->inv;
    s.memory_nodes = static_cast<int>(inv.node_count());
    s.cpu_cap_bytes = (R->host_pool->limit_blocks() != 0
            ? R->host_pool->limit_blocks()
            : R->host_pool->block_count()) *
        chunk_class_bytes(); // the bound (0=unlimited reads as held)
    s.node.reserve(inv.node_count());
    for (std::size_t n = 0; n < inv.node_count(); ++n)
    {
        NodeId const node = static_cast<NodeId>(n);
        NodePoolStats p;
        p.host = inv.is_host(node);
        p.pool_blocks = inv.pool_cardinality(node);
        p.pool_bytes = p.pool_blocks * chunk_class_bytes();
        p.free_blocks = R->executor != nullptr
            ? R->executor->applied().free_count(node)
            : p.pool_blocks;
        p.free_bytes = p.free_blocks * chunk_class_bytes();
        p.alloc_count = p.host
            ? static_cast<std::size_t>(
                  R->host_pool->segment_count())
            : (n < R->exec_lanes.size() && R->exec_lanes[n] != nullptr
                    ? R->exec_lanes[n]->segment_count()
                    : 0) +
                R->dev[n].alloc_count; // arena segments + fixed carves
        p.workspace_bytes = p.host
            ? 0
            : R->dev[n].workspace_bytes + R->dev[n].cublas_workspace_bytes +
                R->dev[n].copy_desc_bytes;
        s.node.push_back(p);
    }
    // The per-lane busy counters: the tracer's own facts when it
    // is armed (an untraced run places no stamps, so its lanes read
    // zero — the counters are the trace's own product).
    s.lane_busy_ns.assign(inv.lane_count(), 0);
    if (R->tracer != nullptr && R->tracer->armed())
    {
        for (MeasuredTraceRecord const &r :
            R->tracer->measured_records())
        {
            if (r.lane != kNoLane &&
                static_cast<std::size_t>(r.lane) < s.lane_busy_ns.size())
            {
                s.lane_busy_ns[r.lane] += r.end_ns - r.start_ns;
            }
        }
    }
    s.cross_node_copies = R->copy_log_.size();
    return s;
}

ManagerStats manager_stats()
{
    ManagerStats m;
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        return m;
    }
    std::lock_guard<std::mutex> lock(R->mu);
    m.link_ops = R->link_ops;
    m.unlink_ops = R->unlink_ops;
    m.copy_ops = R->copy_ops;
    m.tile_ops = R->tile_ops;
    m.plan_walks = R->batches_planned;
    m.plan_ops_planned = R->planned_ops;
    m.plan_ops_issued = R->executor != nullptr
        ? R->executor->ops_issued()
        : 0;
    m.plan_reconcile_ok = true;
    if (R->executor != nullptr)
    {
        for (PlanExecutor::PlanIssue const &issue :
            R->executor->issues())
        {
            m.plan_reconcile_ok =
                m.plan_reconcile_ok && issue.reconcile_ok;
        }
    }
    m.simulate_makespan_ns = R->simulate_makespan_ns;
    return m;
}

bool handle_swept(Handle const &handle)
{
    if (handle.impl_ == nullptr)
    {
        return false;
    }
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        return false;
    }
    std::lock_guard<std::mutex> lock(R->mu);
    return handle.impl_->dead_done;
}

std::vector<TaskView> tasks()
{
    std::vector<TaskView> out;
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        return out;
    }
    std::lock_guard<std::mutex> lock(R->mu);
    out = R->task_log;
    if (R->error_set)
    {
        for (TaskView &v : out)
        {
            if (v.phase == TaskPhase::Running)
            {
                v.phase = TaskPhase::Failed;
            }
        }
    }
    // The still-queued tail: submitted, not yet planned.
    std::lock_guard<std::mutex> sm(R->submit_mu);
    for (Task *t = R->queue.pending_head_.load(std::memory_order_relaxed);
         t != nullptr;
         t = t->next)
    {
        TaskView v;
        v.submit_seq = t->submit_seq;
        v.worker = static_cast<WorkerId>(t->node);
        v.kind = R->inv->is_host(t->node) ? Kind::Cpu : Kind::Cuda;
        v.footprint = t->footprint;
        v.phase = TaskPhase::Submitted;
        v.codelet = t->codelet != nullptr ? t->codelet->name()
                                          : std::string();
        out.push_back(std::move(v));
    }
    return out;
}

std::vector<CopyRecord> copy_log()
{
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        return {};
    }
    std::lock_guard<std::mutex> lock(R->mu);
    return R->copy_log_;
}

WorkerId current_worker()
{
    if (!tls.in_kernel)
    {
        return worker_unspecified;
    }
    return static_cast<WorkerId>(tls.node);
}

Kind current_kind()
{
    if (!tls.in_kernel)
    {
        return Kind::Cpu;
    }
    return tls.host_node ? Kind::Cpu : Kind::Cuda;
}

std::uint64_t kernel_launches()
{
    RuntimeState *R = rt.get();
    if (R == nullptr)
    {
        return 0;
    }
    return R->kernel_launches_.load(std::memory_order_relaxed);
}

bool trace_enabled()
{
    RuntimeState *R = rt.get();
    return R != nullptr && R->tracer != nullptr && R->tracer->armed();
}

bool trace_dump(std::string const &path)
{
    RuntimeState *R = rt.get();
    if (R == nullptr || R->tracer == nullptr)
    {
        return false;
    }
    return R->tracer->dump(path);
}

} // namespace nnhaul
