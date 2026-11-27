/*! @copyright (c) 2026-present Skolkovo Institute of Science and
 *                              Technology (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file src/task.hh
 * The task symbol and its registration facts: one Codelet IS one task
 * symbol whose two bodies are its two implementations; the
 * scalar-argument blob is copied at submit and is the task's
 * registration-time content; the resolved execution node is recorded
 * at submit. The pending chain — submission-order bookkeeping the
 * engine drains at close-and-plan — is unchanged.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

// Task + the executability plumbing + TaskArena + the pending chain
// under the chunk-graph engine: submission is BOOKKEEPING ONLY
// — insert() appends tasks in submit order, resolves
// the node once (recorded with the task), and nothing launches until
// the host asks the queue to move (wait(), a host acquire, or the
// signal_graph_end hint), which triggers close-and-plan: the whole
// pending chain becomes one batch, planned once from a drained
// snapshot, and executed as the plan's literal ops. What survives of
// the old queue is the submitter-side chain (append and take) and the
// arena's stable-address recycling.

#include "data.hh"
#include "nnhaul/nnhaul.hh"
#include "plan.hh"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <new>
#include <optional>
#include <vector>

namespace nnhaul
{

// ---- the task symbol's identity and its runnable nodes  ----

// The plan-time TaskId IS the codelet identity: the Codelet
// constructor assigns the dense id, and registration resolves identity
// -> executable per node — every CUDA body loaded in each device
// node's context, each CPU body the host node's executable. One task
// symbol, two implementations.
inline TaskId task_id_of(Codelet const &codelet)
{
    return codelet.id();
}

// The runnable-node set derived MECHANICALLY from the codelet's bodies
// over the inventory — every CUDA device node iff the codelet has a
// CUDA body, the host node iff it has a CPU body. Ascending by NodeId,
// so the lowest runnable node is nodes.front(). This is the fact the
// engine renders into the ExecutabilityTable at registration; a
// codelet with neither body returns the empty set (runnable nowhere —
// every TILE_OP naming it is the check-time fault task-not-runnable).
inline std::vector<NodeId> runnable_nodes_of(
    Codelet const &codelet, NodeInventory const &inv)
{
    std::vector<NodeId> nodes;
    if (codelet.has_cuda())
    {
        nodes.reserve(inv.device_count() + 1);
        for (std::size_t d = 0; d < inv.device_count(); ++d)
        {
            nodes.push_back(static_cast<NodeId>(d));
        }
    }
    if (codelet.has_cpu())
    {
        nodes.push_back(inv.host_id());
    }
    return nodes;
}

// The worker_unspecified default — the lowest-numbered node of the
// codelet's runnable set with the engine's setter state applied (a
// CUDA-body codelet resolves to the device-class default, absent the
// setter device node 0; a CPU-only codelet to the host).
// A pure function of (codelet bodies, inventory, setter state),
// evaluated ONCE at submit and recorded with the task. Nullopt when
// the codelet has neither body: no runnable node exists to name, the
// caller defect insert reports loudly.
inline std::optional<NodeId> default_node_of(Codelet const &codelet,
    NodeInventory const &inv,
    NodeId default_cuda_node,
    NodeId default_cpu_node)
{
    if (codelet.has_cuda())
    {
        return default_cuda_node;
    }
    if (codelet.has_cpu())
    {
        return default_cpu_node;
    }
    // inv is deliberately unused: the setter defaults arrive already
    // validated, and the remaining corner — a resolution naming an id
    // the inventory cannot hold (k = 0, a stale id) — is insert's
    // caller-defect range check 0..k, the engine insert seam.
    (void) inv;
    return std::nullopt;
}

// ---- the submitted task and its registration facts ----

struct Task
{
    Task *prev = nullptr;
    Task *next = nullptr;

    Codelet *codelet = nullptr;

    // The resolved execution node, recorded at submit: evaluated once
    // at submit, recorded with the task, and carried into the plan like
    // any explicit assignment — an explicit WorkerId verbatim, or the
    // default. The lowering walk emits it as the TILE_OP's node field.
    NodeId node = 0;

    // nntile submits at most 12 buffers per task; a larger list falls
    // back to the heap (kept for API completeness, never exercised by
    // nntile).
    static constexpr std::size_t kInlineBufs = 12;
    struct Buf
    {
        Handle *handle = nullptr;
        Access access = Access::Read; // the declared access IS the plan's
                                      // access (no version counters
                                      // anywhere)
    };
    std::size_t nbufs = 0;
    Buf bufs_inline[kInlineBufs];
    std::vector<Buf> bufs_heap;

    Buf *bufs()
    {
        return nbufs <= kInlineBufs ? bufs_inline : bufs_heap.data();
    }
    Buf const *bufs() const
    {
        return nbufs <= kInlineBufs ? bufs_inline : bufs_heap.data();
    }

    // cl_args copy: 64 bytes inline, heap fallback. The submitter frees
    // its blob the moment insert() returns, so the copy is synchronous
    // — and the copy IS the task's registration-time fact: the
    // scalar arguments travel with the task symbol, so a workload
    // whose scalars vary per op must either multiply task symbols or
    // route the varying values through operand chunk bytes.
    static constexpr std::size_t kClSboBytes = 64;
    std::size_t cl_size = 0;
    bool cl_inline = true;
    alignas(std::max_align_t) unsigned char cl_sbo[kClSboBytes];
    std::vector<unsigned char> cl_heap;

    void store_cl(void const *args, std::size_t n)
    {
        cl_size = n;
        if (n == 0)
        {
            cl_inline = true;
            return;
        }
        if (n <= kClSboBytes)
        {
            cl_inline = true;
            std::memcpy(cl_sbo, args, n);
        }
        else
        {
            cl_inline = false;
            cl_heap.resize(n);
            std::memcpy(cl_heap.data(), args, n);
        }
    }

    void const *cl_data() const
    {
        if (cl_size == 0)
        {
            return nullptr;
        }
        return cl_inline ? static_cast<void const *>(cl_sbo)
                         : static_cast<void const *>(cl_heap.data());
    }

    void destroy_cl()
    {
        if (!cl_inline)
        {
            std::vector<unsigned char>().swap(cl_heap);
        }
        if (nbufs > kInlineBufs)
        {
            std::vector<Buf>().swap(bufs_heap);
        }
        cl_size = 0;
        cl_inline = true;
        nbufs = 0;
    }

    std::uint64_t submit_seq = 0;

    // Diagnostic history key (0-entropy when the codelet has no
    // footprint fn — footprint_of below fills it).
    std::uint64_t footprint = 0;

    // List membership, for O(1) invalidate-marker placement. linked:
    // the node is in the pending chain or the queue (cleared when the
    // node is unlinked/freed). pending: the node is still in the
    // submitter-side chain. Relaxed atomics: written under submit_mu
    // or rt_mu and read across the switch during marker placement.
    std::atomic<bool> linked{false};
    std::atomic<bool> pending{false};
};

// ---- the SeqTask conversion surface  ----

// One submitted task as the checker's task sequence sees it: the task
// symbol (the codelet identity) and its operand tiles with their
// accesses. The pending chain's submit order is the sequence's order.
inline SeqTask seq_task_of(Task const &task)
{
    SeqTask st;
    st.task = task_id_of(*task.codelet);
    st.operands.reserve(task.nbufs);
    for (std::size_t i = 0; i < task.nbufs; ++i)
    {
        Task::Buf const &buf = task.bufs()[i];
        st.operands.push_back(
            TileAccess{buf.handle->impl_->tile_id, buf.access});
    }
    return st;
}

// ---- the footprint key (folded here from the deleted
// ---- src/insert_common.hh; the cap message moves to api.cc) ----

// FNV-1a key over codelet id, kind label, and (footprint fn result or
// the per-buffer sizes) — the diagnostic history key shape the task
// views pin. One implementation so every path produces byte-identical
// keys for identical inputs.
inline void mix_bytes(std::uint64_t &hash, unsigned char byte)
{
    hash ^= byte;
    hash *= 1099511628211ull;
}

inline void mix_u32(std::uint64_t &hash, std::uint32_t value)
{
    for (int i = 0; i < 4; ++i)
    {
        mix_bytes(
            hash, static_cast<unsigned char>((value >> (8 * i)) & 0xffu));
    }
}

inline void mix_u64(std::uint64_t &hash, std::uint64_t value)
{
    for (int i = 0; i < 8; ++i)
    {
        mix_bytes(
            hash, static_cast<unsigned char>((value >> (8 * i)) & 0xffu));
    }
}

inline std::uint64_t footprint_of(Codelet &codelet, Task::Buf const *bufs,
    std::size_t nbufs, void const *cl_data, std::size_t cl_size)
{
    std::uint64_t hash = 14695981039346656037ull;
    mix_u32(hash, codelet.id());
    // The diagnostic kind label the key has always mixed (Cuda = 1);
    // value kept for key stability.
    mix_u32(hash, static_cast<std::uint32_t>(Kind::Cuda));
    if (codelet.footprint_ != nullptr)
    {
        mix_u64(hash, codelet.footprint_(cl_data, cl_size));
    }
    else
    {
        for (std::size_t i = 0; i < nbufs; ++i)
        {
            mix_u64(hash, bufs[i].handle->impl_->nbytes);
        }
    }
    return hash;
}

// Chunked arena + free list. Allocation happens on the submitter
// thread under submit_mu; completed tasks return to the free list from
// the finish poll under rt_mu, so the free list has its own tiny
// mutex. A deque keeps every Task at a stable address.
class TaskArena
{
  public:
    Task *alloc()
    {
        {
            std::lock_guard<std::mutex> lock(free_mu_);
            if (free_head_ != nullptr)
            {
                Task *t = free_head_;
                free_head_ = t->next;
                t->~Task();
                new (t) Task();
                return t;
            }
        }
        chunk_.emplace_back();
        return &chunk_.back();
    }

    void free(Task *t)
    {
        t->destroy_cl();
        t->linked.store(false, std::memory_order_relaxed);
        t->pending.store(false, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(free_mu_);
        t->next = free_head_;
        free_head_ = t;
    }

  private:
    std::deque<Task> chunk_;
    Task *free_head_ = nullptr;
    std::mutex free_mu_;
};

// The submitter-side pending chain. insert() appends fully built
// tasks under submit_mu (the ONLY producer, the host thread);
// close-and-plan takes the whole chain (submit_mu) and the batch it
// becomes is planned and executed to completion before the chain can
// be read again — so the chain needs no second cursor set and no
// The submitter and the engine: one producer, one take.
struct TaskQueue
{
    std::atomic<Task *> pending_head_{nullptr}; // appended, not taken
    Task *pending_tail_ = nullptr;              // submit_mu only

    bool empty() const
    {
        return pending_head_.load(std::memory_order_relaxed) == nullptr;
    }

    bool has_pending() const
    {
        return pending_head_.load(std::memory_order_relaxed) != nullptr;
    }

    // Submitter side (submit_mu): splice a fully built task chain.
    void append_chain(Task *first, Task *last)
    {
        last->next = nullptr;
        first->prev = nullptr;
        for (Task *t = first; t != nullptr; t = t == last ? nullptr : t->next)
        {
            t->linked.store(true, std::memory_order_relaxed);
            t->pending.store(true, std::memory_order_relaxed);
        }
        if (pending_tail_ == nullptr)
        {
            pending_tail_ = last;
            pending_head_.store(first, std::memory_order_relaxed);
        }
        else
        {
            pending_tail_->next = first;
            first->prev = pending_tail_;
            pending_tail_ = last;
        }
    }

    // Engine side (submit_mu): take the whole submitted chain, in
    // submit order. The chain's internal prev/next links stay intact
    // for the walk; the queue is left empty.
    void take_pending(Task *&first, Task *&last)
    {
        first = pending_head_.load(std::memory_order_relaxed);
        last = pending_tail_;
        pending_head_.store(nullptr, std::memory_order_relaxed);
        pending_tail_ = nullptr;
    }
};

} // namespace nnhaul
