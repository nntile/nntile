/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                 (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/include/nntile/op_arena.hh
 * Flat arenas for IR op nodes: chunked bump allocation with per-chunk
 * live counts, plus fixed-capacity inline edge spans.
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <new>
#ifdef __linux__
#include <sys/mman.h>  // mmap, madvise, MADV_HUGEPAGE
#endif
#include <utility>
#include <vector>

namespace nntile
{

//! Fixed-capacity inline span of node pointers (op inputs / outputs).
//!
//! The recorded IR's per-op edge vectors were one heap allocation per edge
//! list per op (~2 mallocs/op scattered across the heap); every compile
//! pass then pointer-chased them past LLC. This span stores up to ``Cap``
//! edges INLINE in the op node (zero allocations; the object itself comes
//! from the op arena, so consecutive ops' edges are consecutive memory)
//! and spills beyond 12 inline slots; reachable via torch.cat with >12
//! inputs.
//!
//! Copy semantics: deep copy (a clone owns its own spill vector).
template <class NodeT, std::size_t Cap>
class OpEdgeSpan
{
  public:
    using const_iterator = NodeT *const *;

    OpEdgeSpan() noexcept = default;

    OpEdgeSpan(const OpEdgeSpan &other)
        : size_(other.size_)
        , spill_(other.spill_ == nullptr
                  ? nullptr
                  : new std::vector<NodeT *>(*other.spill_))
    {
        for (std::size_t i = 0; i < size_; ++i)
        {
            inline_[i] = other.inline_[i];
        }
    }

    OpEdgeSpan &operator=(const OpEdgeSpan &other)
    {
        if (this != &other)
        {
            std::vector<NodeT *> *const fresh =
                other.spill_ == nullptr
                    ? nullptr
                    : new std::vector<NodeT *>(*other.spill_);
            delete spill_;
            spill_ = fresh;
            size_ = other.size_;
            for (std::size_t i = 0; i < size_; ++i)
            {
                inline_[i] = other.inline_[i];
            }
        }
        return *this;
    }

    OpEdgeSpan(OpEdgeSpan &&other) noexcept
        : size_(other.size_)
        , spill_(other.spill_)
    {
        other.size_ = 0;
        other.spill_ = nullptr;
        for (std::size_t i = 0; i < size_; ++i)
        {
            inline_[i] = other.inline_[i];
        }
    }

    OpEdgeSpan &operator=(OpEdgeSpan &&other) noexcept
    {
        if (this != &other)
        {
            delete spill_;
            spill_ = other.spill_;
            size_ = other.size_;
            for (std::size_t i = 0; i < size_; ++i)
            {
                inline_[i] = other.inline_[i];
            }
            other.size_ = 0;
            other.spill_ = nullptr;
        }
        return *this;
    }

    ~OpEdgeSpan()
    {
        delete spill_;
    }

    OpEdgeSpan &operator=(std::initializer_list<NodeT *> list)
    {
        clear();
        for (NodeT *p : list)
        {
            push_back(p);
        }
        return *this;
    }

    void assign(std::initializer_list<NodeT *> list)
    {
        (*this) = list;
    }

    //! Iterator-pair form: torch_dispatch's lowered builders populate
    //! edges from iterator ranges (NNTILE_TORCH_NATIVE_OPS config).
    template <class It>
    void assign(It first, It last)
    {
        clear();
        for (It it = first; it != last; ++it)
        {
            push_back(*it);
        }
    }

    void push_back(NodeT *p)
    {
        if (spill_ != nullptr)
        {
            spill_->push_back(p);
            return;
        }
        if (size_ < Cap)
        {
            inline_[size_] = p;
            ++size_;
            return;
        }
        // Spill: migrate the inline prefix so the span stays ONE
        // contiguous range (begin()/end() iterate a single block).
        std::vector<NodeT *> *const fresh =
            new std::vector<NodeT *>(inline_, inline_ + Cap);
        fresh->push_back(p);
        spill_ = fresh;
    }

    void clear()
    {
        if (spill_ != nullptr)
        {
            spill_->clear();
        }
        size_ = 0;
    }

    const_iterator begin() const
    {
        return spill_ != nullptr ? spill_->data() : inline_;
    }

    const_iterator end() const
    {
        return begin() + size();
    }

    NodeT *const *data() const
    {
        return begin();
    }

    NodeT *operator[](std::size_t i) const
    {
        return begin()[i];
    }

    std::size_t size() const
    {
        return spill_ != nullptr ? spill_->size()
                                 : static_cast<std::size_t>(size_);
    }

    bool empty() const
    {
        return size() == 0;
    }

  private:
    std::uint8_t size_ = 0;
    NodeT *inline_[Cap] = {};
    std::vector<NodeT *> *spill_ = nullptr;
};

//! Chunked bump arena for IR op nodes (``make_op`` below).
//!
//! Allocations bump within large chunks; each allocation carries a hidden
//! header (placed directly before the user block) naming its chunk, and
//! deallocation drops that chunk's live count. A chunk whose live count
//! reaches zero is RECYCLED into the persistent pre-touched pool (never
//! returned to the kernel; pages stay resident for the next phase - the
//! owner's preallocate-and-touch order), so process RSS is bounded by the
//! high-water of live IR, not the lifetime total. Layout is
//! creation-ordered and contiguous, which is the point: compile passes
//! stream over it instead of pointer-chasing hundreds of thousands of
//! scattered small blocks.
class OpArena
{
  public:
    OpArena() = default;
    OpArena(const OpArena &) = delete;
    OpArena &operator=(const OpArena &) = delete;

    void *allocate(std::size_t n, std::size_t align)
    {
        if (align > kAlignment)
        {
            // No shipped IR type is over-aligned; fail loudly if one
            // appears rather than silently misaligning.
            throw std::bad_alloc();
        }
        std::lock_guard<std::mutex> guard(lock_);
        n = round_up(n == 0 ? 1 : n);
        if (n + kAllocHeader > kChunkUsable)
        {
            throw std::bad_alloc();
        }
        if (current_ == nullptr
            || current_->used + kAllocHeader + n > kChunkUsable)
        {
            new_chunk_locked();
        }
        Chunk *const chunk = current_;
        char *const hdr_mem = chunk_data(chunk) + chunk->used;
        chunk->used += kAllocHeader + n;
        ++chunk->live;
        AllocHeader *const hdr =
            reinterpret_cast<AllocHeader *>(hdr_mem);
        hdr->chunk = chunk;
        return hdr_mem + kAllocHeader;
    }

    void deallocate(void *p) noexcept
    {
        if (p == nullptr)
        {
            return;
        }
        std::lock_guard<std::mutex> guard(lock_);
        AllocHeader *const hdr = reinterpret_cast<AllocHeader *>(
            static_cast<char *>(p) - static_cast<std::ptrdiff_t>(
                                         kAllocHeader));
        Chunk *const chunk = hdr->chunk;
        if (--chunk->live == 0 && chunk != current_)
        {
            free_chunk_locked(chunk);
        }
    }

  private:
    struct Chunk
    {
        std::size_t live = 0;
        std::size_t used = 0;
    };

    struct AllocHeader
    {
        Chunk *chunk = nullptr;
    };

    // 2 MiB chunks, THP-backed: the host runs transparent hugepages in
    // madvise mode, so explicitly advising MADV_HUGEPAGE gives the arena
    // one fault per 2 MiB instead of 512 per-4KiB faults - the fresh-page
    // fault storm was the last per-element cost growth with N (flat
    // in-phase chunks, but ~15-40% per-op growth across the 16->32
    // doubling while the whole working set re-faulted).
    static constexpr std::size_t kChunkBytes = 2u << 20; // 2 MiB
    static constexpr std::size_t kAlignment = 16;
    static constexpr std::size_t kAllocHeader = kAlignment;
    static constexpr std::size_t kChunkUsable =
        kChunkBytes - sizeof(Chunk);

    static std::size_t round_up(std::size_t n)
    {
        return (n + kAlignment - 1) / kAlignment * kAlignment;
    }

    static char *chunk_data(Chunk *chunk)
    {
        return reinterpret_cast<char *>(chunk) + sizeof(Chunk);
    }

    void new_chunk_locked()
    {
        sweep_locked();
        // Owner order: PREALLOCATE and PRE-TOUCH. Chunks are drawn from a
        // persistent recycled pool (never returned to the kernel), so the
        // page-fault cost is paid once at pool growth, off the compile
        // path — steady-state phases reuse already-resident pages.
        if (!recycled_.empty())
        {
            Chunk *const chunk = recycled_.back();
            recycled_.pop_back();
            chunk->live = 0;
            chunk->used = 0;
            chunks_.push_back(chunk);
            current_ = chunk;
            return;
        }
        // MADV_HUGEPAGE only grants 2 MiB pages on 2 MiB-ALIGNED virtual
        // ranges: ::operator new returns 16-byte-aligned memory, so a
        // kChunkBytes mapping there contains zero aligned 2 MiB windows
        // and the madvise is a silent no-op (measured: AnonHugePages 0).
        // Over-allocate by one chunk, carve the aligned interior, and
        // free the head/tail remainders so the mapping stays exact.
        char *mem = nullptr;
#if defined(__linux__)
        char *const raw = static_cast<char *>(
            ::mmap(nullptr, kChunkBytes * 2, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (raw == MAP_FAILED)
        {
            throw std::bad_alloc();
        }
        std::uintptr_t const aligned =
            (reinterpret_cast<std::uintptr_t>(raw) + kChunkBytes - 1)
            & ~(kChunkBytes - 1);
        mem = reinterpret_cast<char *>(aligned);
        if (mem != raw)
        {
            ::munmap(raw, static_cast<std::size_t>(mem - raw));
        }
        char *const tail = mem + kChunkBytes;
        std::size_t const tailBytes = static_cast<std::size_t>(
            raw + kChunkBytes * 2 - tail);
        if (tailBytes != 0)
        {
            ::munmap(tail, tailBytes);
        }
        ::madvise(mem, kChunkBytes, MADV_HUGEPAGE);
#else
        char *const raw = static_cast<char *>(
            ::operator new(kChunkBytes + kChunkBytes - 1));
        std::uintptr_t const aligned =
            (reinterpret_cast<std::uintptr_t>(raw) + kChunkBytes - 1)
            & ~(kChunkBytes - 1);
        mem = reinterpret_cast<char *>(aligned);
#endif
        // Pre-fault: touch every 4 KiB page now (one fault each without
        // THP; one fault per 2 MiB when the aligned THP lands), so no
        // compile phase ever pays the fresh-page fault storm.
        for (std::size_t off = 0; off < kChunkBytes; off += 4096)
        {
            mem[off] = 0;
        }
        Chunk *const chunk = reinterpret_cast<Chunk *>(mem);
        chunk->live = 0;
        chunk->used = 0;
        chunks_.push_back(chunk);
        current_ = chunk;
    }

    void sweep_locked()
    {
        chunks_.erase(
            std::remove_if(
                chunks_.begin(),
                chunks_.end(),
                [&](Chunk *chunk) {
                    if (chunk != current_ && chunk->live == 0)
                    {
                        // Recycle, never free: the pool keeps its pages
                        // resident for the next phase.
                        recycled_.push_back(chunk);
                        return true;
                    }
                    return false;
                }),
            chunks_.end());
    }

    void free_chunk_locked(Chunk *chunk)
    {
        // Erase ONLY this chunk: the erase(found, end()) form used to
        // drop the whole tail after it, destroying live chunks' slots
        // and possibly current_'s while their blocks stayed allocated.
        auto it = std::find(chunks_.begin(), chunks_.end(), chunk);
        assert(it != chunks_.end());
        chunks_.erase(it);
        recycled_.push_back(chunk);
    }

    std::mutex lock_;
    std::vector<Chunk *> chunks_;
    std::vector<Chunk *> recycled_;
    Chunk *current_ = nullptr;
};

//! The process-wide op arena. Deliberately never destroyed: IR ops may
//! outlive module teardown order at exit, and the arena recycles its
//! pre-touched chunks for the process lifetime (RSS = live-IR high-water).
inline OpArena *op_arena() noexcept
{
    static OpArena *const arena = new OpArena();
    return arena;
}

//! Stateless allocator adapter over ``op_arena()`` for
//! ``std::allocate_shared`` of IR op nodes.
template <class T>
struct OpArenaAllocator
{
    using value_type = T;

    OpArenaAllocator() noexcept = default;

    template <class U>
    OpArenaAllocator(OpArenaAllocator<U> const &) noexcept
    {
    }

    T *allocate(std::size_t n)
    {
        return static_cast<T *>(
            op_arena()->allocate(n * sizeof(T), alignof(T)));
    }

    void deallocate(T *p, std::size_t) noexcept
    {
        op_arena()->deallocate(p);
    }
};

template <class T, class U>
bool operator==(
    OpArenaAllocator<T> const &, OpArenaAllocator<U> const &) noexcept
{
    return true;
}

template <class T, class U>
bool operator!=(
    OpArenaAllocator<T> const &, OpArenaAllocator<U> const &) noexcept
{
    return false;
}

//! Allocate an IR op node (control block + object in one arena block,
//! creation-ordered and contiguous with its siblings).
template <class OpT, class... Args>
std::shared_ptr<OpT> make_op(Args &&...args)
{
    return std::allocate_shared<OpT>(
        OpArenaAllocator<OpT>{}, std::forward<Args>(args)...);
}

//! Generic arena ``make_shared`` (tile payloads etc; same mechanism as
//! ``make_op``).
template <class T, class... Args>
std::shared_ptr<T> make_arena_shared(Args &&...args)
{
    return std::allocate_shared<T>(
        OpArenaAllocator<T>{}, std::forward<Args>(args)...);
}

//! Deleter for arena-constructed objects under ``unique_ptr``
//! (graph data nodes): destroys the object, then returns the block to
//! its chunk (chunk freed when its last object dies).
template <class T>
struct ArenaNodeDeleter
{
    void operator()(T *p) const noexcept
    {
        if (p != nullptr)
        {
            p->~T();
            op_arena()->deallocate(p);
        }
    }
};

//! Construct a T in the arena, wrapped in a ``unique_ptr`` that returns
//! the block chunk-wise.
template <class T, class... Args>
std::unique_ptr<T, ArenaNodeDeleter<T>> make_arena_unique(Args &&...args)
{
    void *const mem =
        op_arena()->allocate(sizeof(T), alignof(T));
    try
    {
        T *const p = ::new (mem) T(std::forward<Args>(args)...);
        return std::unique_ptr<T, ArenaNodeDeleter<T>>(p);
    }
    catch (...)
    {
        op_arena()->deallocate(mem);
        throw;
    }
}

} // namespace nntile
