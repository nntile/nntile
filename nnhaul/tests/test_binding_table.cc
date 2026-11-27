/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file tests the
 * binding table: the free lists as derived views, the fold's O(1)
 * transitions, replica outdating across N segments, and the boundary
 * halves.
 *
 * @file tests/test_binding_table.cc
 * The N-segment binding table's suite (replacing the chunk_manager
 * slots): FreeList's seed/pop/push/remove discipline with
 * ascending pop order and loud misuse; the all-free init; a scripted
 * fold whose every op is followed by the free-list-equals-complement
 * full scan; replica outdating across nodes (the TILE_OP write
 * semantics — the writer stays actual, every other replica turns
 * outdated, actual ones included, unlinked nodes
 * untouched); equals()/snapshot() round trips with a one-flag and a
 * one-binding difference caught on EVERY node's segment; a 10k-op
 * churn over small pools checked against a shadow fold; and the loud
 * programmer-misuse surface of the header's contract.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "../src/binding_table.hh"

// check.hh's shared macro pulls the OLD public header (nnhaul/nnhaul.hh)
// beside the new contract types, and the two disagree on nnhaul::Access
// today — this suite needs no runtime guards, so the CHECK macro stands
// here alone, same shape as check.hh's.
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using namespace nnhaul;

int g_checks_failed = 0;

void check_impl(bool ok, char const *expr, char const *file, int line)
{
    if (!ok)
    {
        std::cerr << "CHECK failed: " << expr << " (" << file << ":" << line
                  << ")\n";
        g_checks_failed += 1;
    }
}

#define CHECK(cond)                                                           \
    check_impl(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

// A loud-throw probe: runs f, true iff it threw std::runtime_error,
// and when needle is given also demands the census-style message carry
// it .
template <typename F>
bool throws_loud(F &&f, char const *needle = nullptr)
{
    try
    {
        f();
    }
    catch (std::runtime_error const &e)
    {
        if (needle != nullptr &&
            std::string(e.what()).find(needle) == std::string::npos)
        {
            std::cerr << "CHECK failed: the throw message lacks '"
                      << needle << "': " << e.what() << "\n";
            g_checks_failed += 1;
        }
        return true;
    }
    catch (...)
    {
        std::cerr << "CHECK failed: a non-runtime_error exception\n";
        g_checks_failed += 1;
        return true;
    }
    return false;
}

EntryBinding binding(NodeId node, ChunkId chunk, TileId tile,
    ChunkIndex index, bool actual)
{
    EntryBinding b;
    b.node = node;
    b.chunk = chunk;
    b.tile = tile;
    b.index = index;
    b.actual = actual;
    return b;
}

// The full invariant scan — the derived-view guarantee and every query
// agreeing, for every node, every chunk: the free VIEW's membership
// equals the dense complement , at() agrees with linked()
// and flag_actual(), the reverse map names every linked binding
// , free_count matches the scan, snapshot() is the
// (node, chunk)-sorted truth, and equals() round trips it.
void check_invariants(
    BindingTable const &t, std::vector<std::size_t> const &cards)
{
    std::vector<EntryBinding> expected;
    for (std::size_t n = 0; n < cards.size(); ++n)
    {
        std::size_t free_seen = 0;
        for (std::size_t c = 0; c < cards[n]; ++c)
        {
            bool const is_free = !t.linked(
                static_cast<NodeId>(n), static_cast<ChunkId>(c));
            CHECK(t.free(static_cast<NodeId>(n),
                static_cast<ChunkId>(c)) == is_free);
            auto const b = t.at(
                static_cast<NodeId>(n), static_cast<ChunkId>(c));
            if (is_free)
            {
                CHECK(!b.has_value());
                free_seen += 1;
                continue;
            }
            CHECK(b.has_value());
            CHECK(t.flag_actual(static_cast<NodeId>(n),
                static_cast<ChunkId>(c)) == b->second);
            CHECK(t.binding_of(static_cast<NodeId>(n), b->first) ==
                std::optional<ChunkId>(static_cast<ChunkId>(c)));
            expected.push_back(binding(static_cast<NodeId>(n),
                static_cast<ChunkId>(c), b->first.tile, b->first.index,
                b->second));
        }
        CHECK(t.free_count(static_cast<NodeId>(n)) == free_seen);
        // pop_free: nullopt exactly when the node is exhausted, else a
        // chunk the free view itself calls free (the peek never pops —
        // the removal happens at link, the derived-view guarantee).
        auto const next = t.pop_free(static_cast<NodeId>(n));
        if (free_seen == 0)
        {
            CHECK(!next.has_value());
        }
        else
        {
            CHECK(next.has_value());
            CHECK(t.free(static_cast<NodeId>(n), *next));
            CHECK(!t.linked(static_cast<NodeId>(n), *next));
        }
    }
    std::vector<EntryBinding> const snap = t.snapshot();
    CHECK(snap.size() == expected.size());
    for (std::size_t i = 0; i < snap.size() && i < expected.size(); ++i)
    {
        CHECK(snap[i] == expected[i]);
    }
    CHECK(t.equals(snap)); // the round trip (the exit fold feeds the
                           // successor's boundary check)
}

// ---- the FreeList's own discipline  ----

void test_free_list()
{
    FreeList fl;

    // Unseeded: the id space is empty, every act is out of range and
    // loud (programmer misuse, never serviced).
    CHECK(throws_loud([&] { (void) fl.member(0); }));
    CHECK(throws_loud([&] { fl.push(0); }));
    CHECK(throws_loud([&] { fl.remove(0); }));

    // Ascending pop order after seed: 0, 1, 2, ... then nullopt.
    fl.seed(5);
    CHECK(fl.size() == 5);
    for (ChunkId i = 0; i < 5; ++i)
    {
        CHECK(fl.member(i));
        CHECK(fl.pop() == std::optional<ChunkId>(i));
    }
    CHECK(fl.pop() == std::nullopt);
    CHECK(fl.size() == 0);

    // Membership invariants under push/remove: remove takes the named
    // id wherever it sits (swap-with-back), pop keeps taking the back.
    fl.seed(6);
    for (ChunkId i = 0; i < 6; ++i)
    {
        CHECK(fl.member(i));
    }
    CHECK(fl.pop() == std::optional<ChunkId>(0));
    CHECK(fl.pop() == std::optional<ChunkId>(1));
    CHECK(fl.size() == 4);
    fl.remove(3); // mid-list: ids [5,4,3,2] -> [5,4,2] (the back, 2,
                  // swaps into 3's slot)
    CHECK(!fl.member(3));
    CHECK(fl.member(2));
    CHECK(fl.member(4));
    CHECK(fl.member(5));
    CHECK(fl.size() == 3);
    CHECK(fl.pop() == std::optional<ChunkId>(2));
    CHECK(fl.pop() == std::optional<ChunkId>(4));
    CHECK(fl.pop() == std::optional<ChunkId>(5));
    CHECK(fl.pop() == std::nullopt);

    // push restores membership; the pushed id is the next pop.
    fl.push(3);
    fl.push(1);
    CHECK(fl.member(3));
    CHECK(fl.member(1));
    CHECK(fl.size() == 2);
    CHECK(fl.pop() == std::optional<ChunkId>(1));
    CHECK(fl.pop() == std::optional<ChunkId>(3));

    // Loud misuse: duplicate push, non-member remove, out-of-space ids.
    fl.push(2);
    CHECK(throws_loud([&] { fl.push(2); }, "duplicate member"));
    CHECK(throws_loud([&] { fl.remove(4); }, "non-member"));
    CHECK(throws_loud([&] { (void) fl.member(6); }));
    CHECK(throws_loud([&] { fl.push(6); }));
    CHECK(throws_loud([&] { fl.remove(6); }));
    CHECK(fl.member(2)); // untouched by the refused acts
    CHECK(fl.size() == 1);
}

// ---- init: N segments, all free  ----

void test_init_all_free()
{
    std::vector<std::size_t> const cards{3, 2, 4};
    BindingTable const t(cards);
    CHECK(t.node_count() == 3);
    CHECK(t.pool_cardinality(0) == 3);
    CHECK(t.pool_cardinality(1) == 2);
    CHECK(t.pool_cardinality(2) == 4);
    CHECK(t.snapshot().empty());
    CHECK(t.equals({})); // the all-free world is the empty entry state
    for (std::size_t n = 0; n < cards.size(); ++n)
    {
        CHECK(t.free_count(static_cast<NodeId>(n)) == cards[n]);
        // the seed's ascending handout: the first peek of every node
        CHECK(t.pop_free(static_cast<NodeId>(n)) ==
            std::optional<ChunkId>(0));
    }

    // The census render: one segment per node, pools right, no
    // bindings, the free list printed beside at full size.
    std::vector<SegmentSnapshot> const render = t.render();
    CHECK(render.size() == 3);
    for (std::size_t n = 0; n < render.size(); ++n)
    {
        CHECK(render[n].node == static_cast<NodeId>(n));
        CHECK(render[n].pool == cards[n]);
        CHECK(render[n].bindings.empty());
        CHECK(render[n].free_list.size() == cards[n]);
    }

    // The default constructor is the empty table: every act is loud.
    BindingTable bare;
    CHECK(bare.node_count() == 0);
    CHECK(throws_loud([&] { (void) bare.linked(0, 0); }, "names node"));
}

// ---- a scripted fold, complement-scanned after EVERY op ----

void test_scripted_sequence()
{
    std::vector<std::size_t> const cards{3, 2};
    BindingTable t(cards);
    check_invariants(t, cards);

    LogicalChunk const lc11{1, 1};
    LogicalChunk const lc20{2, 0};

    t.link(0, 2, lc11);
    check_invariants(t, cards);
    CHECK(!t.flag_actual(0, 2)); // LINK creates OUTDATED

    t.link(1, 0, lc11); // the replica of lc11 on node 1
    check_invariants(t, cards);
    CHECK(*t.binding_of(0, lc11) == 2u);
    CHECK(*t.binding_of(1, lc11) == 0u);

    t.actualize(0, 2);
    check_invariants(t, cards);
    CHECK(t.flag_actual(0, 2));
    CHECK(!t.flag_actual(1, 0)); // COPY's no-other-flag shape: the
                                 // other replica keeps its flag

    t.outdate_replicas(lc11, 0, 2);
    check_invariants(t, cards);
    CHECK(t.flag_actual(0, 2)); // the writer stays actual
    CHECK(!t.flag_actual(1, 0)); // the other replica is outdated

    t.link(1, 1, lc20);
    check_invariants(t, cards);
    t.actualize(1, 1);
    check_invariants(t, cards);
    t.outdate_replicas(lc20, 1, 1);
    check_invariants(t, cards);
    CHECK(t.flag_actual(1, 1)); // the write's own binding

    t.unlink(0, 2); // the flag drops with the binding
    check_invariants(t, cards);
    CHECK(!t.linked(0, 2));
    CHECK(t.at(0, 2) == std::nullopt);
    CHECK(t.free(0, 2));
    CHECK(t.binding_of(0, lc11) == std::nullopt);
    CHECK(*t.binding_of(1, lc11) == 0u); // UNLINK touches no replica
                                         //

    t.link(0, 0, lc11); // the relink: lc11 repatriated to node 0
    check_invariants(t, cards);
    CHECK(!t.flag_actual(0, 0)); // fresh bindings start OUTDATED

    t.actualize(0, 0);
    check_invariants(t, cards);

    // The boundary half against a hand-built entry state: exact, then
    // one flag off, then one binding off.
    std::vector<EntryBinding> const expected{binding(0, 0, 1, 1, true),
        binding(1, 0, 1, 1, false), binding(1, 1, 2, 0, true)};
    CHECK(t.equals(expected));
    CHECK(t.snapshot() == expected); // entry-for-entry, same order
    std::vector<EntryBinding> reversed(expected.rbegin(), expected.rend());
    CHECK(!t.equals(reversed)); // unsorted entry is a difference: the
                                // plan sorts Plan.entry by (node,
                                // chunk) and equals demands it
    {
        std::vector<EntryBinding> one_flag_off = expected;
        one_flag_off[1].actual = true;
        CHECK(!t.equals(one_flag_off));
        std::vector<EntryBinding> one_binding_off = expected;
        one_binding_off[2].tile = 9;
        CHECK(!t.equals(one_binding_off));
    }
}

// ---- replica outdating across N segments  ----

void test_replica_outdating()
{
    std::vector<std::size_t> const cards{4, 3, 3, 2};
    BindingTable t(cards);
    LogicalChunk const lc{7, 3};
    LogicalChunk const other{5, 9};

    // lc replicated on nodes 0, 1, 2; node 3 carries no lc binding.
    t.link(0, 0, lc);
    t.link(1, 0, lc);
    t.link(2, 0, lc);
    // An unrelated replica pair elsewhere: actual on both its nodes.
    t.link(0, 1, other);
    t.link(3, 0, other);
    t.actualize(0, 1);
    t.actualize(3, 0);

    // Every lc replica starts OUTDATED, then ALL turn actual — the
    // outdating below must catch the actual ones too.
    t.actualize(0, 0);
    t.actualize(1, 0);
    t.actualize(2, 0);
    CHECK(t.flag_actual(0, 0));
    CHECK(t.flag_actual(1, 0));
    CHECK(t.flag_actual(2, 0));

    // The TILE_OP write semantics: actualize the writer, then outdate
    // every OTHER linked replica of its logical chunk.
    t.actualize(1, 0);
    t.outdate_replicas(lc, 1, 0);
    CHECK(t.flag_actual(1, 0)); // exactly the writer stays actual
    CHECK(!t.flag_actual(0, 0)); // actual replicas included
    CHECK(!t.flag_actual(2, 0));
    CHECK(t.at(0, 0)->first == lc); // links untouched — flags only
    CHECK(t.at(2, 0)->first == lc);
    // The unlinked third node is unaffected: its own bindings and
    // flags stand, its free chunks stay free.
    CHECK(!t.linked(3, 1));
    CHECK(t.flag_actual(3, 0)); // other's replica on node 3
    CHECK(t.flag_actual(0, 1)); // other's replica on node 0
    check_invariants(t, cards);

    // A late replica links OUTDATED after the write :
    // out of date and never produced are the same state .
    t.link(3, 1, lc);
    CHECK(!t.flag_actual(3, 1));
    CHECK(*t.binding_of(3, lc) == 1u);

    // A different writer flips the pattern: node 2 writes this time.
    t.actualize(2, 0);
    t.outdate_replicas(lc, 2, 0);
    CHECK(t.flag_actual(2, 0));
    CHECK(!t.flag_actual(1, 0));
    CHECK(!t.flag_actual(0, 0));
    CHECK(!t.flag_actual(3, 1));
    // The unrelated pair still stands actual on both its nodes.
    CHECK(t.flag_actual(3, 0));
    CHECK(t.flag_actual(0, 1));
    check_invariants(t, cards);
}

// ---- equals()/snapshot() boundary halves  ----

void test_boundary()
{
    std::vector<std::size_t> const cards{2, 2, 2, 2};
    BindingTable t(cards);
    for (std::size_t n = 0; n < cards.size(); ++n)
    {
        t.link(static_cast<NodeId>(n), 0,
            LogicalChunk{static_cast<TileId>(10 + n), 0});
    }
    t.link(0, 1, LogicalChunk{20, 0}); // node 0 carries a second binding
    t.actualize(0, 0);                 // actual on nodes 0 and 2 only
    t.actualize(2, 0);
    check_invariants(t, cards);

    std::vector<EntryBinding> const snap = t.snapshot();
    CHECK(snap.size() == 5);
    // Sorted by (node, chunk) — the exit fold's own order.
    for (std::size_t i = 1; i < snap.size(); ++i)
    {
        CHECK(snap[i - 1].node < snap[i].node ||
            (snap[i - 1].node == snap[i].node &&
                snap[i - 1].chunk < snap[i].chunk));
    }
    CHECK(t.equals(snap)); // the round trip

    // An unsorted entry is a difference even when every binding is
    // faithfully listed — adjacent swap and full reversal both.
    {
        std::vector<EntryBinding> swapped = snap;
        std::swap(swapped[0], swapped[1]);
        CHECK(!t.equals(swapped));
        std::vector<EntryBinding> rev(snap.rbegin(), snap.rend());
        CHECK(!t.equals(rev));
    }

    // A one-FLAG difference on ANY node's segment is caught: flip
    // node n's own binding flag, one node at a time, restore, and the
    // table must agree again.
    for (std::size_t n = 0; n < cards.size(); ++n)
    {
        std::vector<EntryBinding> entry = snap;
        bool *flag = nullptr;
        for (EntryBinding &b : entry)
        {
            if (b.node == static_cast<NodeId>(n))
            {
                flag = &b.actual;
                break;
            }
        }
        CHECK(flag != nullptr); // every node carries a binding
        *flag = !*flag;
        CHECK(!t.equals(entry));
        *flag = !*flag;
        CHECK(t.equals(entry));
    }

    // A one-BINDING difference on any node: the tile, the index, the
    // chunk named, a dropped entry, a phantom entry on a free chunk.
    for (std::size_t n = 0; n < cards.size(); ++n)
    {
        std::vector<EntryBinding> entry = snap;
        EntryBinding *owned = nullptr;
        for (EntryBinding &b : entry)
        {
            if (b.node == static_cast<NodeId>(n))
            {
                owned = &b;
                break;
            }
        }
        CHECK(owned != nullptr);

        TileId const tile = owned->tile;
        owned->tile = tile + 100;
        CHECK(!t.equals(entry));
        owned->tile = tile;
        ChunkIndex const index = owned->index;
        owned->index = index + 7;
        CHECK(!t.equals(entry));
        owned->index = index;
        CHECK(t.equals(entry));

        std::vector<EntryBinding> dropped = entry;
        for (std::size_t i = 0; i < dropped.size(); ++i)
        {
            if (dropped[i].node == static_cast<NodeId>(n))
            {
                dropped.erase(dropped.begin() +
                    static_cast<std::ptrdiff_t>(i));
                break;
            }
        }
        CHECK(!t.equals(dropped));

        std::vector<EntryBinding> phantom = entry;
        // Node 0's top slot is LINKED — its second binding; every
        // other node's top slot is free, so the phantom runs there.
        ChunkId const free_chunk = static_cast<ChunkId>(cards[n] - 1);
        bool const top_free = !t.linked(
            static_cast<NodeId>(n), free_chunk);
        if (top_free)
        {
            phantom.push_back(
                binding(static_cast<NodeId>(n), free_chunk, 99, 9, false));
            CHECK(!t.equals(phantom));
        }
    }

    // An entry naming an unknown node or an unknown chunk: a
    // difference, never a throw (the census is the caller's).
    CHECK(!t.equals(std::vector<EntryBinding>{binding(9, 0, 10, 0, true)}));
    CHECK(!t.equals(std::vector<EntryBinding>{binding(0, 5, 10, 0, true)}));

    // The duplicate-slot sneak: node 0's (0,0) entry replaced by a
    // copy of its (0,1) — the total agrees, one slot is named twice
    // (the pair not strictly ascending) and another not at all.
    {
        std::vector<EntryBinding> dup = snap;
        std::size_t at00 = snap.size();
        std::size_t at01 = snap.size();
        for (std::size_t i = 0; i < dup.size(); ++i)
        {
            if (dup[i].node == 0 && dup[i].chunk == 0)
            {
                at00 = i;
            }
            if (dup[i].node == 0 && dup[i].chunk == 1)
            {
                at01 = i;
            }
        }
        CHECK(at00 < snap.size());
        CHECK(at01 < snap.size());
        dup[at00] = dup[at01];
        CHECK(!t.equals(dup));
    }

    // A size mismatch on a single-binding node: duplicating it.
    {
        std::vector<EntryBinding> dup = snap;
        for (EntryBinding const &b : snap)
        {
            if (b.node == 1)
            {
                dup.push_back(b);
                break;
            }
        }
        CHECK(!t.equals(dup));
    }

    // The all-free table equals the empty entry and nothing naming a
    // slot.
    BindingTable all_free(cards);
    CHECK(all_free.equals({}));
    CHECK(!all_free.equals(
        std::vector<EntryBinding>{binding(0, 0, 1, 0, false)}));
}

// ---- the churn: 10k transitions against a shadow fold ----

void test_churn()
{
    std::vector<std::size_t> const cards{8, 5, 3, 7};
    BindingTable t(cards);

    // The shadow fold: the test's own reference semantics for every
    // transition, compared against the table at the end.
    struct Slot
    {
        bool linked = false;
        TileId tile = 0;
        ChunkIndex index = 0;
        bool actual = false;
    };
    std::vector<std::vector<Slot>> shadow(
        cards.size(), std::vector<Slot>());
    for (std::size_t n = 0; n < cards.size(); ++n)
    {
        shadow[n].resize(cards[n]);
    }

    std::mt19937 rng(42u);
    std::uint64_t fresh = 0;
    auto next_lc = [&fresh]() {
        fresh += 1;
        return LogicalChunk{static_cast<TileId>(fresh >> 8),
            static_cast<ChunkIndex>(fresh & 0xffu)};
    };

    std::size_t const k_ops = 10000;
    for (std::size_t step = 0; step < k_ops; ++step)
    {
        std::size_t const n = rng() % cards.size();
        std::size_t const c = rng() % cards[n];
        NodeId const node = static_cast<NodeId>(n);
        ChunkId const chunk = static_cast<ChunkId>(c);
        Slot &s = shadow[n][c];
        std::uint32_t const op = rng() % 4;
        if (op == 0)
        {
            // Link onto a free chunk with a fresh logical chunk (never
            // a double-binding by construction).
            if (!s.linked)
            {
                LogicalChunk const lc = next_lc();
                t.link(node, chunk, lc);
                s.linked = true;
                s.tile = lc.tile;
                s.index = lc.index;
                s.actual = false;
            }
        }
        else if (op == 1)
        {
            // Unlink a linked chunk — the relink churn's other half.
            if (s.linked)
            {
                t.unlink(node, chunk);
                s.linked = false;
                s.actual = false;
            }
        }
        else if (op == 2)
        {
            if (s.linked)
            {
                t.actualize(node, chunk);
                s.actual = true;
            }
        }
        else
        {
            // The TILE_OP write semantics on a random writer.
            if (s.linked)
            {
                LogicalChunk const lc{s.tile, s.index};
                t.actualize(node, chunk);
                t.outdate_replicas(lc, node, chunk);
                s.actual = true;
                for (std::size_t n2 = 0; n2 < shadow.size(); ++n2)
                {
                    for (std::size_t c2 = 0; c2 < shadow[n2].size(); ++c2)
                    {
                        Slot &o = shadow[n2][c2];
                        if (o.linked && o.tile == lc.tile &&
                            o.index == lc.index && !(n2 == n && c2 == c))
                        {
                            o.actual = false;
                        }
                    }
                }
            }
        }

        // The periodic full scan (every op of the scripted section
        // above; the churn scans every 500 — pools are small).
        if (step % 500 == 499)
        {
            check_invariants(t, cards);
        }
    }

    // Correctness after churn: the table equals the shadow fold,
    // binding for binding, flag for flag, in (node, chunk) order.
    std::vector<EntryBinding> expected;
    for (std::size_t n = 0; n < cards.size(); ++n)
    {
        for (std::size_t c = 0; c < cards[n]; ++c)
        {
            if (shadow[n][c].linked)
            {
                expected.push_back(binding(static_cast<NodeId>(n),
                    static_cast<ChunkId>(c), shadow[n][c].tile,
                    shadow[n][c].index, shadow[n][c].actual));
            }
        }
    }
    CHECK(t.snapshot() == expected);
    CHECK(t.equals(expected));
    check_invariants(t, cards);
}

// ---- the loud programmer-misuse surface (the header's contract) ----

void test_misuse()
{
    BindingTable t(std::vector<std::size_t>{2, 3});

    // Out-of-range nodes and chunks: loud, on every act.
    CHECK(throws_loud([&] { t.link(2, 0, LogicalChunk{1, 0}); },
        "names node"));
    CHECK(throws_loud([&] { t.link(0, 5, LogicalChunk{1, 0}); },
        "names chunk"));
    CHECK(throws_loud([&] { t.unlink(7, 0); }));
    CHECK(throws_loud([&] { t.unlink(0, 3); }));
    CHECK(throws_loud([&] { t.actualize(2, 0); }));
    CHECK(throws_loud([&] { t.actualize(0, 3); }));
    CHECK(throws_loud([&] { (void) t.linked(2, 0); }));
    CHECK(throws_loud([&] { (void) t.linked(0, 3); }));
    CHECK(throws_loud([&] { (void) t.at(2, 0); }));
    CHECK(throws_loud([&] { (void) t.at(0, 3); }));
    CHECK(throws_loud([&] { (void) t.flag_actual(2, 0); }));
    CHECK(throws_loud([&] { (void) t.binding_of(5, LogicalChunk{1, 1}); }));
    CHECK(throws_loud([&] { (void) t.pop_free(9); }));
    CHECK(throws_loud([&] { (void) t.free_count(9); }));
    CHECK(throws_loud([&] { (void) t.free(9, 0); }));
    CHECK(throws_loud([&] { (void) t.free(0, 9); }));
    CHECK(throws_loud([&] { (void) t.pool_cardinality(200); }));
    CHECK(throws_loud([&] { t.outdate_replicas(LogicalChunk{1, 0}, 4, 0); }));

    // flag_actual on a FREE chunk (in range): the flag is a binding's
    // property  — loud.
    CHECK(throws_loud([&] { (void) t.flag_actual(0, 0); }, "free"));

    // The FreeList discipline surfacing through the transitions: a
    // caller that skipped the checker's rows fails loud, not silent.
    t.link(0, 0, LogicalChunk{1, 0});
    CHECK(throws_loud([&] { t.link(0, 0, LogicalChunk{2, 2}); },
        "non-member")); // double-link misuse
    CHECK(throws_loud([&] { t.link(0, 1, LogicalChunk{1, 0}); },
        "reverse map")); // double-binding misuse
    t.unlink(0, 0);
    CHECK(throws_loud([&] { t.unlink(0, 0); },
        "duplicate member")); // unlink-free misuse

    // The refused acts left the table coherent: all free again.
    check_invariants(t, std::vector<std::size_t>{2, 3});

    // A cardinality the ChunkId space cannot address: loud at init,
    // before any allocation is attempted.
    CHECK(throws_loud([] {
        BindingTable bad(std::vector<std::size_t>{
            1, static_cast<std::size_t>(FreeList::kNotFree) + 1});
    }, "overflows"));
}

} // namespace

int main()
{
    test_free_list();
    test_init_all_free();
    test_scripted_sequence();
    test_replica_outdating();
    test_boundary();
    test_churn();
    test_misuse();

    if (g_checks_failed != 0)
    {
        std::cerr << g_checks_failed << " check(s) failed\n";
    }
    return g_checks_failed == 0 ? 0 : 1;
}
