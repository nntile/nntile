/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file tests/test_footprint.cc
 * The task symbol's diagnostic history key (src/task.hh): the FNV-1a
 * mix pinned against the published test vectors, the two arms
 * footprint_of keys on (the codelet's footprint fn result, else the
 * per-buffer sizes), and the same rule visible through the public task
 * views. The duration-history sampling the old suite pinned
 * (sample_of, predicted_ns) died with the reactive engine.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "check.hh"
#include "../src/task.hh"
#include "nnhaul/test_hooks.hh"

#include <cstdint>
#include <cstring>
#include <string>

namespace
{

// The published FNV-1a 64 test vectors ("a", "foobar"): the basis
// 14695981039346656037 and prime 1099511628211 over the byte mix are
// the algorithm, pinned from outside the implementation.
std::uint64_t fnv_of_char_chain(char const *s)
{
    std::uint64_t hash = 14695981039346656037ull;
    for (char const *p = s; *p != '\0'; ++p)
    {
        nnhaul::mix_bytes(
            hash, static_cast<unsigned char>(*p));
    }
    return hash;
}

std::uint64_t footprint_fn_copy(
    void const *cl_args, std::size_t cl_arg_size) noexcept
{
    std::uint64_t v = 0;
    if (cl_arg_size == sizeof(v) && cl_args != nullptr)
    {
        std::memcpy(&v, cl_args, sizeof(v));
    }
    return v;
}

void noop_cuda(void **buffers, void *cl_args) noexcept
{
    (void) buffers;
    (void) cl_args;
}

// The pure function's two arms, hand-driven over task.hh's own types.
void check_footprint_of()
{
    using nnhaul::Access;
    nnhaul::Handle small(256);
    nnhaul::Handle large(4096);
    small.set_name("small");
    large.set_name("large");

    nnhaul::Codelet plain("fp-plain", nullptr, nullptr, nullptr);
    nnhaul::Codelet keyed("fp-keyed", nullptr, nullptr, footprint_fn_copy);
    nnhaul::Codelet other("fp-other", nullptr, nullptr, nullptr);

    nnhaul::Task::Buf bufs[1];
    bufs[0].handle = &small;
    bufs[0].access = Access::Write;

    // Same inputs: byte-identical keys, twice in a row.
    std::uint64_t const a1 = nnhaul::footprint_of(
        plain, bufs, 1, nullptr, 0);
    std::uint64_t const a2 = nnhaul::footprint_of(
        plain, bufs, 1, nullptr, 0);
    CHECK(a1 == a2);

    // The size arm: a different operand nbytes moves the key.
    nnhaul::Task::Buf big;
    big.handle = &large;
    big.access = Access::Write;
    std::uint64_t const a3 =
        nnhaul::footprint_of(plain, &big, 1, nullptr, 0);
    CHECK(a3 != a1);

    // The codelet identity is mixed: two codelets over one shape
    // differ (ids are assigned densely from 1, never equal here).
    CHECK(nnhaul::footprint_of(other, bufs, 1, nullptr, 0) != a1);

    // The fn arm: a codelet WITH a footprint fn keys on its result,
    // not the buffer sizes — same shape, different cl_args, different
    // keys; same cl_args over different shapes, SAME key.
    std::uint64_t const one = 1;
    std::uint64_t const two = 2;
    std::uint64_t const k1 = nnhaul::footprint_of(
        keyed, bufs, 1, &one, sizeof(one));
    std::uint64_t const k2 = nnhaul::footprint_of(
        keyed, bufs, 1, &two, sizeof(two));
    CHECK(k1 != k2);
    CHECK(nnhaul::footprint_of(keyed, &big, 1, &one, sizeof(one)) == k1);

    // A null fn keys on the sizes: the plain codelet's key changes
    // with the shape (a1 vs a3 above), the keyed one's does not.
}

} // namespace

int main()
{
    // ---- the FNV-1a mix against the published vectors ----
    CHECK(fnv_of_char_chain("a") == 0xaf63dc4c8601ec8cull);
    CHECK(fnv_of_char_chain("foobar") == 0x85944171f73967e8ull);

    // The fixed-width helpers are the byte mix LSB-first: mix_u32(v)
    // is four mix_bytes steps over v's bytes 0..3, mix_u64 the eight
    // over 0..7 — pinned against the same hand chain.
    {
        std::uint64_t via_u32 = 14695981039346656037ull;
        nnhaul::mix_u32(via_u32, 0x04030201u);
        std::uint64_t hand = 14695981039346656037ull;
        for (unsigned char b : {0x01u, 0x02u, 0x03u, 0x04u})
        {
            nnhaul::mix_bytes(hand, b);
        }
        CHECK(via_u32 == hand);

        std::uint64_t via_u64 = 14695981039346656037ull;
        nnhaul::mix_u64(via_u64, 0x0807060504030201ull);
        hand = 14695981039346656037ull;
        for (unsigned char b :
            {0x01u, 0x02u, 0x03u, 0x04u, 0x05u, 0x06u, 0x07u, 0x08u})
        {
            nnhaul::mix_bytes(hand, b);
        }
        CHECK(via_u64 == hand);
    }

    check_footprint_of();

    // ---- the same rule through the public task views ----
    {
        GpuRuntimeGuard guard(
            0, 1, std::size_t{32} << 20, 2 * nnhaul::chunk_class_bytes());
        nnhaul::Handle small(256);
        nnhaul::Handle large(4096);
        small.set_name("small");
        large.set_name("large");
        nnhaul::Codelet pause("fp-pause", nullptr, noop_cuda, nullptr);

        nnhaul::insert(
            pause, {{&small, nnhaul::Access::Write}}, nullptr, 0, 0);
        nnhaul::wait();
        nnhaul::insert(
            pause, {{&small, nnhaul::Access::Write}}, nullptr, 0, 0);
        nnhaul::wait();
        nnhaul::insert(
            pause, {{&large, nnhaul::Access::Write}}, nullptr, 0, 0);
        nnhaul::wait();

        std::uint64_t first = 0;
        int seen = 0;
        std::uint64_t large_key = 0;
        for (auto const &task : nnhaul::tasks())
        {
            if (task.codelet != "fp-pause" ||
                task.phase != nnhaul::TaskPhase::Done)
            {
                continue;
            }
            if (seen == 0)
            {
                first = task.footprint;
            }
            else if (seen == 1)
            {
                CHECK(task.footprint == first); // same shape, same key.
            }
            else
            {
                large_key = task.footprint;
            }
            seen += 1;
        }
        CHECK(seen == 3);
        CHECK(first != 0);
        CHECK(large_key != first); // a different shape, a different key.
    }

    return g_checks_failed == 0 ? 0 : 1;
}
