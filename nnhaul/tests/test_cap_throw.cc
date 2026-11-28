/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file tests/test_cap_throw.cc
 * A tile larger than the GPU cap is rejected at insert with a cap error.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "check.hh"
#include "nnhaul/test_hooks.hh"

#include <string>

namespace
{

void flag_kernel(void **buffers, void *cl_args) noexcept
{
    (void) buffers;
    auto *flag = *static_cast<int **>(cl_args);
    *flag = 1;
}

} // namespace

int main()
{
    // Whole-class chunks are the unit of allocation: the cap of one class
    // backs a one-chunk arena; a tile at or below the class fits, a
    // tile needing a second chunk cannot be covered and fails loudly.
    std::size_t const cap = nnhaul::chunk_class_bytes();
    GpuRuntimeGuard guard(0, 1, 1u << 20, cap);

    nnhaul::Handle fit(cap);
    fit.set_name("fit-tile");
    int ran_fit = 0;
    int ran_big = 0;
    nnhaul::Codelet fit_op("fit", nullptr, flag_kernel, nullptr);
    nnhaul::Codelet big_op("big", nullptr, flag_kernel, nullptr);

    int *fit_flag = &ran_fit;
    nnhaul::insert(
        fit_op, {{&fit, nnhaul::Access::Write}}, &fit_flag, sizeof(fit_flag));
    nnhaul::wait();
    CHECK(ran_fit == 1);
    fit.invalidate();
    nnhaul::wait();

    // A handle far beyond any bin: acquire(Write) must throw from the
    // size check, not index past the host-page pool's bin arrays (the
    // write path allocates the shadow page before any cap gate).
    nnhaul::Handle giant(1ull << 48);
    giant.set_name("giant-tile");
    bool giant_threw = false;
    try
    {
        auto a = giant.acquire(nnhaul::Access::Write);
        (void) a;
    }
    catch (nnhaul::Error const &err)
    {
        giant_threw = true;
        std::string const what = err.what();
        CHECK(what.find("too large") != std::string::npos);
    }
    CHECK(giant_threw);

    nnhaul::Handle big(2 * cap);
    big.set_name("big-tile");
    bool threw = false;
    try
    {
        int *big_flag = &ran_big;
        nnhaul::insert(big_op,
            {{&big, nnhaul::Access::Write}},
            &big_flag,
            sizeof(big_flag));
        nnhaul::wait();
    }
    catch (nnhaul::Error const &err)
    {
        threw = true;
        std::string const what = err.what();
        // The impossible tile fails loudly at one of the two lawful
        // layers: the submit-time caller-defect fast-fail ("too large
        // for every node pool") when no node's pool covers the whole
        // tile, or the plan-time pool-exhaustion census when a larger
        // host pool admits the insert but the task's own node cannot
        // cover its working set .
        // Which fires depends on the host pool's lawful sizing; both
        // abort cleanly with the kernel never run.
        CHECK(what.find("too large") != std::string::npos ||
            what.find("pool-exhaustion") != std::string::npos);
    }
    CHECK(threw);
    CHECK(ran_big == 0);
    CHECK(nnhaul::kernel_launches() == 1);

    return g_checks_failed == 0 ? 0 : 1;
}
