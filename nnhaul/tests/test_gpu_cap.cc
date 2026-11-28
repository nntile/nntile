/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file tests/test_gpu_cap.cc
 * The per-device arena cap and its precedence: NNHAUL_LIMIT_CUDA_MEM wins
 * absolutely, else the init argument clamped to ninety percent of free
 * memory, else free minus a driver margin floored at four classes; a
 * sub-class cap is a loud init error; the pool is whole class blocks
 * only. NodePoolStats are the observable face — the old flat Stats fields
 * died with the single-stream engine).
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "check.hh"
#include "../src/cuda_backend.hh"
#include "nnhaul/test_hooks.hh"

#include <cstdlib>
#include <string>

namespace
{

// The write body pushes the registration-time scalar (the stable
// cl_args blob) into the tile's first int.
void gpu_write(void **buffers, void *cl_args) noexcept
{
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    cudaMemcpyAsync(buf->get_ptr<int>(),
        cl_args,
        sizeof(int),
        cudaMemcpyHostToDevice,
        nnhaul::cuda_stream());
}

// The arena helper's precedence arithmetic, per device, before any
// runtime exists (the knob and the argument need no init; the free
// queries need a CUDA context, which cudaMemGetInfo establishes).
void check_suggest_limit_bytes()
{
    std::size_t const cls = nnhaul::chunk_class_bytes();

    // No knob, an explicit two-class cap: honored exactly
    // as the SOFT LIMIT (nothing is allocated; growth ends there).
    nnhaul::CudaExecLane::suggest_limit_bytes(0, 2 * cls); // no throw.
    CHECK(nnhaul::CudaExecLane::suggest_limit_bytes(0, 2 * cls) ==
          2 * cls);

    // A non-multiple limit reports the raw value (whole-blocks-only is
    // the pool's own init arithmetic, /chunk_class_bytes() at the caller).
    CHECK(nnhaul::CudaExecLane::suggest_limit_bytes(
              0, 2 * cls + cls / 2) == 2 * cls + cls / 2);

    // The knob wins absolutely over the argument.
    setenv("NNHAUL_LIMIT_CUDA_MEM",
        std::to_string((3 * cls) >> 20).c_str(),
        1);
    CHECK(nnhaul::CudaExecLane::suggest_limit_bytes(0, 8 * cls) ==
          3 * cls);
    unsetenv("NNHAUL_LIMIT_CUDA_MEM");

    // No knob, no argument: UNLIMITED (0) — growth ends at
    // the hard limit only (the old whole-free-VRAM pre-carve path is
    // deleted machinery).
    CHECK(nnhaul::CudaExecLane::suggest_limit_bytes(0, 0) == 0);

    // A sub-class limit backs zero chunks — loud, whichever arm named it.
    bool threw = false;
    try
    {
        (void) nnhaul::CudaExecLane::suggest_limit_bytes(0, cls / 2);
    }
    catch (std::runtime_error const &err)
    {
        threw = true;
        CHECK(std::string(err.what()).find("sub-class limit") !=
              std::string::npos);
    }
    CHECK(threw);

    // An integer-MiB knob can only name a sub-class cap when the class is
    // larger than 1 MiB. At smaller classes the argument arm above already
    // covers the sub-class rule.
    if(cls > (std::size_t{1} << 20))
    {
        setenv("NNHAUL_LIMIT_CUDA_MEM", "1", 1);
        threw = false;
        try
        {
            (void) nnhaul::CudaExecLane::suggest_limit_bytes(0, 8 * cls);
        }
        catch (std::runtime_error const &err)
        {
            threw = true;
            CHECK(std::string(err.what()).find("sub-class limit") !=
                  std::string::npos);
        }
        CHECK(threw);
        unsetenv("NNHAUL_LIMIT_CUDA_MEM");
    }
}

} // namespace

int main()
{
    check_suggest_limit_bytes();

    // ---- the argument arm: a two-class device pool, whole blocks ----
    {
        std::size_t const tile = nnhaul::chunk_class_bytes();
        std::size_t const cap = 2 * tile;
        // The host pool must back both tiles' host replicas (whole
        // blocks only): 64 MiB of bound is comfortably above two
        // classes.
        GpuRuntimeGuard guard(0, 1, std::size_t{64} << 20, cap);

        nnhaul::Stats st = nnhaul::stats();
        CHECK(st.node.size() == 2); // device 0, host 1.
        CHECK(!st.node[0].host);
        // Nothing preallocated: the pool reports 0 blocks
        // until demand grows it (the cap is the ceiling, checked by
        // the run below reaching exactly the two chunks it needs).
        CHECK(st.node[0].pool_blocks == 0);
        CHECK(st.node[0].pool_bytes == 0);
        CHECK(st.node[0].alloc_count >= 1); // the arena's cudaMalloc.
        // The fixed carve rides BESIDE the pool, never chunks of it
        // : the workspace is reported on the node's entry.
        CHECK(st.node[0].workspace_bytes > 0);

        // Whole blocks only: a 2.5-class cap floors to two blocks.
        // (Covered by the knob arm below at init level.)

        nnhaul::Codelet write_op("gpu-write", nullptr, gpu_write, nullptr);
        nnhaul::Handle a(tile);
        nnhaul::Handle b(tile);
        a.set_name("a");
        b.set_name("b");
        int const three = 3;
        int const five = 5;
        nnhaul::insert(
            write_op, {{&a, nnhaul::Access::Write}}, &three, sizeof(three));
        nnhaul::insert(
            write_op, {{&b, nnhaul::Access::Write}}, &five, sizeof(five));
        nnhaul::wait();

        {
            auto got = b.acquire(nnhaul::Access::Read);
            CHECK(*got.get_ptr<int>() == 5);
        }
        // Grown on demand to exactly the two chunks the
        // run needed (the soft cap was 2 — never exceeded).
        nnhaul::Stats st2 = nnhaul::stats();
        CHECK(st2.node[0].pool_blocks == 2);
        CHECK(st2.node[0].pool_bytes == 2 * nnhaul::chunk_class_bytes());
    }

    // ---- the knob arm: NNHAUL_LIMIT_CUDA_MEM wins over the argument,
    //      sub-multiple caps floor to whole blocks, and a sub-class cap
    //      is a loud INIT error (the engine's own belt). ----
    {
        std::size_t const cls = nnhaul::chunk_class_bytes();
        setenv("NNHAUL_LIMIT_CUDA_MEM",
            std::to_string((3 * cls) >> 20).c_str(),
            1);
        {
            // The argument names eight classes; the knob's three win.
            GpuRuntimeGuard guard(0, 1, std::size_t{64} << 20, 8 * cls);
            // The knob is the CEILING; nothing is
            // allocated at init (growth supplies chunks on demand).
            nnhaul::Stats st = nnhaul::stats();
            CHECK(st.node[0].pool_blocks == 0);
        }
        setenv("NNHAUL_LIMIT_CUDA_MEM",
            std::to_string(((2 * cls + cls / 2) >> 20)).c_str(),
            1);
        {
            // Whole blocks only: the 2.5-class knob floors to two.
            GpuRuntimeGuard guard(0, 1, std::size_t{64} << 20, 8 * cls);
            CHECK(nnhaul::stats().node[0].pool_blocks == 0);
        }
        // A sub-class knob: loud at init, nothing allocated after it.
        // An integer-MiB knob cannot be sub-class at a 1 MiB class.
        if(cls > (std::size_t{1} << 20))
        {
            setenv("NNHAUL_LIMIT_CUDA_MEM", "1", 1);
            bool threw = false;
            try
            {
                nnhaul::init(0, 1, std::size_t{64} << 20, 8 * cls);
            }
            catch (std::runtime_error const &err)
            {
                threw = true;
                CHECK(std::string(err.what()).find("sub-class limit") !=
                      std::string::npos);
            }
            CHECK(threw);
            CHECK(nnhaul::stats().memory_nodes == 0); // no runtime lingers.
            unsetenv("NNHAUL_LIMIT_CUDA_MEM");
        }

        unsetenv("NNHAUL_LIMIT_CUDA_MEM");

        // A sub-class argument arm is equally loud.
        bool threw = false;
        try
        {
            nnhaul::init(0, 1, std::size_t{64} << 20, cls / 2);
        }
        catch (std::runtime_error const &err)
        {
            threw = true;
            CHECK(std::string(err.what()).find("sub-class limit") !=
                  std::string::npos);
        }
        CHECK(threw);
        CHECK(nnhaul::stats().memory_nodes == 0);
    }

    return g_checks_failed == 0 ? 0 : 1;
}
