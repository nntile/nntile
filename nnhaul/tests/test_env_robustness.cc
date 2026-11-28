/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file tests/test_env_robustness.cc
 * Environment-variable parsing robustness: overflow, trailing garbage, and
 * sign handling for the nnhaul knobs.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

// Environment-variable parsing robustness: overflow, trailing
// garbage, and sign handling for the three nnhaul knobs. The parsers
// are header-inline and side-effect free, so this is a plain unit
// test. NNHAUL_CUBLAS_WORKSPACE_MB is cached at
// first use per process, so each gets exactly one scenario here (the
// overflow that used to wrap); the raw helper and the per-init
// NNHAUL_LIMIT_CUDA_MEM cover the rest.
#include "../src/engine.hh"

#include "check.hh"

#include <cstdlib>
#include <string>

namespace
{

void set_env(char const *name, char const *value)
{
    setenv(name, value, 1);
}

void unset_env(char const *name)
{
    unsetenv(name);
}

} // namespace

int main()
{
    // The raw parser: trailing garbage, empty, and out-of-range
    // (strtol clamps to LONG_MAX and sets ERANGE) are all rejected.
    long parsed = -1;
    CHECK(nnhaul::env_parse_long("1500", parsed) && parsed == 1500);
    CHECK(!nnhaul::env_parse_long("100abc", parsed));
    CHECK(!nnhaul::env_parse_long("", parsed));
    CHECK(!nnhaul::env_parse_long("99999999999999999999", parsed));
    CHECK(nnhaul::env_parse_long("-5", parsed) && parsed == -5);

    // NNHAUL_LIMIT_CUDA_MEM (read per call): sane value, then the
    // rejections. A wrapped cap of 0 used to silently select the
    // 90%-of-free default.
    set_env("NNHAUL_LIMIT_CUDA_MEM", "5120");
    CHECK(nnhaul::limit_cuda_mem_bytes() == (std::size_t{5120} << 20));

    bool threw = false;
    try
    {
        set_env("NNHAUL_LIMIT_CUDA_MEM", "9223372036854775807");
        (void) nnhaul::limit_cuda_mem_bytes();
    }
    catch (nnhaul::Error const &)
    {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try
    {
        set_env("NNHAUL_LIMIT_CUDA_MEM", "-5");
        (void) nnhaul::limit_cuda_mem_bytes();
    }
    catch (nnhaul::Error const &)
    {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try
    {
        set_env("NNHAUL_LIMIT_CUDA_MEM", "10MiB");
        (void) nnhaul::limit_cuda_mem_bytes();
    }
    catch (nnhaul::Error const &)
    {
        threw = true;
    }
    CHECK(threw);
    unset_env("NNHAUL_LIMIT_CUDA_MEM");

    // NNHAUL_CUBLAS_WORKSPACE_MB (cached): a value that parses but
    // overflows the MiB-to-bytes shift clamps to 4 GiB instead of
    // wrapping to an arbitrary byte count.
    set_env("NNHAUL_CUBLAS_WORKSPACE_MB", "9999999999");
    CHECK(nnhaul::cublas_workspace_bytes() == (std::size_t{4096} << 20));

    return g_checks_failed == 0 ? 0 : 1;
}
