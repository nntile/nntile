/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file tests/check.hh
 * CHECK macro and RAII runtime guards shared by the runtime tests.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include "nnhaul/nnhaul.hh"

#include <iostream>

inline int g_checks_failed = 0;

inline void check_impl(bool ok, char const *expr, char const *file, int line)
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

class RuntimeGuard
{
  public:
    RuntimeGuard(int ncpu, std::size_t cap) { nnhaul::init(ncpu, 0, cap, 0); }

    ~RuntimeGuard() { nnhaul::shutdown(); }

    RuntimeGuard(RuntimeGuard const &) = delete;
    RuntimeGuard &operator=(RuntimeGuard const &) = delete;
};

class GpuRuntimeGuard
{
  public:
    GpuRuntimeGuard(
        int ncpu, int ncuda, std::size_t cpu_cap, std::size_t gpu_cap)
    {
        nnhaul::init(ncpu, ncuda, cpu_cap, gpu_cap);
    }

    ~GpuRuntimeGuard() { nnhaul::shutdown(); }

    GpuRuntimeGuard(GpuRuntimeGuard const &) = delete;
    GpuRuntimeGuard &operator=(GpuRuntimeGuard const &) = delete;
};
