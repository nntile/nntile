/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file tests/test_sgemm.cc
 * A cuBLAS SGEMM codelet on the worker stream, checked against a CPU
 * reference.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "check.hh"

#include <vector>

namespace
{

constexpr int kN = 32;

void gemm(void **buffers, void *cl_args) noexcept
{
    auto *mismatch = *static_cast<int **>(cl_args);
    auto *a = static_cast<nnhaul::Buffer *>(buffers[0]);
    auto *b = static_cast<nnhaul::Buffer *>(buffers[1]);
    auto *c = static_cast<nnhaul::Buffer *>(buffers[2]);
    float one = 1.f;
    float zero = 0.f;
    cublasSetStream(nnhaul::cublas_handle(), nnhaul::cuda_stream());
    cublasSgemm(nnhaul::cublas_handle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        kN,
        kN,
        kN,
        &one,
        a->get_ptr<float>(),
        kN,
        b->get_ptr<float>(),
        kN,
        &zero,
        c->get_ptr<float>(),
        kN);
    cudaStream_t bound = nullptr;
    cublasGetStream(nnhaul::cublas_handle(), &bound);
    if (bound != nnhaul::cuda_stream() || nnhaul::device_index() != 0)
    {
        *mismatch = 1;
    }
}

} // namespace

int main()
{
    constexpr std::size_t bytes = kN * kN * sizeof(float);
    GpuRuntimeGuard guard(0, 1, 1u << 26, 1u << 26);
    nnhaul::Handle a(bytes);
    nnhaul::Handle b(bytes);
    nnhaul::Handle c(bytes);
    a.set_name("A");
    b.set_name("B");
    c.set_name("C");
    {
        auto am = a.acquire(nnhaul::Access::Write);
        auto bm = b.acquire(nnhaul::Access::Write);
        float *av = am.get_ptr<float>();
        float *bv = bm.get_ptr<float>();
        for (int col = 0; col < kN; ++col)
        {
            for (int row = 0; row < kN; ++row)
            {
                av[col * kN + row] = row == col ? 1.f : 0.f;
                bv[col * kN + row] = static_cast<float>(col * kN + row);
            }
        }
    }
    int mismatch = 0;
    int *mismatch_ptr = &mismatch;
    nnhaul::Codelet op("sgemm", nullptr, gemm, nullptr);
    nnhaul::insert(op,
        {{&a, nnhaul::Access::Read},
            {&b, nnhaul::Access::Read},
            {&c, nnhaul::Access::Write}},
        &mismatch_ptr,
        sizeof(mismatch_ptr));
    nnhaul::wait();
    std::vector<float> got(static_cast<std::size_t>(kN * kN));
    {
        auto cm = c.acquire(nnhaul::Access::Read);
        float *cv = cm.get_ptr<float>();
        for (int i = 0; i < kN * kN; ++i)
        {
            got[static_cast<std::size_t>(i)] = cv[i];
        }
    }
    CHECK(mismatch == 0);
    for (int i = 0; i < kN * kN; ++i)
    {
        CHECK(got[static_cast<std::size_t>(i)] == static_cast<float>(i));
    }
    return g_checks_failed == 0 ? 0 : 1;
}
