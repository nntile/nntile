/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/tests/smoke_add.cc
 * Minimal libtorch_nntile PrivateUse1 smoke (add + context).
 */

#include "parity_helpers.hh"

#include <torch_nntile/runtime.hh>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("libtorch_nntile smoke add on PrivateUse1", "[smoke]")
{
    torch_nntile::init_context(
        /*ncpu=*/1,
        /*ncuda=*/0,
        /*ooc_enabled=*/0,
        /*ooc_path=*/"/tmp/nntile_ooc",
        /*ooc_size=*/16ull * 1024ull * 1024ull,
        /*logger=*/0,
        /*verbose=*/0,
        /*cpu_fallback=*/false);
    torch_nntile::restrict_cpu();

    c10::Device const dev = torch_nntile::test::nntile_device();
    at::Tensor lhs = torch::tensor({1.f, 2.f, 3.f}).to(dev);
    at::Tensor rhs = torch::tensor({4.f, 5.f, 6.f}).to(dev);
    at::Tensor out = lhs + rhs;
    if (torch_nntile::has_pending_graph())
    {
        torch_nntile::compile_graph();
        torch_nntile::run_graph();
        torch_nntile::wait_for_all();
    }
    torch_nntile::test::assert_close(
        out,
        torch::tensor({5.f, 7.f, 9.f}),
        1e-5,
        1e-5,
        "smoke add");

    torch_nntile::shutdown_context();
}
