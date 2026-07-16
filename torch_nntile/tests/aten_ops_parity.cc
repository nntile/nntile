/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/tests/aten_ops_parity.cc
 * ATen ops on PrivateUse1 vs CPU with forward + backward (libtorch_nntile).
 */

#include "parity_helpers.hh"

#include <torch_nntile/runtime.hh>

#include <catch2/catch_test_macros.hpp>

namespace
{

struct ContextGuard
{
    ContextGuard()
    {
        if (!torch_nntile::is_context_initialized())
        {
            torch_nntile::init_context(
                1, 0, 0, "/tmp/nntile_ooc", 16ull * 1024ull * 1024ull,
                0, 0, false);
            torch_nntile::restrict_cpu();
        }
    }

    ~ContextGuard()
    {
        if (torch_nntile::is_context_initialized())
        {
            torch_nntile::wait_for_all();
            torch_nntile::reset_graph_session();
        }
    }
};

at::Tensor seeded(at::IntArrayRef shape)
{
    torch::manual_seed(0);
    return torch::randn(shape, torch::dtype(torch::kFloat32))
        .set_requires_grad(true);
}

} // namespace

TEST_CASE("aten add fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto a = seeded({4, 6});
    auto b = seeded({4, 6});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return xs[0] + xs[1];
        },
        {a, b});
}

TEST_CASE("aten mul fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto a = seeded({4, 6});
    auto b = seeded({4, 6});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return xs[0] * xs[1];
        },
        {a, b});
}

TEST_CASE("aten relu fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({4, 8});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::relu(xs[0]);
        },
        {x});
}

TEST_CASE("aten silu fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({4, 8});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::silu(xs[0]);
        },
        {x});
}

TEST_CASE("aten gelu fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({4, 8});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::gelu(xs[0]);
        },
        {x});
}

TEST_CASE("aten softmax fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({3, 8});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::softmax(xs[0], /*dim=*/-1);
        },
        {x});
}

TEST_CASE("aten mm fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto a = seeded({5, 7});
    auto b = seeded({7, 4});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::mm(xs[0], xs[1]);
        },
        {a, b});
}

TEST_CASE("aten bmm fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto a = seeded({2, 5, 7});
    auto b = seeded({2, 7, 4});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::bmm(xs[0], xs[1]);
        },
        {a, b});
}

TEST_CASE("aten linear fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({3, 8});
    auto w = seeded({5, 8});
    auto bias = seeded({5});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::linear(xs[0], xs[1], xs[2]);
        },
        {x, w, bias});
}

TEST_CASE("aten cat fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto a = seeded({2, 4});
    auto b = seeded({3, 4});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::cat({xs[0], xs[1]}, /*dim=*/0);
        },
        {a, b});
}

TEST_CASE("aten transpose fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({3, 5});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return xs[0].transpose(0, 1).contiguous();
        },
        {x});
}
