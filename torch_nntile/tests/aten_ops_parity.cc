/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/tests/aten_ops_parity.cc
 * ATen forward parity on PrivateUse1 vs CPU (libtorch_nntile).
 *
 * Backward is covered by the torch_nntile Python extension suite: LibTorch
 * C++ autograd currently queries Accelerator streams for PrivateUse1 and
 * fails without the Python device-module registration.
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
    return torch::randn(shape, torch::dtype(torch::kFloat32));
}

void assert_forward(
    std::function<at::Tensor(std::vector<at::Tensor> const &)> op,
    std::vector<at::Tensor> inputs_cpu)
{
    at::Tensor y_ref = op(inputs_cpu);
    c10::Device const dev = torch_nntile::test::nntile_device();
    std::vector<at::Tensor> nnt_inputs;
    nnt_inputs.reserve(inputs_cpu.size());
    for (at::Tensor const &t : inputs_cpu)
    {
        nnt_inputs.push_back(t.contiguous().to(dev));
    }
    at::Tensor y_nnt = op(nnt_inputs);
    torch_nntile::test::assert_close(y_nnt, y_ref, 1e-4, 1e-4, "forward");
}

} // namespace

TEST_CASE("aten add forward matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    assert_forward(
        [](std::vector<at::Tensor> const &xs)
        {
            return xs[0] + xs[1];
        },
        {seeded({4, 6}), seeded({4, 6})});
}

TEST_CASE("aten mul forward matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    assert_forward(
        [](std::vector<at::Tensor> const &xs)
        {
            return xs[0] * xs[1];
        },
        {seeded({4, 6}), seeded({4, 6})});
}

TEST_CASE("aten relu forward matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    assert_forward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::relu(xs[0]);
        },
        {seeded({4, 8})});
}

TEST_CASE("aten silu forward matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    assert_forward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::silu(xs[0]);
        },
        {seeded({4, 8})});
}

TEST_CASE("aten gelu forward matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    assert_forward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::gelu(xs[0]);
        },
        {seeded({4, 8})});
}

TEST_CASE("aten softmax forward matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    assert_forward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::softmax(xs[0], /*dim=*/-1);
        },
        {seeded({3, 8})});
}

TEST_CASE("aten mm forward matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    assert_forward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::mm(xs[0], xs[1]);
        },
        {seeded({5, 7}), seeded({7, 4})});
}

TEST_CASE("aten bmm forward matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    assert_forward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::bmm(xs[0], xs[1]);
        },
        {seeded({2, 5, 7}), seeded({2, 7, 4})});
}

TEST_CASE("aten linear forward matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    assert_forward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::linear(xs[0], xs[1], xs[2]);
        },
        {seeded({3, 8}), seeded({5, 8}), seeded({5})});
}

TEST_CASE("aten cat forward matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    assert_forward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::cat({xs[0], xs[1]}, /*dim=*/0);
        },
        {seeded({2, 4}), seeded({3, 4})});
}

TEST_CASE("aten transpose forward matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    assert_forward(
        [](std::vector<at::Tensor> const &xs)
        {
            return xs[0].transpose(0, 1).contiguous();
        },
        {seeded({3, 5})});
}
