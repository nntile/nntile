/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/tests/aten_ops_parity.cc
 * ATen ops on PrivateUse1 vs CPU (libtorch_nntile; torch-native StarPU path).
 */

#include "parity_helpers.hh"

#include <ATen/ops/rms_norm.h>
#include <torch_nntile/runtime.hh>

#include <catch2/catch_test_macros.hpp>

#include <optional>
#include <tuple>

namespace
{

struct ContextGuard
{
    ContextGuard()
    {
        if (!torch_nntile::is_context_initialized())
        {
            torch_nntile::init_context(
                1,
                0,
                0,
                "/tmp/nntile_ooc",
                16ull * 1024ull * 1024ull,
                0,
                0,
                false);
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

TEST_CASE("aten add alpha fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto a = seeded({3, 5});
    auto b = seeded({3, 5});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::add(xs[0], xs[1], /*alpha=*/1.5);
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

TEST_CASE("aten gelu tanh fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({4, 8});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::gelu(xs[0], "tanh");
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

TEST_CASE("aten addmm fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto bias = seeded({4});
    auto mat1 = seeded({5, 7});
    auto mat2 = seeded({7, 4});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::addmm(xs[0], xs[1], xs[2]);
        },
        {bias, mat1, mat2});
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

TEST_CASE("aten linear no bias fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({3, 8});
    auto w = seeded({5, 8});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return torch::linear(xs[0], xs[1]);
        },
        {x, w});
}

TEST_CASE("aten avg_pool2d fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({2, 3, 8, 8});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return at::avg_pool2d(
                xs[0],
                {2, 2},
                {2, 2},
                {0, 0},
                false,
                true,
                std::nullopt);
        },
        {x});
}

TEST_CASE("aten adaptive_avg_pool2d fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({2, 3, 7, 5});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return at::_adaptive_avg_pool2d(xs[0], {3, 2});
        },
        {x});
}

TEST_CASE("aten upsample_nearest2d fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({2, 3, 5, 7});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return at::upsample_nearest2d(
                xs[0],
                c10::IntArrayRef({10, 14}),
                std::nullopt,
                std::nullopt);
        },
        {x});
}

TEST_CASE(
    "aten upsample_nearest2d scale_factor fwd+bwd matches CPU",
    "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({2, 3, 4, 4});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            // Matches F.interpolate(..., scale_factor=2): both size and
            // scales are passed to the device schema.
            return at::upsample_nearest2d(
                xs[0],
                c10::IntArrayRef({8, 8}),
                /*scales_h=*/2.0,
                /*scales_w=*/2.0);
        },
        {x});
}

TEST_CASE("aten upsample_bilinear2d fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({2, 3, 5, 7});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return at::upsample_bilinear2d(
                xs[0],
                c10::IntArrayRef({10, 14}),
                /*align_corners=*/false,
                std::nullopt,
                std::nullopt);
        },
        {x});
}

TEST_CASE("aten max_pool2d_with_indices fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({2, 3, 8, 8});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return std::get<0>(at::max_pool2d_with_indices(
                xs[0],
                {2, 2},
                {2, 2},
                {0, 0},
                {1, 1},
                false));
        },
        {x});
}

TEST_CASE("aten convolution fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({2, 3, 8, 8});
    auto w = seeded({4, 3, 3, 3});
    auto b = seeded({4});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return at::convolution(
                xs[0],
                xs[1],
                xs[2],
                {1, 1},
                {1, 1},
                {1, 1},
                false,
                {0, 0},
                1);
        },
        {x, w, b},
        1e-4,
        1e-4,
        2e-3,
        2e-3);
}

TEST_CASE("aten native_batch_norm fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({2, 3, 4, 4});
    auto w = seeded({3});
    auto b = seeded({3});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return std::get<0>(at::native_batch_norm(
                xs[0],
                xs[1],
                xs[2],
                std::nullopt,
                std::nullopt,
                true,
                0.1,
                1e-5));
        },
        {x, w, b},
        1e-4,
        1e-4,
        2e-3,
        2e-3);
}

TEST_CASE("aten rms_norm weight fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({2, 3, 8});
    auto w = seeded({8});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return at::rms_norm(
                xs[0],
                {8},
                std::optional<at::Tensor>(xs[1]),
                1e-5);
        },
        {x, w});
}

TEST_CASE("aten rms_norm no weight fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({2, 3, 8});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return at::rms_norm(
                xs[0],
                {8},
                std::nullopt,
                1e-5);
        },
        {x});
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

TEST_CASE("aten hypot fwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    torch::manual_seed(0);
    at::Tensor a = torch::randn({4, 6}, torch::kFloat32);
    at::Tensor b = torch::randn({4, 6}, torch::kFloat32);
    at::Tensor y_ref = torch::hypot(a, b);
    c10::Device const dev = torch_nntile::test::nntile_device();
    at::Tensor y = torch::hypot(a.to(dev), b.to(dev));
    torch_nntile::test::assert_close(y, y_ref);
}

TEST_CASE("aten sum dim fwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    torch::manual_seed(0);
    at::Tensor x = torch::randn({3, 5, 7}, torch::kFloat32);
    at::Tensor y_ref = torch::sum(x, /*dim=*/-1);
    c10::Device const dev = torch_nntile::test::nntile_device();
    at::Tensor y = torch::sum(x.to(dev), /*dim=*/-1);
    torch_nntile::test::assert_close(y, y_ref);
}

TEST_CASE("aten repeat fwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    torch::manual_seed(0);
    at::Tensor x = torch::randn({2, 3}, torch::kFloat32);
    at::Tensor y_ref = x.repeat({2, 1});
    c10::Device const dev = torch_nntile::test::nntile_device();
    at::Tensor y = x.to(dev).repeat({2, 1});
    torch_nntile::test::assert_close(y, y_ref);
}

TEST_CASE("aten narrow fwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    torch::manual_seed(0);
    at::Tensor x = torch::randn({2, 8}, torch::kFloat32);
    at::Tensor y_ref = x.narrow(/*dim=*/1, /*start=*/2, /*length=*/4);
    c10::Device const dev = torch_nntile::test::nntile_device();
    at::Tensor y = x.to(dev).narrow(1, 2, 4);
    torch_nntile::test::assert_close(y, y_ref);
}

TEST_CASE("aten narrow fwd+bwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({2, 8});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            return xs[0].narrow(/*dim=*/1, /*start=*/2, /*length=*/4);
        },
        {x});
}

TEST_CASE(
    "aten last-dim slice+cat fwd+bwd matches CPU",
    "[aten][parity]")
{
    ContextGuard guard;
    auto x = seeded({2, 4, 8});
    torch_nntile::test::assert_op_forward_backward(
        [](std::vector<at::Tensor> const &xs)
        {
            at::Tensor rot = xs[0].slice(/*dim=*/-1, 0, 4);
            at::Tensor pass = xs[0].slice(/*dim=*/-1, 4, 8);
            return torch::cat({rot, pass}, /*dim=*/-1);
        },
        {x});
}

TEST_CASE("aten vector_norm fwd matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    torch::manual_seed(0);
    at::Tensor x = torch::randn({3, 5}, torch::kFloat32);
    at::Tensor y_ref = torch::linalg_vector_norm(x, /*ord=*/2, /*dim=*/-1);
    c10::Device const dev = torch_nntile::test::nntile_device();
    at::Tensor y = torch::linalg_vector_norm(x.to(dev), 2, -1);
    torch_nntile::test::assert_close(y, y_ref);
}

TEST_CASE("aten arange as first graph op matches CPU", "[aten][parity]")
{
    ContextGuard guard;
    c10::Device const dev = torch_nntile::test::nntile_device();
    at::Tensor y = torch::arange(
        /*end=*/8,
        torch::TensorOptions().dtype(torch::kLong).device(dev));
    at::Tensor y_ref = torch::arange(
        /*end=*/8,
        torch::TensorOptions().dtype(torch::kLong));
    torch_nntile::test::assert_close(y, y_ref);
    at::Tensor y2 = torch::arange(
        /*start=*/0,
        /*end=*/8,
        torch::TensorOptions().dtype(torch::kLong).device(dev));
    torch_nntile::test::assert_close(y2, y_ref);
    at::Tensor yf = torch::arange(
        /*end=*/4,
        torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    at::Tensor yf_ref = torch::arange(
        /*end=*/4,
        torch::TensorOptions().dtype(torch::kFloat32));
    torch_nntile::test::assert_close(yf, yf_ref);
}
