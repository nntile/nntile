/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file nntile/tests/torch_native/tile/ops/torch_dispatch.cc
 * TileGraph torch_dispatch vs CPU aten.
 */

#include "aten_ref.hh"
#include "context_fixture.hh"

#include <nntile/starpu/torch_dispatch.hh>
#include <nntile/tile.hh>
#include <nntile/tile/ops/torch_dispatch.hh>

#include <ATen/ops/add.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/relu.h>

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace nntile;
namespace tg = nntile::tile;
namespace tn = nntile::test::torch_native;

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "TileGraph torch_unary Relu matches aten",
    "[torch_native][tile]")
{
    const std::vector<Index> shape = {2, 3};
    std::vector<float> in = {-2.f, -1.f, 0.f, 1.f, 2.f, 3.f};
    std::vector<float> ref(6, 0.f);

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor tin = tn::blob_cpu(in.data(), shape);
            at::Tensor tr = tn::blob_cpu(ref.data(), shape);
            at::relu_out(tr, tin);
        });

    TileGraph graph("torch_relu_tile");
    auto *x = graph.data(shape, "x", DataType::FP32);
    auto *y = graph.data(shape, "y", DataType::FP32);
    tg::torch_unary(starpu::TorchKind::Relu, x, y);

    Runtime runtime(graph);
    runtime.compile();
    runtime.bind_data(x, in);
    runtime.execute();
    runtime.wait();
    tn::require_close(runtime.get_output<float>(y), ref);
}

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "TileGraph torch_binary Mul matches aten",
    "[torch_native][tile]")
{
    const std::vector<Index> shape = {2, 2};
    std::vector<float> a = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> b = {4.f, 3.f, 2.f, 1.f};
    std::vector<float> ref(4, 0.f);

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor ta = tn::blob_cpu(a.data(), shape);
            at::Tensor tb = tn::blob_cpu(b.data(), shape);
            at::Tensor tr = tn::blob_cpu(ref.data(), shape);
            at::mul_out(tr, ta, tb);
        });

    TileGraph graph("torch_mul_tile");
    auto *xa = graph.data(shape, "a", DataType::FP32);
    auto *xb = graph.data(shape, "b", DataType::FP32);
    auto *out = graph.data(shape, "out", DataType::FP32);
    tg::torch_binary(starpu::TorchKind::Mul, xa, xb, out);

    Runtime runtime(graph);
    runtime.compile();
    runtime.bind_data(xa, a);
    runtime.bind_data(xb, b);
    runtime.execute();
    runtime.wait();
    tn::require_close(runtime.get_output<float>(out), ref);
}

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "TileGraph torch_binary Add matches aten",
    "[torch_native][tile]")
{
    const std::vector<Index> shape = {2, 2};
    const Scalar alpha = 1.5f;
    std::vector<float> a = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> b = {4.f, 3.f, 2.f, 1.f};
    std::vector<float> ref(4, 0.f);

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor ta = tn::blob_cpu(a.data(), shape);
            at::Tensor tb = tn::blob_cpu(b.data(), shape);
            at::Tensor tr = tn::blob_cpu(ref.data(), shape);
            at::add_out(tr, ta, tb, static_cast<double>(alpha));
        });

    TileGraph graph("torch_add_tile");
    auto *xa = graph.data(shape, "a", DataType::FP32);
    auto *xb = graph.data(shape, "b", DataType::FP32);
    auto *out = graph.data(shape, "out", DataType::FP32);
    starpu::TorchDispatchArgs extra;
    extra.scalars[0] = alpha;
    tg::torch_binary(starpu::TorchKind::Add, xa, xb, out, extra);

    Runtime runtime(graph);
    runtime.compile();
    runtime.bind_data(xa, a);
    runtime.bind_data(xb, b);
    runtime.execute();
    runtime.wait();
    tn::require_close(runtime.get_output<float>(out), ref);
}
