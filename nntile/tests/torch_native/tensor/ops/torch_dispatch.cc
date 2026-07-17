/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file nntile/tests/torch_native/tensor/ops/torch_dispatch.cc
 * TensorGraph torch_dispatch structure + untiled execute vs aten.
 */

#include "aten_ref.hh"
#include "context_fixture.hh"

#include <nntile/starpu/torch_dispatch.hh>
#include <nntile/tensor.hh>
#include <nntile/tensor/ops/torch_dispatch.hh>
#include <nntile/tile.hh>

#include <ATen/ops/mm.h>
#include <ATen/ops/relu.h>

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace nntile;
namespace gt = nntile::tensor;
namespace tn = nntile::test::torch_native;

TEST_CASE(
    "TensorGraph torch_unary structure",
    "[torch_native][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef in = graph.data({3, 4}, DataType::FP32);
    nntile::TensorRef out = nntile::TensorRef::adopt(
        gt::torch_unary(starpu::TorchKind::Relu, in, {3, 4}));

    REQUIRE(graph.num_ops() == 1);
    REQUIRE(graph.ops()[0]->op_name() == "TORCH_UNARY");
    REQUIRE(graph.ops()[0]->outputs()[0] == out);
}

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "TensorGraph torch_unary Relu untiled matches aten",
    "[torch_native][tensor]")
{
    const std::vector<Index> shape = {2, 4};
    std::vector<float> in = {-1.f, 0.f, 1.f, 2.f, -3.f, 4.f, -0.5f, 0.25f};
    std::vector<float> ref(8, 0.f);

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor tin = tn::blob_cpu(in.data(), shape);
            at::Tensor tr = tn::blob_cpu(ref.data(), shape);
            at::relu_out(tr, tin);
        });

    TensorGraph graph("torch_relu");
    nntile::TensorRef x = graph.data(shape, DataType::FP32);
    nntile::TensorRef y = nntile::TensorRef::adopt(
        gt::torch_unary(starpu::TorchKind::Relu, x, shape));

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(x, in);
    runtime.execute();
    runtime.wait();
    tn::require_close(runtime.get_output<float>(y), ref);
}

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "TensorGraph torch_binary Mm untiled matches aten",
    "[torch_native][tensor]")
{
    std::vector<float> a = {1.f, 2.f, 3.f, 4.f}; // 2x2
    std::vector<float> b = {5.f, 6.f, 7.f, 8.f}; // 2x2
    std::vector<float> ref(4, 0.f);

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor ta = tn::blob_cpu(a.data(), {2, 2});
            at::Tensor tb = tn::blob_cpu(b.data(), {2, 2});
            at::Tensor tr = tn::blob_cpu(ref.data(), {2, 2});
            at::mm_out(tr, ta, tb);
        });

    TensorGraph graph("torch_mm");
    nntile::TensorRef xa = graph.data({2, 2}, DataType::FP32);
    nntile::TensorRef xb = graph.data({2, 2}, DataType::FP32);
    nntile::TensorRef out = nntile::TensorRef::adopt(
        gt::torch_binary(
            starpu::TorchKind::Mm,
            xa,
            xb,
            {2, 2}));

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(xa, a);
    runtime.bind_data(xb, b);
    runtime.execute();
    runtime.wait();
    tn::require_close(runtime.get_output<float>(out), ref);
}
