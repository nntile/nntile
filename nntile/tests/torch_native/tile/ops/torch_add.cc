/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file nntile/tests/torch_native/tile/ops/torch_add.cc
 * TileGraph torch_add vs aten::add_out.
 */

#include "aten_ref.hh"
#include "context_fixture.hh"

#include <nntile/tile.hh>
#include <nntile/tile/ops/torch_add.hh>

#include <ATen/ops/add.h>

#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <vector>

using namespace nntile;
namespace tg = nntile::tile;
namespace tn = nntile::test::torch_native;

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "TileGraph torch_add matches aten::add_out",
    "[torch_native][tile]")
{
    const std::vector<Index> shape = {3, 5};
    const Index nelems = std::accumulate(
        shape.begin(),
        shape.end(),
        Index{1},
        std::multiplies<>());
    const Scalar alpha = 1.25f;

    std::vector<float> x_data(nelems), y_data(nelems), ref(nelems, 0.f);
    for (Index i = 0; i < nelems; ++i)
    {
        x_data[static_cast<size_t>(i)] = static_cast<float>(i);
        y_data[static_cast<size_t>(i)] = static_cast<float>(-2 * i);
    }

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor tx = tn::blob_cpu(x_data.data(), shape);
            at::Tensor ty = tn::blob_cpu(y_data.data(), shape);
            at::Tensor tr = tn::blob_cpu(ref.data(), shape);
            at::add_out(tr, tx, ty, alpha);
        });

    TileGraph graph("torch_add_tile");
    auto *x = graph.data(shape, "x", DataType::FP32);
    auto *y = graph.data(shape, "y", DataType::FP32);
    auto *z = graph.data(shape, "z", DataType::FP32);
    tg::torch_add(x, y, z, alpha);

    Runtime runtime(graph);
    runtime.compile();
    runtime.bind_data(x, x_data);
    runtime.bind_data(y, y_data);
    runtime.execute();
    runtime.wait();

    const std::vector<float> got = runtime.get_output<float>(z);
    tn::require_close(got, ref);
}
