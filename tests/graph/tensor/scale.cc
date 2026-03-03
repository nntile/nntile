/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/scale.cc
 * Test TensorGraph scale operation against nntile::tensor::scale.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include <numeric>

#include "nntile/context.hh"
#include "nntile/graph/tensor/scale.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/scale.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;

template<typename T>
void check_scale_vs_tensor_api(const std::vector<Index>& shape, Scalar alpha)
{
    using Y = typename T::repr_t;
    const Index nelems =
        std::accumulate(shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("scale_test");
    auto* src_node = graph.data(shape, "src", DataType::FP32);
    src_node->mark_input(true);

    auto* dst_node = scale(alpha, src_node, "dst");
    dst_node->mark_output(true);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    // Generate input data once
    std::vector<float> src_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + 1));
    }

    // --- TensorGraph path ---
    runtime.bind_data("src", src_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path (same input data) ---
    tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, 0);
    tensor::Tensor<T> src(traits, distr);
    tensor::Tensor<T> dst(traits, distr);

    {
        auto tile = src.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            loc[i] = static_cast<Y>(src_data[i]);
        }
        loc.release();
    }

    tensor::scale<T>(alpha, src, dst);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(nelems);
    {
        auto tile = dst.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i)
        {
            tensor_result[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    }

    // --- Compare ---
    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tensor_result[i]) < 1e-5f);
    }
}

TEST_CASE("TensorGraph scale matches tensor::scale", "[graph][tensor]")
{
    Context context(1, 0, 0, "/tmp/nntile_ooc", 16777216, 0);

    SECTION("alpha=2.5, shape {4,5}")
    {
        check_scale_vs_tensor_api<nntile::fp32_t>({4, 5}, 2.5);
    }
    SECTION("alpha=-1, shape {6}")
    {
        check_scale_vs_tensor_api<nntile::fp32_t>({6}, -1.0);
    }
}
