#include "compiled_test_utils.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph EmbeddingBackward Smoke",
    "[graph][compiled][smoke]")
{
    InputOverrides overrides;
    overrides.int64_inputs["index"] = {0, 1, 2, 3, 4, 5};

    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& embed = g.tensor({4, 2, 3}, "embed", DataType::FP32);
            auto& index = g.tensor({2, 3}, "index", DataType::INT64);
            auto& vocab = g.tensor({4, 10}, "vocab", DataType::FP32);
            embedding_backward(embed, index, vocab, 0);
        },
        {"embed", "index", "vocab"},
        overrides
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph SubtractIndexedOutputs Smoke",
    "[graph][compiled][smoke]")
{
    InputOverrides overrides;
    overrides.int64_inputs["labels"] = {0, 2};

    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& labels = g.tensor({2}, "labels", DataType::INT64);
            auto& x = g.tensor({3, 2}, "x", DataType::FP32);
            subtract_indexed_outputs(labels, x, 0.5f, -1);
        },
        {"labels", "x"},
        overrides
    );
}
