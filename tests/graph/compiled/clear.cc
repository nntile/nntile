#include "compiled_test_utils.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph ClearBool vs Tensor",
    "[graph][verification]")
{
    LogicalGraph g("test");
    auto& x = g.tensor({4}, "x", DataType::BOOL);
    clear(x);

    auto compiled = CompiledGraph::compile(g);

    std::vector<char> x_data = {static_cast<char>(true), static_cast<char>(false), static_cast<char>(true), static_cast<char>(false)};
    compiled.bind_data("x", x_data);

    compiled.execute();
    compiled.wait();

    auto out = compiled.get_output<char>("x");
    REQUIRE(out.size() == 4);
    for(const auto& v : out)
    {
        REQUIRE(static_cast<bool>(v) == false);
    }
}
