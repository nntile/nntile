#pragma once

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <functional>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "nntile/base_types.hh"
#include "nntile/context.hh"
#include "nntile/graph.hh"
#include "nntile/tensor/tensor.hh"

#include <starpu.h>

namespace nntile::graph::test
{

// Fixture to initialize NNTile context for graph tests
class GraphTestFixture
{
protected:
    nntile::Context context;
public:
    GraphTestFixture():
        context(
            1, 0, 0, "/tmp/nntile_ooc", 16777216, 0, "localhost", 5001, 0
        )
    {}
};

inline size_t tensor_nelems(const std::vector<Index>& shape)
{
    size_t size = 1;
    for(auto dim : shape)
    {
        size *= static_cast<size_t>(dim);
    }
    return size;
}

template<typename ValueT>
inline std::vector<ValueT> make_pattern(size_t size, ValueT scale)
{
    std::vector<ValueT> data(size);
    for(size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<ValueT>((i % 10) + 1) * scale;
    }
    return data;
}

template<typename WrapperT>
inline void write_tensor(nntile::tensor::Tensor<WrapperT>& tensor,
                         const std::vector<typename WrapperT::repr_t>& data)
{
    auto tile = tensor.get_tile(0);
    auto tile_local = tile.acquire(STARPU_W);
    for(Index i = 0; i < tensor.nelems; ++i)
    {
        tile_local[i] = data[static_cast<size_t>(i)];
    }
    tile_local.release();
}

inline void write_tensor(nntile::tensor::Tensor<bool_t>& tensor,
                                 const std::vector<char>& data)
{
    auto tile = tensor.get_tile(0);
    auto tile_local = tile.acquire(STARPU_W);
    for(Index i = 0; i < tensor.nelems; ++i) {
        tile_local[i] = nntile::bool_t(data[static_cast<size_t>(i)]);
    }
    tile_local.release();
}

template<typename WrapperT>
inline std::vector<typename WrapperT::repr_t> read_tensor(
    const nntile::tensor::Tensor<WrapperT>& tensor)
{
    auto tile = tensor.get_tile(0);
    auto tile_local = tile.acquire(STARPU_R);
    std::vector<typename WrapperT::repr_t> data(
        static_cast<size_t>(tensor.nelems));
    for(Index i = 0; i < tensor.nelems; ++i)
    {
        data[static_cast<size_t>(i)] =
            static_cast<typename WrapperT::repr_t>(tile_local[i]);
    }
    tile_local.release();
    return data;
}

struct InputOverrides
{
    std::map<std::string, std::vector<float>> float_inputs;
    std::map<std::string, std::vector<double>> double_inputs;
    std::map<std::string, std::vector<long long>> int64_inputs;
    std::map<std::string, std::vector<nntile::bool_t>> bool_inputs;
};

inline void bind_inputs(CompiledGraph& compiled,
                        const LogicalGraph& g,
                        const std::vector<std::string>& input_names,
                        const InputOverrides& overrides = {})
{
    for(const auto& name : input_names)
    {
        const auto& tensor = g.get_tensor(name);
        const auto size = tensor_nelems(tensor->shape());
        const auto dtype = tensor->dtype();

        if(dtype == DataType::BOOL)
        {
            auto it = overrides.bool_inputs.find(name);
            std::vector<char> data(size);
            if(it != overrides.bool_inputs.end())
            {
                for(size_t i = 0; i < size; ++i)
                {
                    data[i] = static_cast<char>(it->second[i].value ? 1 : 0);
                }
            }
            else
            {
                for(size_t i = 0; i < size; ++i)
                {
                    data[i] = static_cast<char>((i % 2) == 0);
                }
            }
            compiled.bind_data(name, data);
        }
        else if(dtype == DataType::INT64)
        {
            auto it = overrides.int64_inputs.find(name);
            std::vector<long long> data = (it != overrides.int64_inputs.end())
                ? it->second
                : std::vector<long long>(size);
            if(it == overrides.int64_inputs.end())
            {
                for(size_t i = 0; i < size; ++i)
                {
                    data[i] = static_cast<long long>(i);
                }
            }
            compiled.bind_data(name, data);
        }
        else if(dtype == DataType::FP64)
        {
            auto it = overrides.double_inputs.find(name);
            std::vector<double> data = (it != overrides.double_inputs.end())
                ? it->second
                : make_pattern<double>(size, 0.1);
            compiled.bind_data(name, data);
        }
        else
        {
            auto it = overrides.float_inputs.find(name);
            std::vector<float> data = (it != overrides.float_inputs.end())
                ? it->second
                : make_pattern<float>(size, 0.1f);
            compiled.bind_data(name, data);
        }
    }
}

inline void run_compiled_graph(
    const std::function<void(LogicalGraph&)>& build_graph,
    const std::vector<std::string>& input_names,
    const InputOverrides& overrides = {})
{
    LogicalGraph g("test");
    build_graph(g);
    auto compiled = CompiledGraph::compile(g);
    bind_inputs(compiled, g, input_names, overrides);
    compiled.execute();
    compiled.wait();
}

template<typename WrapperT>
inline void verify_graph_vs_tensor(
    const std::function<void(LogicalGraph&)>& build_graph,
    const std::function<void(std::map<std::string, std::vector<typename WrapperT::repr_t>>&,
                             std::map<std::string, std::vector<typename WrapperT::repr_t>>&,
                             const nntile::Context&)>& run_tensor_direct,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const nntile::Context& context,
    const std::map<std::string, std::vector<typename WrapperT::repr_t>>& custom_inputs = {},
    const InputOverrides& input_overrides = {})
{
    using ValueT = typename WrapperT::repr_t;

    LogicalGraph g("test");
    build_graph(g);

    auto compiled = CompiledGraph::compile(g);

    std::map<std::string, std::vector<ValueT>> input_data;
    std::map<std::string, std::vector<ValueT>> graph_outputs;

    InputOverrides merged = input_overrides;
    for(const auto& [name, data] : custom_inputs)
    {
        if constexpr (std::is_same_v<ValueT, double>)
        {
            merged.double_inputs[name] = data;
        }
        else
        {
            merged.float_inputs[name] = data;
        }
    }

    bind_inputs(compiled, g, input_names, merged);

    for(const auto& name : input_names)
    {
        const auto& tensor = g.get_tensor(name);
        const auto dtype = tensor->dtype();
        if(dtype == DataType::INT64)
        {
            auto it = merged.int64_inputs.find(name);
            if(it != merged.int64_inputs.end())
            {
                input_data[name].resize(it->second.size());
                for(size_t i = 0; i < it->second.size(); ++i)
                {
                    input_data[name][i] = static_cast<ValueT>(it->second[i]);
                }
            }
            else
            {
                input_data[name] = make_pattern<ValueT>(tensor_nelems(tensor->shape()),
                    static_cast<ValueT>(0.1));
            }
        }
        else if(dtype == DataType::BOOL)
        {
            auto it = merged.bool_inputs.find(name);
            if(it != merged.bool_inputs.end())
            {
                input_data[name].resize(it->second.size());
                for(size_t i = 0; i < it->second.size(); ++i)
                {
                    input_data[name][i] = static_cast<ValueT>(it->second[i].value ? 1 : 0);
                }
            }
            else
            {
                input_data[name] = make_pattern<ValueT>(tensor_nelems(tensor->shape()),
                    static_cast<ValueT>(0.1));
            }
        }
        else if(dtype == DataType::FP64)
        {
            auto it = merged.double_inputs.find(name);
            std::vector<double> double_data = (it != merged.double_inputs.end())
                ? it->second
                : make_pattern<double>(tensor_nelems(tensor->shape()), 0.1);
            input_data[name].resize(double_data.size());
            for(size_t i = 0; i < double_data.size(); ++i)
            {
                input_data[name][i] = static_cast<ValueT>(double_data[i]);
            }
        }
        else
        {
            auto it = merged.float_inputs.find(name);
            std::vector<float> float_data = (it != merged.float_inputs.end())
                ? it->second
                : make_pattern<float>(tensor_nelems(tensor->shape()), 0.1f);
            input_data[name].resize(float_data.size());
            for(size_t i = 0; i < float_data.size(); ++i)
            {
                input_data[name][i] = static_cast<ValueT>(float_data[i]);
            }
        }
    }

    compiled.execute();
    compiled.wait();

    for(const auto& name : output_names)
    {
        graph_outputs[name] = compiled.get_output<ValueT>(name);
    }

    std::map<std::string, std::vector<ValueT>> tensor_outputs;
    run_tensor_direct(input_data, tensor_outputs, context);

    for(const auto& name : output_names)
    {
        REQUIRE(graph_outputs[name].size() == tensor_outputs[name].size());
        for(size_t i = 0; i < graph_outputs[name].size(); ++i)
        {
            if constexpr (std::is_floating_point_v<ValueT>)
            {
                REQUIRE(graph_outputs[name][i] ==
                        Catch::Approx(tensor_outputs[name][i]).epsilon(1e-5));
            }
            else
            {
                REQUIRE(graph_outputs[name][i] == tensor_outputs[name][i]);
            }
        }
    }
}

} // namespace nntile::graph::test
