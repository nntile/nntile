/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/optim/optimizer.cc
 * Base Optimizer class implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/optim/optimizer.hh"

#include <cstring>
#include <iostream>
#include <stdexcept>

#include "nntile/graph/dtype.hh"

namespace nntile::graph::optim
{

Optimizer::Optimizer(NNGraph* graph, module::Module* module)
    : graph_(graph)
{
    if(graph_ == nullptr)
    {
        throw std::invalid_argument(
            "Optimizer: graph pointer must be non-null");
    }
    if(module == nullptr)
    {
        throw std::invalid_argument(
            "Optimizer: module pointer must be non-null");
    }
    collect_params(module);
}

void Optimizer::collect_params(module::Module* module)
{
    auto param_grads = module->parameter_gradients_recursive();
    auto named_params = module->named_parameters_recursive();

    for(std::size_t i = 0; i < named_params.size(); ++i)
    {
        const auto& [pname, param] = named_params[i];
        NNGraph::TensorNode* grad = param->grad();
        if(grad == nullptr)
        {
            continue;
        }
        ParamState ps;
        ps.name = pname;
        ps.param = param;
        ps.grad = grad;
        param_states_.push_back(std::move(ps));
    }
}

std::vector<std::pair<std::string, NNGraph::TensorNode*>>
Optimizer::named_state_tensors() const
{
    std::vector<std::pair<std::string, NNGraph::TensorNode*>> result;
    for(const auto& ps : param_states_)
    {
        for(const auto& [buf_name, buf_tensor] : ps.buffers)
        {
            result.emplace_back(buf_name, buf_tensor);
        }
    }
    return result;
}

void Optimizer::save(const std::string& path) const
{
    io::SafeTensorsWriter writer;
    for(const auto& ps : param_states_)
    {
        for(const auto& [buf_name, buf_tensor] : ps.buffers)
        {
            if(buf_tensor == nullptr)
            {
                continue;
            }
            const auto* hint = buf_tensor->data()->get_bind_hint();
            if(hint == nullptr)
            {
                continue;
            }
            const auto& idx_shape = buf_tensor->shape();
            std::vector<std::int64_t> shape(idx_shape.begin(),
                                            idx_shape.end());
            writer.add_tensor(buf_name, buf_tensor->dtype(), shape, *hint);
        }
    }
    writer.write(path);
}

void Optimizer::load(const std::string& path)
{
    io::SafeTensorsReader reader(path);
    for(auto& ps : param_states_)
    {
        for(auto& [buf_name, buf_tensor] : ps.buffers)
        {
            if(buf_tensor == nullptr)
            {
                continue;
            }
            if(!reader.has_tensor(buf_name))
            {
                continue;
            }
            const auto& info = reader.tensor_info(buf_name);
            if(!io::is_safetensors_dtype_compatible(
                   io::dtype_to_safetensors(info.dtype), buf_tensor->dtype()))
            {
                throw std::runtime_error(
                    "Optimizer::load: dtype mismatch for '" + buf_name + "'");
            }
            auto data = reader.read_tensor(buf_name);
            buf_tensor->data()->set_bind_hint(std::move(data));
            buf_tensor->mark_input(true);
        }
    }
}

namespace
{

template<typename T>
std::vector<std::uint8_t> get_output_bytes(TensorGraph::Runtime& runtime,
                                           const std::string& name)
{
    auto data = runtime.get_output<T>(name);
    std::vector<std::uint8_t> bytes(data.size() * sizeof(T));
    std::memcpy(bytes.data(), data.data(), bytes.size());
    return bytes;
}

std::vector<std::uint8_t> sync_tensor_bytes(TensorGraph::Runtime& runtime,
                                            const std::string& name,
                                            DataType dtype)
{
    switch(dtype)
    {
        case DataType::FP32:
        case DataType::FP32_FAST_TF32:
        case DataType::FP32_FAST_FP16:
        case DataType::FP32_FAST_BF16:
            return get_output_bytes<float>(runtime, name);
        case DataType::FP64:
            return get_output_bytes<double>(runtime, name);
        case DataType::INT64:
            return get_output_bytes<std::int64_t>(runtime, name);
        default:
            throw std::runtime_error(
                "sync_tensor_bytes: unsupported dtype for tensor '" +
                name + "'");
    }
}

} // anonymous namespace

void Optimizer::sync_from_runtime(TensorGraph::Runtime& runtime)
{
    for(auto& ps : param_states_)
    {
        for(auto& [buf_name, buf_tensor] : ps.buffers)
        {
            if(buf_tensor == nullptr)
            {
                continue;
            }
            auto bytes = sync_tensor_bytes(
                runtime, buf_name, buf_tensor->dtype());
            buf_tensor->data()->set_bind_hint(std::move(bytes));
        }
    }
}

void Optimizer::import_hf(const io::SafeTensorsReader& reader,
                          const std::string& prefix)
{
    for(auto& ps : param_states_)
    {
        for(auto& [buf_name, buf_tensor] : ps.buffers)
        {
            if(buf_tensor == nullptr)
            {
                continue;
            }
            std::string hf_name = prefix.empty()
                ? buf_name
                : prefix + "." + buf_name;
            if(!reader.has_tensor(hf_name))
            {
                continue;
            }
            auto data = reader.read_tensor(hf_name);
            buf_tensor->data()->set_bind_hint(std::move(data));
            buf_tensor->mark_input(true);
        }
    }
}

void Optimizer::export_hf(io::SafeTensorsWriter& writer,
                          const std::string& prefix) const
{
    for(const auto& ps : param_states_)
    {
        for(const auto& [buf_name, buf_tensor] : ps.buffers)
        {
            if(buf_tensor == nullptr)
            {
                continue;
            }
            const auto* hint = buf_tensor->data()->get_bind_hint();
            if(hint == nullptr)
            {
                continue;
            }
            std::string hf_name = prefix.empty()
                ? buf_name
                : prefix + "." + buf_name;
            const auto& idx_shape = buf_tensor->shape();
            std::vector<std::int64_t> shape(idx_shape.begin(),
                                            idx_shape.end());
            writer.add_tensor(hf_name, buf_tensor->dtype(), shape, *hint);
        }
    }
}

std::string Optimizer::repr() const
{
    return "Optimizer(num_params=" +
           std::to_string(param_states_.size()) + ")";
}

std::string Optimizer::to_string() const
{
    std::string s = repr() + "\n";
    for(const auto& ps : param_states_)
    {
        s += "  " + ps.name + ":";
        for(const auto& [buf_name, buf_tensor] : ps.buffers)
        {
            s += " " + buf_name;
        }
        s += "\n";
    }
    return s;
}

void Optimizer::print() const
{
    std::cout << to_string() << std::endl;
}

} // namespace nntile::graph::optim
