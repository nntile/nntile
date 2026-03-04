/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/graph_runtime.cc
 * TensorGraph::Runtime implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/graph_runtime.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor/graph_op_node.hh"
#include "nntile/tensor/tensor.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void allocate_and_register(
    const TensorGraph::TensorNode* node,
    const std::vector<Index>& shape,
    std::map<std::string, std::shared_ptr<void>>& runtime_data,
    std::map<std::string, DataType>& data_dtypes,
    std::map<const TensorGraph::TensorNode*, std::shared_ptr<void>>&
        tensor_map)
{
    std::vector<Index> tile_shape = shape;
    auto t = std::make_shared<nntile::tensor::Tensor<T>>(
        nntile::tensor::TensorTraits(shape, tile_shape));
    runtime_data[node->name()] = t;
    data_dtypes[node->name()] = node->dtype();
    tensor_map[node] = t;
}

} // namespace

TensorGraph::Runtime::Runtime(const TensorGraph& graph)
    : graph_(graph)
{
}

void TensorGraph::Runtime::compile()
{
    if(compiled_)
    {
        return;
    }

    allocate_impl();

    execution_order_.clear();
    execution_order_.reserve(graph_.ops().size());
    for(const auto& op : graph_.ops())
    {
        execution_order_.push_back(op);
    }

    data_is_input_.clear();
    data_is_output_.clear();
    for(const auto& node : graph_.tensor_nodes())
    {
        if(node->is_input())
        {
            data_is_input_.insert(node->name());
        }
        if(node->is_output())
        {
            data_is_output_.insert(node->name());
        }
    }

    eliminate_dead_ops();

    data_last_use_.clear();
    for(size_t i = 0; i < execution_order_.size(); ++i)
    {
        for(const auto* input : execution_order_[i]->inputs())
        {
            data_last_use_[input->name()] = i;
        }
    }

    compiled_ = true;
}

void TensorGraph::Runtime::allocate_impl()
{
    for(const auto& node : graph_.tensor_nodes())
    {
        DataType dtype = node->dtype();
        std::vector<Index> shape = node->shape();

        switch(dtype)
        {
            case DataType::FP32:
                allocate_and_register<nntile::fp32_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tensor_map_);
                break;
            case DataType::FP32_FAST_TF32:
                allocate_and_register<nntile::fp32_fast_tf32_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tensor_map_);
                break;
            case DataType::FP32_FAST_FP16:
                allocate_and_register<nntile::fp32_fast_fp16_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tensor_map_);
                break;
            case DataType::FP32_FAST_BF16:
                allocate_and_register<nntile::fp32_fast_bf16_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tensor_map_);
                break;
            case DataType::FP64:
                allocate_and_register<nntile::fp64_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tensor_map_);
                break;
            case DataType::FP16:
                allocate_and_register<nntile::fp16_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tensor_map_);
                break;
            case DataType::BF16:
                allocate_and_register<nntile::bf16_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tensor_map_);
                break;
            case DataType::INT64:
                allocate_and_register<nntile::int64_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tensor_map_);
                break;
            case DataType::BOOL:
                allocate_and_register<nntile::bool_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tensor_map_);
                break;
            default:
                throw std::runtime_error(
                    "Unsupported data type for tensor allocation");
        }
    }
}

void TensorGraph::Runtime::eliminate_dead_ops()
{
    const size_t n = execution_order_.size();
    if(n == 0)
    {
        return;
    }

    std::unordered_map<std::string, std::unordered_set<size_t>> producer;
    std::unordered_map<std::string, std::unordered_set<size_t>> consumer;
    std::unordered_set<std::string> consumed;

    for(size_t i = 0; i < n; ++i)
    {
        const auto& op = execution_order_[i];
        for(const auto* out : op->outputs())
        {
            producer[out->name()].insert(i);
        }
        for(const auto* in : op->inputs())
        {
            consumed.insert(in->name());
            consumer[in->name()].insert(i);
        }
    }

    std::unordered_set<std::string> live_data(
        data_is_output_.begin(), data_is_output_.end());
    for(const auto& name : data_is_input_)
    {
        live_data.insert(name);
    }
    if(data_is_output_.empty())
    {
        for(const auto& p : producer)
        {
            if(consumed.count(p.first) == 0)
            {
                live_data.insert(p.first);
            }
        }
    }
    if(live_data.empty())
    {
        return;
    }

    std::set<size_t> live_ops;
    bool changed = true;
    while(changed)
    {
        changed = false;
        auto live_data_copy = live_data;
        for(const auto& t : live_data_copy)
        {
            auto prod_it = producer.find(t);
            if(prod_it != producer.end())
            {
                for(size_t op_idx : prod_it->second)
                {
                    if(live_ops.insert(op_idx).second)
                    {
                        changed = true;
                        for(const auto* in : execution_order_[op_idx]->inputs())
                        {
                            if(live_data.insert(in->name()).second)
                            {
                                changed = true;
                            }
                        }
                    }
                }
            }
            auto cons_it = consumer.find(t);
            if(cons_it != consumer.end())
            {
                for(size_t op_idx : cons_it->second)
                {
                    if(execution_order_[op_idx]->outputs().empty() &&
                       live_ops.insert(op_idx).second)
                    {
                        changed = true;
                        for(const auto* in : execution_order_[op_idx]->inputs())
                        {
                            if(live_data.insert(in->name()).second)
                            {
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<std::shared_ptr<OpNode>> filtered;
    filtered.reserve(live_ops.size());
    for(size_t i = 0; i < n; ++i)
    {
        if(live_ops.count(i))
        {
            filtered.push_back(execution_order_[i]);
        }
    }
    execution_order_ = std::move(filtered);
}

void TensorGraph::Runtime::execute()
{
    if(!compiled_)
    {
        throw std::runtime_error(
            "TensorGraph::Runtime::execute: graph not compiled");
    }
    for(size_t i = 0; i < execution_order_.size(); ++i)
    {
        execution_order_[i]->execute(*this);
        invalidate_unused_tensors(i);
    }
}

void TensorGraph::Runtime::wait()
{
    starpu_task_wait_for_all();
}

void TensorGraph::Runtime::invalidate_data(const std::string& name)
{
    auto dtype_it = data_dtypes_.find(name);
    if(dtype_it == data_dtypes_.end())
    {
        return;
    }
    DataType dtype = dtype_it->second;

    switch(dtype)
    {
        case DataType::FP32:
            get_data<nntile::fp32_t>(name).invalidate_submit();
            break;
        case DataType::FP32_FAST_TF32:
            get_data<nntile::fp32_fast_tf32_t>(name).invalidate_submit();
            break;
        case DataType::FP32_FAST_FP16:
            get_data<nntile::fp32_fast_fp16_t>(name).invalidate_submit();
            break;
        case DataType::FP32_FAST_BF16:
            get_data<nntile::fp32_fast_bf16_t>(name).invalidate_submit();
            break;
        case DataType::FP64:
            get_data<nntile::fp64_t>(name).invalidate_submit();
            break;
        case DataType::FP16:
            get_data<nntile::fp16_t>(name).invalidate_submit();
            break;
        case DataType::BF16:
            get_data<nntile::bf16_t>(name).invalidate_submit();
            break;
        case DataType::INT64:
            get_data<nntile::int64_t>(name).invalidate_submit();
            break;
        case DataType::BOOL:
            get_data<nntile::bool_t>(name).invalidate_submit();
            break;
        default:
            throw std::runtime_error(
                "invalidate_data: unsupported data type " +
                dtype_to_string(dtype) + " for data '" + name + "'");
    }
}

void TensorGraph::Runtime::invalidate_unused_tensors(size_t op_idx)
{
    const auto& op = execution_order_.at(op_idx);
    std::unordered_set<std::string> seen;

    // Invalidate inputs that were last used by this op
    for(const auto* input : op->inputs())
    {
        const std::string& input_name = input->name();
        if(!seen.insert(input_name).second)
        {
            continue;
        }
        if(data_is_input_.count(input_name) ||
           data_is_output_.count(input_name))
        {
            continue;
        }
        bool is_inplace = false;
        for(const auto* out : op->outputs())
        {
            if(out->name() == input_name)
            {
                is_inplace = true;
                break;
            }
        }
        if(is_inplace)
        {
            continue;
        }
        auto it = data_last_use_.find(input_name);
        if(it != data_last_use_.end() && it->second == op_idx)
        {
            invalidate_data(input_name);
        }
    }

    // Invalidate outputs that are never consumed (data_last_use_ only tracks
    // inputs; outputs without consumers would otherwise leak memory)
    for(const auto* output : op->outputs())
    {
        const std::string& output_name = output->name();
        if(!seen.insert(output_name).second)
        {
            continue;
        }
        if(data_is_input_.count(output_name) ||
           data_is_output_.count(output_name))
        {
            continue;
        }
        if(data_last_use_.count(output_name) == 0)
        {
            invalidate_data(output_name);
        }
    }
}

} // namespace nntile::graph
