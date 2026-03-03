/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/graph_runtime.hh
 * TensorGraph::Runtime - runtime execution of a TensorGraph.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <starpu.h>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tensor/graph_decl.hh>
#include <nntile/graph/tensor/graph_data_node.hh>
#include <nntile/tensor/tensor.hh>

namespace nntile::graph
{

//! Runtime - holds runtime state for executing a TensorGraph.
//! Takes a reference to the symbolic graph; compile() allocates and builds
//! execution order; execute() runs the ops.
class TensorGraph::Runtime
{
public:
    using TensorNode = TensorGraph::TensorNode;
    using OpNode = TensorGraph::OpNode;

    explicit Runtime(const TensorGraph& graph);

    //! Compile the graph for execution. Allocates runtime data.
    void compile();

    //! Bind data to a data node (copies data)
    template<typename T>
    void bind_data(const std::string& name, const T* data, size_t count);

    //! Bind data from vector
    template<typename T>
    void bind_data(const std::string& name, const std::vector<T>& data);

    //! Execute the graph
    void execute();

    //! Wait for all operations to complete
    void wait();

    //! Get output data (copies data out)
    template<typename T>
    std::vector<T> get_output(const std::string& name);

    //! Get typed runtime data by name (for bind_data, get_output)
    template<typename T>
    nntile::tensor::Tensor<T>& get_data(const std::string& name);

    //! Get typed runtime tensor for a data node (used by OpNode::execute)
    template<typename T>
    nntile::tensor::Tensor<T>& get_tensor(const TensorNode* node);

    //! Get data type of a data node by name
    DataType get_dtype(const std::string& name) const
    {
        return data_dtypes_.at(name);
    }

    //! Get data type of a data node
    DataType get_dtype(const TensorNode* node) const
    {
        return node->dtype();
    }

    //! True if compile() has been called
    bool is_compiled() const { return compiled_; }

private:
    void allocate_impl();
    void eliminate_dead_ops();
    void invalidate_data(const std::string& name);
    void invalidate_unused_inputs(size_t op_idx);

    const TensorGraph& graph_;
    std::map<const TensorNode*, std::shared_ptr<void>> tensor_map_;
    std::map<std::string, std::shared_ptr<void>> runtime_data_;
    std::map<std::string, DataType> data_dtypes_;
    std::vector<std::shared_ptr<OpNode>> execution_order_;
    std::set<std::string> data_is_input_;
    std::set<std::string> data_is_output_;
    std::map<std::string, size_t> data_last_use_;
    bool compiled_ = false;
};

// -----------------------------------------------------------------------------
// Template implementation
// -----------------------------------------------------------------------------

template<typename T>
nntile::tensor::Tensor<T>& TensorGraph::Runtime::get_data(const std::string& name)
{
    auto it = runtime_data_.find(name);
    if(it == runtime_data_.end())
    {
        throw std::runtime_error("Data not found: " + name);
    }
    return *static_cast<nntile::tensor::Tensor<T>*>(it->second.get());
}

template<typename T>
nntile::tensor::Tensor<T>& TensorGraph::Runtime::get_tensor(
    const TensorNode* node)
{
    auto it = tensor_map_.find(node);
    if(it == tensor_map_.end())
    {
        throw std::runtime_error(
            "Runtime::get_tensor: node not found");
    }
    auto ptr = std::static_pointer_cast<nntile::tensor::Tensor<T>>(it->second);
    if(!ptr)
    {
        throw std::runtime_error(
            "Runtime::get_tensor: wrong type");
    }
    return *ptr;
}

template<typename T>
void TensorGraph::Runtime::bind_data(const std::string& name, const T* data,
                                    size_t count)
{
    auto it = runtime_data_.find(name);
    if(it == runtime_data_.end())
    {
        throw std::runtime_error("Data not found: " + name);
    }
    if(!data_is_input_.count(name) && !data_is_output_.count(name))
    {
        throw std::runtime_error(
            "bind_data: data '" + name +
            "' must be marked as input or output (or both); "
            "call mark_input(true) or mark_output(true) on the data node");
    }

    auto dtype_it = data_dtypes_.find(name);
    if(dtype_it == data_dtypes_.end())
    {
        throw std::runtime_error("bind_data: data dtype not found: " + name);
    }
    DataType dtype = dtype_it->second;

    if(dtype == DataType::FP32)
    {
        auto& tensor = get_data<nntile::fp32_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for data " + name);
        }
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] = nntile::fp32_t(static_cast<float>(data[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::FP32_FAST_TF32)
    {
        auto& tensor = get_data<nntile::fp32_fast_tf32_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for data " + name);
        }
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] =
                nntile::fp32_fast_tf32_t(static_cast<float>(data[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::FP32_FAST_FP16)
    {
        auto& tensor = get_data<nntile::fp32_fast_fp16_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for data " + name);
        }
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] =
                nntile::fp32_fast_fp16_t(static_cast<float>(data[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::FP32_FAST_BF16)
    {
        auto& tensor = get_data<nntile::fp32_fast_bf16_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for data " + name);
        }
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] =
                nntile::fp32_fast_bf16_t(static_cast<float>(data[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::FP64)
    {
        auto& tensor = get_data<nntile::fp64_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for data " + name);
        }
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] = nntile::fp64_t(static_cast<double>(data[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::FP16)
    {
        auto& tensor = get_data<nntile::fp16_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for data " + name);
        }
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] = nntile::fp16_t(static_cast<float>(data[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::BF16)
    {
        auto& tensor = get_data<nntile::bf16_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for data " + name);
        }
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] = nntile::bf16_t(static_cast<float>(data[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::INT64)
    {
        auto& tensor = get_data<nntile::int64_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for data " + name);
        }
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] = nntile::int64_t(
                static_cast<std::int64_t>(data[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::BOOL)
    {
        auto& tensor = get_data<nntile::bool_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for data " + name);
        }
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] = nntile::bool_t(static_cast<bool>(data[i]));
        }
        tile_local.release();
    }
    else
    {
        throw std::runtime_error("Unsupported data type for binding");
    }
}

template<typename T>
void TensorGraph::Runtime::bind_data(const std::string& name,
                                    const std::vector<T>& data)
{
    bind_data(name, data.data(), data.size());
}

template<typename T>
std::vector<T> TensorGraph::Runtime::get_output(const std::string& name)
{
    auto data_it = runtime_data_.find(name);
    if(data_it == runtime_data_.end())
    {
        throw std::runtime_error("Data not found: " + name);
    }
    auto dtype_it = data_dtypes_.find(name);
    if(dtype_it == data_dtypes_.end())
    {
        throw std::runtime_error("Data dtype not found: " + name);
    }
    DataType dtype = dtype_it->second;
    std::vector<T> result;

    if(dtype == DataType::FP32)
    {
        auto& tensor = get_data<nntile::fp32_t>(name);
        result.resize(tensor.nelems);
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        for(Index i = 0; i < tensor.nelems; ++i)
        {
            result[i] = static_cast<T>(static_cast<float>(tile_local[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::FP32_FAST_TF32)
    {
        auto& tensor = get_data<nntile::fp32_fast_tf32_t>(name);
        result.resize(tensor.nelems);
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        for(Index i = 0; i < tensor.nelems; ++i)
        {
            result[i] = static_cast<T>(static_cast<float>(tile_local[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::FP32_FAST_FP16)
    {
        auto& tensor = get_data<nntile::fp32_fast_fp16_t>(name);
        result.resize(tensor.nelems);
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        for(Index i = 0; i < tensor.nelems; ++i)
        {
            result[i] = static_cast<T>(static_cast<float>(tile_local[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::FP32_FAST_BF16)
    {
        auto& tensor = get_data<nntile::fp32_fast_bf16_t>(name);
        result.resize(tensor.nelems);
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        for(Index i = 0; i < tensor.nelems; ++i)
        {
            result[i] = static_cast<T>(static_cast<float>(tile_local[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::FP64)
    {
        auto& tensor = get_data<nntile::fp64_t>(name);
        result.resize(tensor.nelems);
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        for(Index i = 0; i < tensor.nelems; ++i)
        {
            result[i] = static_cast<T>(static_cast<double>(tile_local[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::FP16)
    {
        auto& tensor = get_data<nntile::fp16_t>(name);
        result.resize(tensor.nelems);
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        for(Index i = 0; i < tensor.nelems; ++i)
        {
            result[i] = static_cast<T>(static_cast<float>(tile_local[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::BF16)
    {
        auto& tensor = get_data<nntile::bf16_t>(name);
        result.resize(tensor.nelems);
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        for(Index i = 0; i < tensor.nelems; ++i)
        {
            result[i] = static_cast<T>(static_cast<float>(tile_local[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::INT64)
    {
        auto& tensor = get_data<nntile::int64_t>(name);
        result.resize(tensor.nelems);
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        for(Index i = 0; i < tensor.nelems; ++i)
        {
            result[i] = static_cast<T>(
                static_cast<std::int64_t>(tile_local[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::BOOL)
    {
        auto& tensor = get_data<nntile::bool_t>(name);
        result.resize(tensor.nelems);
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        for(Index i = 0; i < tensor.nelems; ++i)
        {
            result[i] = static_cast<T>(static_cast<bool>(tile_local[i]));
        }
        tile_local.release();
    }
    else
    {
        throw std::runtime_error("Unsupported data type for get_output");
    }

    return result;
}

} // namespace nntile::graph
