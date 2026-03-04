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
    void invalidate_unused_tensors(size_t op_idx);

    //! Template helpers for bind_data/get_output - dispatch on DataType
    template<typename T, typename NntileT, typename CastT>
    void bind_data_impl(const std::string& name, const T* data, size_t count);
    template<typename T, typename NntileT, typename CastT>
    void get_output_impl(const std::string& name, std::vector<T>& result);

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

namespace detail
{

//! Map tensor element type to DataType for runtime type checking in get_tensor.
//! Only the supported tensor types below have specializations; unsupported T
//! triggers a compile error via the generic template.
template<typename T>
struct dtype_for
{
    static_assert(sizeof(T) == 0,
                  "Unsupported tensor type for get_tensor; use fp32_t, fp64_t, "
                  "fp16_t, bf16_t, int64_t, bool_t, or fp32_fast_* variants");
};

template<> struct dtype_for<nntile::fp32_t>
{
    static constexpr DataType value = DataType::FP32;
};
template<> struct dtype_for<nntile::fp32_fast_tf32_t>
{
    static constexpr DataType value = DataType::FP32_FAST_TF32;
};
template<> struct dtype_for<nntile::fp32_fast_fp16_t>
{
    static constexpr DataType value = DataType::FP32_FAST_FP16;
};
template<> struct dtype_for<nntile::fp32_fast_bf16_t>
{
    static constexpr DataType value = DataType::FP32_FAST_BF16;
};
template<> struct dtype_for<nntile::fp64_t>
{
    static constexpr DataType value = DataType::FP64;
};
template<> struct dtype_for<nntile::fp16_t>
{
    static constexpr DataType value = DataType::FP16;
};
template<> struct dtype_for<nntile::bf16_t>
{
    static constexpr DataType value = DataType::BF16;
};
template<> struct dtype_for<nntile::int64_t>
{
    static constexpr DataType value = DataType::INT64;
};
template<> struct dtype_for<nntile::bool_t>
{
    static constexpr DataType value = DataType::BOOL;
};

} // namespace detail

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
    // Check that requested type T matches the node's dtype before casting.
    // static_pointer_cast from shared_ptr<void> always succeeds (non-null),
    // so we must verify the type explicitly to avoid undefined behavior.
    if(node->dtype() != detail::dtype_for<T>::value)
    {
        throw std::runtime_error(
            "Runtime::get_tensor: wrong type (requested type does not match "
            "tensor dtype)");
    }
    auto ptr = std::static_pointer_cast<nntile::tensor::Tensor<T>>(it->second);
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

    switch(dtype)
    {
        case DataType::FP32:
            bind_data_impl<T, nntile::fp32_t, float>(name, data, count);
            break;
        case DataType::FP32_FAST_TF32:
            bind_data_impl<T, nntile::fp32_fast_tf32_t, float>(name, data,
                                                               count);
            break;
        case DataType::FP32_FAST_FP16:
            bind_data_impl<T, nntile::fp32_fast_fp16_t, float>(name, data,
                                                                count);
            break;
        case DataType::FP32_FAST_BF16:
            bind_data_impl<T, nntile::fp32_fast_bf16_t, float>(name, data,
                                                               count);
            break;
        case DataType::FP64:
            bind_data_impl<T, nntile::fp64_t, double>(name, data, count);
            break;
        case DataType::FP16:
            bind_data_impl<T, nntile::fp16_t, float>(name, data, count);
            break;
        case DataType::BF16:
            bind_data_impl<T, nntile::bf16_t, float>(name, data, count);
            break;
        case DataType::INT64:
            bind_data_impl<T, nntile::int64_t, std::int64_t>(name, data,
                                                            count);
            break;
        case DataType::BOOL:
            bind_data_impl<T, nntile::bool_t, bool>(name, data, count);
            break;
        default:
            throw std::runtime_error("Unsupported data type for binding");
    }
}

template<typename T>
void TensorGraph::Runtime::bind_data(const std::string& name,
                                    const std::vector<T>& data)
{
    bind_data(name, data.data(), data.size());
}

template<typename T, typename NntileT, typename CastT>
void TensorGraph::Runtime::bind_data_impl(const std::string& name,
                                          const T* data, size_t count)
{
    auto& tensor = get_data<NntileT>(name);
    if(tensor.grid.nelems != 1)
    {
        throw std::runtime_error(
            "bind_data: data '" + name +
            "' has more than one tile (grid.nelems=" +
            std::to_string(tensor.grid.nelems) +
            "); only single-tile tensors are supported");
    }
    if(count != static_cast<size_t>(tensor.nelems))
    {
        throw std::runtime_error("Data size mismatch for data " + name);
    }
    auto tile = tensor.get_tile(0);
    auto tile_local = tile.acquire(STARPU_W);
    for(size_t i = 0; i < count; ++i)
    {
        tile_local[i] = NntileT(static_cast<CastT>(data[i]));
    }
    tile_local.release();
}

template<typename T>
std::vector<T> TensorGraph::Runtime::get_output(const std::string& name)
{
    auto data_it = runtime_data_.find(name);
    if(data_it == runtime_data_.end())
    {
        throw std::runtime_error("Data not found: " + name);
    }
    if(!data_is_output_.count(name))
    {
        throw std::runtime_error(
            "get_output: data '" + name +
            "' is not marked as output; intermediate tensors are invalidated "
            "during execution; call mark_output(true) on the data node");
    }
    auto dtype_it = data_dtypes_.find(name);
    if(dtype_it == data_dtypes_.end())
    {
        throw std::runtime_error("Data dtype not found: " + name);
    }
    DataType dtype = dtype_it->second;
    std::vector<T> result;

    switch(dtype)
    {
        case DataType::FP32:
            get_output_impl<T, nntile::fp32_t, float>(name, result);
            break;
        case DataType::FP32_FAST_TF32:
            get_output_impl<T, nntile::fp32_fast_tf32_t, float>(name, result);
            break;
        case DataType::FP32_FAST_FP16:
            get_output_impl<T, nntile::fp32_fast_fp16_t, float>(name, result);
            break;
        case DataType::FP32_FAST_BF16:
            get_output_impl<T, nntile::fp32_fast_bf16_t, float>(name, result);
            break;
        case DataType::FP64:
            get_output_impl<T, nntile::fp64_t, double>(name, result);
            break;
        case DataType::FP16:
            get_output_impl<T, nntile::fp16_t, float>(name, result);
            break;
        case DataType::BF16:
            get_output_impl<T, nntile::bf16_t, float>(name, result);
            break;
        case DataType::INT64:
            get_output_impl<T, nntile::int64_t, std::int64_t>(name, result);
            break;
        case DataType::BOOL:
            get_output_impl<T, nntile::bool_t, bool>(name, result);
            break;
        default:
            throw std::runtime_error("Unsupported data type for get_output");
    }

    return result;
}

template<typename T, typename NntileT, typename CastT>
void TensorGraph::Runtime::get_output_impl(const std::string& name,
                                           std::vector<T>& result)
{
    auto& tensor = get_data<NntileT>(name);
    if(tensor.grid.nelems != 1)
    {
        throw std::runtime_error(
            "get_output: data '" + name +
            "' has more than one tile (grid.nelems=" +
            std::to_string(tensor.grid.nelems) +
            "); only single-tile tensors are supported");
    }
    result.resize(tensor.nelems);
    auto tile = tensor.get_tile(0);
    auto tile_local = tile.acquire(STARPU_R);
    for(Index i = 0; i < tensor.nelems; ++i)
    {
        result[i] = static_cast<T>(static_cast<CastT>(tile_local[i]));
    }
    tile_local.release();
}

} // namespace nntile::graph
