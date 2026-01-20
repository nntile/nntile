/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled_graph.cc
 * Implementation of CompiledGraph class (main body).
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/compiled_graph.hh"

// Include standard headers
#include <algorithm>
#include <memory>
#include <stdexcept>

// Include third-party headers
#include <starpu.h>  // For STARPU_W, STARPU_R

// Include other NNTile headers
#include "nntile/base_types.hh"
#include "nntile/constants.hh"
#include "nntile/graph/op_node.hh"
#include "nntile/graph/tensor_spec.hh"
#include "nntile/graph/compiled/gemm.hh"
#include "nntile/graph/compiled/gelu.hh"
#include "nntile/tensor/tensor.hh"

namespace nntile::graph
{

//! Validate that operation data types are supported
void validate_operation_data_types(const LogicalGraph& logical)
{
    for(const auto& op : logical.ops())
    {
        // Get the data type from the first input tensor
        const auto* first_input = op->inputs()[0];
        DataType dtype = first_input->spec().dtype();

        // Check supported data types based on operation type
        if(op->type() == OpType::GELU || op->type() == OpType::GEMM)
        {
            // These operations support floating point types only
            if(dtype != DataType::FP32 &&
               dtype != DataType::FP32_FAST_TF32 &&
               dtype != DataType::FP32_FAST_FP16 &&
               dtype != DataType::FP32_FAST_BF16 &&
               dtype != DataType::FP64 &&
               dtype != DataType::FP16 &&
               dtype != DataType::BF16)
            {
                throw std::runtime_error(
                    std::string(op_type_to_string(op->type())) +
                    " operation does not support data type " +
                    dtype_to_string(dtype));
            }
        }
        // Add validation for other operations here as they are added
    }
}

//! Compile a logical graph
CompiledGraph CompiledGraph::compile(const LogicalGraph& logical)
{
    // Validate data types before allocating tensors
    validate_operation_data_types(logical);

    CompiledGraph cg;
    cg.allocate_tensors(logical);

    // Build execution order directly from logical graph ops
    // (already topologically sorted)
    cg.execution_order_.clear();
    cg.execution_order_.reserve(logical.num_ops());
    for(const auto& op : logical.ops())
    {
        OpExecutionInfo op_info;
        op_info.type = op->type();
        op_info.attrs = op->attrs();
        op_info.input_names.reserve(op->inputs().size());
        for(const auto* input : op->inputs())
        {
            op_info.input_names.push_back(input->name());
        }
        op_info.output_names.reserve(op->outputs().size());
        for(const auto* output : op->outputs())
        {
            op_info.output_names.push_back(output->name());
        }
        cg.execution_order_.push_back(op_info);
    }

    return cg;
}

//! Allocate NNTile tensors for all graph tensors
void CompiledGraph::allocate_tensors(const LogicalGraph& logical)
{
    for(const auto& node : logical.tensors())
    {
        const auto& spec = node->spec();
        tensor_dtypes_[node->name()] = spec.dtype();

        // Create tensor with single tile (no tiling)
        std::vector<Index> shape = spec.shape();
        std::vector<Index> tile_shape = shape;  // Same as shape = 1 tile

        switch(spec.dtype())
        {
            case DataType::FP32:
            {
                auto t = std::make_shared<
                    nntile::tensor::Tensor<nntile::fp32_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::FP32_FAST_TF32:
            {
                auto t = std::make_shared<
                    nntile::tensor::Tensor<nntile::fp32_fast_tf32_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::FP32_FAST_FP16:
            {
                auto t = std::make_shared<
                    nntile::tensor::Tensor<nntile::fp32_fast_fp16_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::FP32_FAST_BF16:
            {
                auto t = std::make_shared<
                    nntile::tensor::Tensor<nntile::fp32_fast_bf16_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::FP64:
            {
                auto t = std::make_shared<
                    nntile::tensor::Tensor<nntile::fp64_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::FP16:
            {
                auto t = std::make_shared<
                    nntile::tensor::Tensor<nntile::fp16_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::BF16:
            {
                auto t = std::make_shared<
                    nntile::tensor::Tensor<nntile::bf16_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::INT64:
            {
                auto t = std::make_shared<
                    nntile::tensor::Tensor<nntile::int64_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::INT32:
            {
                // INT32 maps to int32_t, but NNTile doesn't have a
                // wrapper for it
                // For now, throw an error as it's not commonly used
                throw std::runtime_error("INT32 data type not yet supported");
            }
            case DataType::BOOL:
            {
                auto t = std::make_shared<
                    nntile::tensor::Tensor<nntile::bool_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            default:
                throw std::runtime_error(
                    "Unsupported data type for tensor allocation");
        }
    }
}

//! Execute the graph
void CompiledGraph::execute()
{
    for(const auto& op_info : execution_order_)
    {
        execute_op(op_info);
    }
}

//! Wait for all operations to complete
void CompiledGraph::wait()
{
    // For now, wait for all StarPU tasks globally
    // Later, we will change it with actual wait on tasks of the compiled graph
    starpu_task_wait_for_all();
}

//! Execute a single operation
void CompiledGraph::execute_op(const OpExecutionInfo& op_info)
{
    switch(op_info.type)
    {
        case OpType::GEMM:
            execute_gemm(*this, op_info);
            break;
        case OpType::GELU:
            execute_gelu(*this, op_info);
            break;
    }
}

//! Get typed tensor pointer
template<typename T>
nntile::tensor::Tensor<T>& CompiledGraph::get_tensor(const std::string& name)
{
    auto it = tensors_.find(name);
    if(it == tensors_.end())
    {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return *static_cast<nntile::tensor::Tensor<T>*>(it->second.get());
}

//! Bind data to a tensor (copies data)
template<typename T>
void CompiledGraph::bind_data(const std::string& name, const T* data,
                              size_t count)
{
    auto it = tensors_.find(name);
    if(it == tensors_.end())
    {
        throw std::runtime_error("Tensor not found: " + name);
    }

    DataType dtype = tensor_dtypes_[name];

    // Check count matches tensor size and convert data to
    // appropriate wrapper type
    if(dtype == DataType::FP32)
    {
        auto& tensor = get_tensor<nntile::fp32_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for tensor " + name);
        }

        // Acquire the single tile and copy data (converting to fp32_t)
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
        auto& tensor = get_tensor<nntile::fp32_fast_tf32_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for tensor " + name);
        }

        // Acquire the single tile and copy data
        // (converting to fp32_fast_tf32_t)
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] = nntile::fp32_fast_tf32_t(
                static_cast<float>(data[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::FP32_FAST_FP16)
    {
        auto& tensor = get_tensor<nntile::fp32_fast_fp16_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for tensor " + name);
        }

        // Acquire the single tile and copy data
        // (converting to fp32_fast_fp16_t)
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] = nntile::fp32_fast_fp16_t(
                static_cast<float>(data[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::FP32_FAST_BF16)
    {
        auto& tensor = get_tensor<nntile::fp32_fast_bf16_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for tensor " + name);
        }

        // Acquire the single tile and copy data
        // (converting to fp32_fast_bf16_t)
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] = nntile::fp32_fast_bf16_t(
                static_cast<float>(data[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::FP64)
    {
        auto& tensor = get_tensor<nntile::fp64_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for tensor " + name);
        }

        // Acquire the single tile and copy data (converting to fp64_t)
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
        auto& tensor = get_tensor<nntile::fp16_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for tensor " + name);
        }

        // Acquire the single tile and copy data (converting to fp16_t)
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
        auto& tensor = get_tensor<nntile::bf16_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for tensor " + name);
        }

        // Acquire the single tile and copy data (converting to bf16_t)
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
        auto& tensor = get_tensor<nntile::int64_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for tensor " + name);
        }

        // Acquire the single tile and copy data (converting to int64_t)
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] = nntile::int64_t(static_cast<long long>(data[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::INT32)
    {
        throw std::runtime_error("INT32 data type not supported for binding");
    }
    else if(dtype == DataType::BOOL)
    {
        auto& tensor = get_tensor<nntile::bool_t>(name);
        if(count != static_cast<size_t>(tensor.nelems))
        {
            throw std::runtime_error("Data size mismatch for tensor " + name);
        }

        // Acquire the single tile and copy data (converting to bool_t)
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

//! Bind data from vector
template<typename T>
void CompiledGraph::bind_data(const std::string& name,
                              const std::vector<T>& data)
{
    bind_data(name, data.data(), data.size());
}

//! Get output data (copies data out)
template<typename T>
std::vector<T> CompiledGraph::get_output(const std::string& name)
{
    DataType dtype = tensor_dtypes_[name];
    std::vector<T> result;

    if(dtype == DataType::FP32)
    {
        auto& tensor = get_tensor<nntile::fp32_t>(name);
        result.resize(tensor.nelems);

        // Acquire the single tile and copy data out (converting from fp32_t)
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
        auto& tensor = get_tensor<nntile::fp32_fast_tf32_t>(name);
        result.resize(tensor.nelems);

        // Acquire the single tile and copy data out
        // (converting from fp32_fast_tf32_t)
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
        auto& tensor = get_tensor<nntile::fp32_fast_fp16_t>(name);
        result.resize(tensor.nelems);

        // Acquire the single tile and copy data out
        // (converting from fp32_fast_fp16_t)
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
        auto& tensor = get_tensor<nntile::fp32_fast_bf16_t>(name);
        result.resize(tensor.nelems);

        // Acquire the single tile and copy data out
        // (converting from fp32_fast_bf16_t)
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
        auto& tensor = get_tensor<nntile::fp64_t>(name);
        result.resize(tensor.nelems);

        // Acquire the single tile and copy data out (converting from fp64_t)
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
        auto& tensor = get_tensor<nntile::fp16_t>(name);
        result.resize(tensor.nelems);

        // Acquire the single tile and copy data out (converting from fp16_t)
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
        auto& tensor = get_tensor<nntile::bf16_t>(name);
        result.resize(tensor.nelems);

        // Acquire the single tile and copy data out (converting from bf16_t)
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
        auto& tensor = get_tensor<nntile::int64_t>(name);
        result.resize(tensor.nelems);

        // Acquire the single tile and copy data out (converting from int64_t)
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        for(Index i = 0; i < tensor.nelems; ++i)
        {
            result[i] = static_cast<T>(static_cast<long long>(tile_local[i]));
        }
        tile_local.release();
    }
    else if(dtype == DataType::INT32)
    {
        throw std::runtime_error(
            "INT32 data type not supported for output retrieval");
    }
    else if(dtype == DataType::BOOL)
    {
        auto& tensor = get_tensor<nntile::bool_t>(name);
        result.resize(tensor.nelems);

        // Acquire the single tile and copy data out (converting from bool_t)
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
        throw std::runtime_error("Unsupported data type for output retrieval");
    }

    return result;
}

// Template instantiations
template nntile::tensor::Tensor<nntile::fp32_t>&
CompiledGraph::get_tensor<nntile::fp32_t>(const std::string& name);

template nntile::tensor::Tensor<nntile::fp32_fast_tf32_t>&
CompiledGraph::get_tensor<nntile::fp32_fast_tf32_t>(const std::string& name);

template nntile::tensor::Tensor<nntile::fp32_fast_fp16_t>&
CompiledGraph::get_tensor<nntile::fp32_fast_fp16_t>(const std::string& name);

template nntile::tensor::Tensor<nntile::fp32_fast_bf16_t>&
CompiledGraph::get_tensor<nntile::fp32_fast_bf16_t>(const std::string& name);

template nntile::tensor::Tensor<nntile::fp64_t>&
CompiledGraph::get_tensor<nntile::fp64_t>(const std::string& name);

template nntile::tensor::Tensor<nntile::fp16_t>&
CompiledGraph::get_tensor<nntile::fp16_t>(const std::string& name);

template nntile::tensor::Tensor<nntile::bf16_t>&
CompiledGraph::get_tensor<nntile::bf16_t>(const std::string& name);

template nntile::tensor::Tensor<nntile::int64_t>&
CompiledGraph::get_tensor<nntile::int64_t>(const std::string& name);

template nntile::tensor::Tensor<nntile::bool_t>&
CompiledGraph::get_tensor<nntile::bool_t>(const std::string& name);

template void CompiledGraph::bind_data<float>(
    const std::string& name, const float* data, size_t count);

template void CompiledGraph::bind_data<double>(
    const std::string& name, const double* data, size_t count);

template void CompiledGraph::bind_data<long long>(
    const std::string& name, const long long* data, size_t count);

template void CompiledGraph::bind_data<float>(
    const std::string& name, const std::vector<float>& data);

template void CompiledGraph::bind_data<double>(
    const std::string& name, const std::vector<double>& data);

template void CompiledGraph::bind_data<long long>(
    const std::string& name, const std::vector<long long>& data);

template std::vector<float>
CompiledGraph::get_output<float>(const std::string& name);

template std::vector<double>
CompiledGraph::get_output<double>(const std::string& name);

} // namespace nntile::graph
