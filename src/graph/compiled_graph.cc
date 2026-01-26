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
#include "nntile/graph/compiled_graph_ops.hh"
#include "nntile/tensor/tensor.hh"

namespace nntile::graph
{

//! Validate that operation data types are supported
void validate_operation_data_types(const LogicalGraph& logical)
{
    for(const auto& op : logical.ops())
    {
        // Get the data type from the first input tensor (or output if no inputs)
        DataType dtype;
        if(!op->inputs().empty())
        {
            dtype = op->inputs()[0]->dtype();
        }
        else if(!op->outputs().empty())
        {
            dtype = op->outputs()[0]->dtype();
        }
        else
        {
            throw std::runtime_error("Operation has no inputs or outputs");
        }

        // Check supported data types based on operation type
        if(op->type() == OpType::GELU || op->type() == OpType::GELU_BACKWARD ||
           op->type() == OpType::GELU_INPLACE || op->type() == OpType::GELUTANH ||
           op->type() == OpType::GELUTANH_INPLACE || op->type() == OpType::GELUTANH_BACKWARD ||
           op->type() == OpType::RELU || op->type() == OpType::RELU_INPLACE ||
           op->type() == OpType::RELU_BACKWARD || op->type() == OpType::SILU ||
           op->type() == OpType::SILU_INPLACE || op->type() == OpType::SILU_BACKWARD ||
           op->type() == OpType::SQRT || op->type() == OpType::SQRT_INPLACE ||
           op->type() == OpType::POW || op->type() == OpType::POW_INPLACE ||
           op->type() == OpType::ADD || op->type() == OpType::ADD_INPLACE ||
           op->type() == OpType::MULTIPLY || op->type() == OpType::MULTIPLY_INPLACE ||
           op->type() == OpType::SUM || op->type() == OpType::SUM_FIBER ||
           op->type() == OpType::SCALE || op->type() == OpType::SCALE_INPLACE ||
           op->type() == OpType::GEMM)
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
        else if(op->type() == OpType::CLEAR)
        {
            if(dtype == DataType::INT32)
            {
                throw std::runtime_error(
                    "CLEAR operation does not support data type " +
                    dtype_to_string(dtype));
            }
        }
        // Special case for embedding: index tensor should be INT64, others floating-point
        else if(op->type() == OpType::EMBEDDING || op->type() == OpType::EMBEDDING_BACKWARD)
        {
            // For embedding operations, check the index tensor (first input) is INT64
            if(!op->inputs().empty() && op->inputs()[0]->dtype() != DataType::INT64)
            {
                throw std::runtime_error(
                    std::string(op_type_to_string(op->type())) +
                    " operation requires INT64 index tensor, got " +
                    dtype_to_string(op->inputs()[0]->dtype()));
            }
            // Check that vocab/embed tensors are floating-point
            for(size_t i = 1; i < op->inputs().size(); ++i)
            {
                DataType input_dtype = op->inputs()[i]->dtype();
                if(input_dtype != DataType::FP32 &&
                   input_dtype != DataType::FP32_FAST_TF32 &&
                   input_dtype != DataType::FP32_FAST_FP16 &&
                   input_dtype != DataType::FP32_FAST_BF16 &&
                   input_dtype != DataType::FP64 &&
                   input_dtype != DataType::FP16 &&
                   input_dtype != DataType::BF16)
                {
                    throw std::runtime_error(
                        std::string(op_type_to_string(op->type())) +
                        " operation requires floating-point tensors, got " +
                        dtype_to_string(input_dtype) + " for input " + std::to_string(i));
                }
            }
            for(size_t i = 0; i < op->outputs().size(); ++i)
            {
                DataType output_dtype = op->outputs()[i]->dtype();
                if(output_dtype != DataType::FP32 &&
                   output_dtype != DataType::FP32_FAST_TF32 &&
                   output_dtype != DataType::FP32_FAST_FP16 &&
                   output_dtype != DataType::FP32_FAST_BF16 &&
                   output_dtype != DataType::FP64 &&
                   output_dtype != DataType::FP16 &&
                   output_dtype != DataType::BF16)
                {
                    throw std::runtime_error(
                        std::string(op_type_to_string(op->type())) +
                        " operation requires floating-point tensors, got " +
                        dtype_to_string(output_dtype) + " for output " + std::to_string(i));
                }
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
        DataType dtype = node->dtype();
        tensor_dtypes_[node->name()] = dtype;

        // Create tensor with single tile (no tiling)
        std::vector<Index> shape = node->shape();
        std::vector<Index> tile_shape = shape;  // Same as shape = 1 tile

        switch(dtype)
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
        case OpType::TRANSPOSE:
            execute_transpose(*this, op_info);
            break;
        case OpType::GELU:
            execute_gelu(*this, op_info);
            break;
        case OpType::GELU_BACKWARD:
            execute_gelu_backward(*this, op_info);
            break;
        case OpType::CLEAR:
            execute_clear(*this, op_info);
            break;
        case OpType::FILL:
            execute_fill(*this, op_info);
            break;
        case OpType::COPY:
            execute_copy(*this, op_info);
            break;

        // Element-wise unary operations
        case OpType::GELU_INPLACE:
            execute_gelu_inplace(*this, op_info);
            break;
        case OpType::GELUTANH:
            execute_gelutanh(*this, op_info);
            break;
        case OpType::GELUTANH_INPLACE:
            execute_gelutanh_inplace(*this, op_info);
            break;
        case OpType::GELUTANH_BACKWARD:
            execute_gelutanh_backward(*this, op_info);
            break;
        case OpType::RELU:
            execute_relu(*this, op_info);
            break;
        case OpType::RELU_INPLACE:
            execute_relu_inplace(*this, op_info);
            break;
        case OpType::RELU_BACKWARD:
            execute_relu_backward(*this, op_info);
            break;
        case OpType::SILU:
            execute_silu(*this, op_info);
            break;
        case OpType::SILU_INPLACE:
            execute_silu_inplace(*this, op_info);
            break;
        case OpType::SILU_BACKWARD:
            execute_silu_backward(*this, op_info);
            break;
        case OpType::SOFTMAX:
            execute_softmax(*this, op_info);
            break;
        case OpType::SOFTMAX_INPLACE:
            execute_softmax_inplace(*this, op_info);
            break;
        case OpType::SQRT:
            execute_sqrt(*this, op_info);
            break;
        case OpType::SQRT_INPLACE:
            execute_sqrt_inplace(*this, op_info);
            break;
        case OpType::POW:
            execute_pow(*this, op_info);
            break;
        case OpType::POW_INPLACE:
            execute_pow_inplace(*this, op_info);
            break;

        // Binary operations
        case OpType::ADD:
            execute_add(*this, op_info);
            break;
        case OpType::ADD_INPLACE:
            execute_add_inplace(*this, op_info);
            break;
        case OpType::ADD_FIBER:
            execute_add_fiber(*this, op_info);
            break;
        case OpType::ADD_FIBER_INPLACE:
            execute_add_fiber_inplace(*this, op_info);
            break;
        case OpType::ADD_SLICE:
            execute_add_slice(*this, op_info);
            break;
        case OpType::ADD_SLICE_INPLACE:
            execute_add_slice_inplace(*this, op_info);
            break;
        case OpType::MULTIPLY:
            execute_multiply(*this, op_info);
            break;
        case OpType::MULTIPLY_INPLACE:
            execute_multiply_inplace(*this, op_info);
            break;
        case OpType::MULTIPLY_FIBER:
            execute_multiply_fiber(*this, op_info);
            break;
        case OpType::MULTIPLY_FIBER_INPLACE:
            execute_multiply_fiber_inplace(*this, op_info);
            break;
        case OpType::MULTIPLY_SLICE:
            execute_multiply_slice(*this, op_info);
            break;
        case OpType::HYPOT:
            execute_hypot(*this, op_info);
            break;
        case OpType::HYPOT_INPLACE:
            execute_hypot_inplace(*this, op_info);
            break;

        // Reduction operations
        case OpType::SUM:
            execute_sum(*this, op_info);
            break;
        case OpType::SUM_FIBER:
            execute_sum_fiber(*this, op_info);
            break;

        // Scale operations
        case OpType::SCALE:
            execute_scale(*this, op_info);
            break;
        case OpType::SCALE_INPLACE:
            execute_scale_inplace(*this, op_info);
            break;
        case OpType::SCALE_FIBER:
            execute_scale_fiber(*this, op_info);
            break;
        case OpType::SCALE_SLICE:
            execute_scale_slice(*this, op_info);
            break;

        // Embedding operations
        case OpType::EMBEDDING:
            execute_embedding(*this, op_info);
            break;
        case OpType::EMBEDDING_BACKWARD:
            execute_embedding_backward(*this, op_info);
            break;

        // Mixed-dtype operations
        case OpType::TOTAL_SUM_ACCUM:
            execute_total_sum_accum(*this, op_info);
            break;

        // Optimizer operations
        case OpType::SGD_STEP:
            execute_sgd_step(*this, op_info);
            break;
        case OpType::ADAM_STEP:
            execute_adam_step(*this, op_info);
            break;
        case OpType::ADAMW_STEP:
            execute_adamw_step(*this, op_info);
            break;

        // RoPE operations
        case OpType::ROPE:
            execute_rope(*this, op_info);
            break;
        case OpType::ROPE_BACKWARD:
            execute_rope_backward(*this, op_info);
            break;

        // Flash attention operations
        case OpType::FLASH_SDPA_FWD_CUDNN:
            execute_flash_sdpa_fwd_cudnn(*this, op_info);
            break;

        default:
            throw std::runtime_error(
                "Unsupported operation type in execute_op: " +
                std::to_string(static_cast<int>(op_info.type)));
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
    auto tensor_it = tensors_.find(name);
    if(tensor_it == tensors_.end())
    {
        throw std::runtime_error("Tensor not found: " + name);
    }
    auto dtype_it = tensor_dtypes_.find(name);
    if(dtype_it == tensor_dtypes_.end())
    {
        throw std::runtime_error("Tensor dtype not found: " + name);
    }
    DataType dtype = dtype_it->second;
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
            result[i] = static_cast<T>(static_cast<std::int64_t>(tile_local[i]));
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

// Use char instead of bool, as std::vector<bool> does not have data member
template void CompiledGraph::bind_data<char>(
    const std::string& name, const char* data, size_t count);

template void CompiledGraph::bind_data<float>(
    const std::string& name, const std::vector<float>& data);

template void CompiledGraph::bind_data<double>(
    const std::string& name, const std::vector<double>& data);

template void CompiledGraph::bind_data<long long>(
    const std::string& name, const std::vector<long long>& data);

// Use char instead of bool, as std::vector<bool> does not have data member
template void CompiledGraph::bind_data<char>(
    const std::string& name, const std::vector<char>& data);

template std::vector<char>
CompiledGraph::get_output<char>(const std::string& name);

template std::vector<float>
CompiledGraph::get_output<float>(const std::string& name);

template std::vector<double>
CompiledGraph::get_output<double>(const std::string& name);

} // namespace nntile::graph
