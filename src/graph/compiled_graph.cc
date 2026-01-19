/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled_graph.cc
 * Implementation of CompiledGraph class.
 *
 * @version 1.1.0
 * */

#include <nntile/graph/compiled_graph.hh>
#include <nntile/tensor/gemm.hh>
#include <nntile/tensor/gelu.hh>
#include <nntile/tensor/tensor.hh>
#include <nntile/constants.hh>
#include <nntile/base_types.hh>
#include <starpu.h>  // For STARPU_W, STARPU_R
#include <stdexcept>
#include <memory>
#include <algorithm>

namespace nntile::graph
{

//! Compile a logical graph
CompiledGraph CompiledGraph::compile(const LogicalGraph& logical)
{
    CompiledGraph cg;
    cg.allocate_tensors(logical);
    cg.compute_execution_order(logical);
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
                auto t = std::make_shared<nntile::tensor::Tensor<nntile::fp32_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::FP32_FAST_TF32:
            {
                auto t = std::make_shared<nntile::tensor::Tensor<nntile::fp32_fast_tf32_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::FP32_FAST_FP16:
            {
                auto t = std::make_shared<nntile::tensor::Tensor<nntile::fp32_fast_fp16_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::FP32_FAST_BF16:
            {
                auto t = std::make_shared<nntile::tensor::Tensor<nntile::fp32_fast_bf16_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::FP64:
            {
                auto t = std::make_shared<nntile::tensor::Tensor<nntile::fp64_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::FP16:
            {
                auto t = std::make_shared<nntile::tensor::Tensor<nntile::fp16_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::BF16:
            {
                auto t = std::make_shared<nntile::tensor::Tensor<nntile::bf16_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::INT64:
            {
                auto t = std::make_shared<nntile::tensor::Tensor<nntile::int64_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::INT32:
            {
                // INT32 maps to int32_t, but NNTile doesn't have a wrapper for it
                // For now, throw an error as it's not commonly used
                throw std::runtime_error("INT32 data type not yet supported");
            }
            case DataType::BOOL:
            {
                auto t = std::make_shared<nntile::tensor::Tensor<nntile::bool_t>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            default:
                throw std::runtime_error("Unsupported data type for tensor allocation");
        }
    }
}

//! Compute topological order of operations
void CompiledGraph::compute_execution_order(const LogicalGraph& logical)
{
    execution_order_.clear();
    std::set<NodeId> executed_tensors;

    // Mark all input tensors (no producer) as executed
    for(const auto& t : logical.tensors())
    {
        if(!t->has_producer())
        {
            executed_tensors.insert(t->id());
        }
    }

    // Keep adding ops whose inputs are all ready
    std::set<NodeId> executed_ops;
    while(execution_order_.size() < logical.num_ops())
    {
        bool added = false;
        for(const auto& op : logical.ops())
        {
            if(executed_ops.count(op->id()))
            {
                continue;
            }

            // Check if all inputs are ready
            bool ready = true;
            for(const auto* input : op->inputs())
            {
                if(!executed_tensors.count(input->id()))
                {
                    ready = false;
                    break;
                }
            }

            if(ready)
            {
                // Extract execution info instead of storing raw pointer
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
                execution_order_.push_back(op_info);
                executed_ops.insert(op->id());
                for(const auto* output : op->outputs())
                {
                    executed_tensors.insert(output->id());
                }
                added = true;
            }
        }
        if(!added)
        {
            throw std::runtime_error("Graph contains cycles or invalid dependencies");
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
            execute_gemm(op_info);
            break;
        case OpType::GELU:
            execute_gelu(op_info);
            break;
    }
}

//! Execute gemm operation
void CompiledGraph::execute_gemm(const OpExecutionInfo& op_info)
{
    const auto& attrs = std::get<GemmAttrs>(op_info.attrs);

    const std::string& a_name = op_info.input_names[0];
    const std::string& b_name = op_info.input_names[1];
    const std::string& c_name = op_info.output_names[0];

    DataType dtype = tensor_dtypes_[a_name];

    if(dtype == DataType::FP32)
    {
        auto& a = get_tensor<nntile::fp32_t>(a_name);
        auto& b = get_tensor<nntile::fp32_t>(b_name);
        auto& c = get_tensor<nntile::fp32_t>(c_name);

        // Use nntile::tensor::gemm
        nntile::tensor::gemm<nntile::fp32_t>(
            static_cast<nntile::Scalar>(attrs.alpha),
            attrs.trans_a ? nntile::TransOp(nntile::TransOp::Trans) :
                    nntile::TransOp(nntile::TransOp::NoTrans),
            a,
            attrs.trans_b ? nntile::TransOp(nntile::TransOp::Trans) :
                    nntile::TransOp(nntile::TransOp::NoTrans),
            b,
            static_cast<nntile::Scalar>(attrs.beta),
            c,
            attrs.ndim,
            attrs.batch_ndim,
            0  // redux = 0
        );
    }
    else if(dtype == DataType::FP32_FAST_TF32)
    {
        auto& a = get_tensor<nntile::fp32_fast_tf32_t>(a_name);
        auto& b = get_tensor<nntile::fp32_fast_tf32_t>(b_name);
        auto& c = get_tensor<nntile::fp32_fast_tf32_t>(c_name);

        nntile::tensor::gemm<nntile::fp32_fast_tf32_t>(
            static_cast<nntile::Scalar>(attrs.alpha),
            attrs.trans_a ? nntile::TransOp(nntile::TransOp::Trans) :
                    nntile::TransOp(nntile::TransOp::NoTrans),
            a,
            attrs.trans_b ? nntile::TransOp(nntile::TransOp::Trans) :
                    nntile::TransOp(nntile::TransOp::NoTrans),
            b,
            static_cast<nntile::Scalar>(attrs.beta),
            c,
            attrs.ndim,
            attrs.batch_ndim,
            0  // redux = 0
        );
    }
    else if(dtype == DataType::FP32_FAST_FP16)
    {
        auto& a = get_tensor<nntile::fp32_fast_fp16_t>(a_name);
        auto& b = get_tensor<nntile::fp32_fast_fp16_t>(b_name);
        auto& c = get_tensor<nntile::fp32_fast_fp16_t>(c_name);

        nntile::tensor::gemm<nntile::fp32_fast_fp16_t>(
            static_cast<nntile::Scalar>(attrs.alpha),
            attrs.trans_a ? nntile::TransOp(nntile::TransOp::Trans) :
                    nntile::TransOp(nntile::TransOp::NoTrans),
            a,
            attrs.trans_b ? nntile::TransOp(nntile::TransOp::Trans) :
                    nntile::TransOp(nntile::TransOp::NoTrans),
            b,
            static_cast<nntile::Scalar>(attrs.beta),
            c,
            attrs.ndim,
            attrs.batch_ndim,
            0  // redux = 0
        );
    }
    else if(dtype == DataType::FP32_FAST_BF16)
    {
        auto& a = get_tensor<nntile::fp32_fast_bf16_t>(a_name);
        auto& b = get_tensor<nntile::fp32_fast_bf16_t>(b_name);
        auto& c = get_tensor<nntile::fp32_fast_bf16_t>(c_name);

        nntile::tensor::gemm<nntile::fp32_fast_bf16_t>(
            static_cast<nntile::Scalar>(attrs.alpha),
            attrs.trans_a ? nntile::TransOp(nntile::TransOp::Trans) :
                    nntile::TransOp(nntile::TransOp::NoTrans),
            a,
            attrs.trans_b ? nntile::TransOp(nntile::TransOp::Trans) :
                    nntile::TransOp(nntile::TransOp::NoTrans),
            b,
            static_cast<nntile::Scalar>(attrs.beta),
            c,
            attrs.ndim,
            attrs.batch_ndim,
            0  // redux = 0
        );
    }
    else if(dtype == DataType::FP64)
    {
        auto& a = get_tensor<nntile::fp64_t>(a_name);
        auto& b = get_tensor<nntile::fp64_t>(b_name);
        auto& c = get_tensor<nntile::fp64_t>(c_name);

        nntile::tensor::gemm<nntile::fp64_t>(
            static_cast<nntile::Scalar>(attrs.alpha),
            attrs.trans_a ? nntile::TransOp(nntile::TransOp::Trans) :
                    nntile::TransOp(nntile::TransOp::NoTrans),
            a,
            attrs.trans_b ? nntile::TransOp(nntile::TransOp::Trans) :
                    nntile::TransOp(nntile::TransOp::NoTrans),
            b,
            static_cast<nntile::Scalar>(attrs.beta),
            c,
            attrs.ndim,
            attrs.batch_ndim,
            0  // redux = 0
        );
    }
    else if(dtype == DataType::FP16)
    {
        auto& a = get_tensor<nntile::fp16_t>(a_name);
        auto& b = get_tensor<nntile::fp16_t>(b_name);
        auto& c = get_tensor<nntile::fp16_t>(c_name);

        nntile::tensor::gemm<nntile::fp16_t>(
            static_cast<nntile::Scalar>(attrs.alpha),
            attrs.trans_a ? nntile::TransOp(nntile::TransOp::Trans) :
                    nntile::TransOp(nntile::TransOp::NoTrans),
            a,
            attrs.trans_b ? nntile::TransOp(nntile::TransOp::Trans) :
                    nntile::TransOp(nntile::TransOp::NoTrans),
            b,
            static_cast<nntile::Scalar>(attrs.beta),
            c,
            attrs.ndim,
            attrs.batch_ndim,
            0  // redux = 0
        );
    }
    else if(dtype == DataType::BF16)
    {
        auto& a = get_tensor<nntile::bf16_t>(a_name);
        auto& b = get_tensor<nntile::bf16_t>(b_name);
        auto& c = get_tensor<nntile::bf16_t>(c_name);

        nntile::tensor::gemm<nntile::bf16_t>(
            static_cast<nntile::Scalar>(attrs.alpha),
            attrs.trans_a ? nntile::TransOp(nntile::TransOp::Trans) :
                    nntile::TransOp(nntile::TransOp::NoTrans),
            a,
            attrs.trans_b ? nntile::TransOp(nntile::TransOp::Trans) :
                    nntile::TransOp(nntile::TransOp::NoTrans),
            b,
            static_cast<nntile::Scalar>(attrs.beta),
            c,
            attrs.ndim,
            attrs.batch_ndim,
            0  // redux = 0
        );
    }
    else if(dtype == DataType::INT64)
    {
        throw std::runtime_error("INT64 data type not supported for gemm operation");
    }
    else if(dtype == DataType::INT32)
    {
        throw std::runtime_error("INT32 data type not supported for gemm operation");
    }
    else if(dtype == DataType::BOOL)
    {
        throw std::runtime_error("BOOL data type not supported for gemm operation");
    }
    else
    {
        throw std::runtime_error("Unsupported data type for gemm");
    }
}

//! Execute gelu operation
void CompiledGraph::execute_gelu(const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];

    DataType dtype = tensor_dtypes_[x_name];

    if(dtype == DataType::FP32)
    {
        auto& x = get_tensor<nntile::fp32_t>(x_name);
        auto& y = get_tensor<nntile::fp32_t>(y_name);

        // Use nntile::tensor::gelu
        nntile::tensor::gelu<nntile::fp32_t>(x, y);
    }
    else if(dtype == DataType::FP32_FAST_TF32)
    {
        auto& x = get_tensor<nntile::fp32_fast_tf32_t>(x_name);
        auto& y = get_tensor<nntile::fp32_fast_tf32_t>(y_name);

        nntile::tensor::gelu<nntile::fp32_fast_tf32_t>(x, y);
    }
    else if(dtype == DataType::FP32_FAST_FP16)
    {
        auto& x = get_tensor<nntile::fp32_fast_fp16_t>(x_name);
        auto& y = get_tensor<nntile::fp32_fast_fp16_t>(y_name);

        nntile::tensor::gelu<nntile::fp32_fast_fp16_t>(x, y);
    }
    else if(dtype == DataType::FP32_FAST_BF16)
    {
        auto& x = get_tensor<nntile::fp32_fast_bf16_t>(x_name);
        auto& y = get_tensor<nntile::fp32_fast_bf16_t>(y_name);

        nntile::tensor::gelu<nntile::fp32_fast_bf16_t>(x, y);
    }
    else if(dtype == DataType::FP64)
    {
        auto& x = get_tensor<nntile::fp64_t>(x_name);
        auto& y = get_tensor<nntile::fp64_t>(y_name);

        nntile::tensor::gelu<nntile::fp64_t>(x, y);
    }
    else if(dtype == DataType::FP16)
    {
        auto& x = get_tensor<nntile::fp16_t>(x_name);
        auto& y = get_tensor<nntile::fp16_t>(y_name);

        nntile::tensor::gelu<nntile::fp16_t>(x, y);
    }
    else if(dtype == DataType::BF16)
    {
        auto& x = get_tensor<nntile::bf16_t>(x_name);
        auto& y = get_tensor<nntile::bf16_t>(y_name);

        nntile::tensor::gelu<nntile::bf16_t>(x, y);
    }
    else if(dtype == DataType::INT64)
    {
        throw std::runtime_error("INT64 data type not supported for gelu operation");
    }
    else if(dtype == DataType::INT32)
    {
        throw std::runtime_error("INT32 data type not supported for gelu operation");
    }
    else if(dtype == DataType::BOOL)
    {
        throw std::runtime_error("BOOL data type not supported for gelu operation");
    }
    else
    {
        throw std::runtime_error("Unsupported data type for gelu");
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
void CompiledGraph::bind_data(const std::string& name, const T* data, size_t count)
{
    auto it = tensors_.find(name);
    if(it == tensors_.end())
    {
        throw std::runtime_error("Tensor not found: " + name);
    }

    DataType dtype = tensor_dtypes_[name];

    // Check count matches tensor size and convert data to appropriate wrapper type
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

        // Acquire the single tile and copy data (converting to fp32_fast_tf32_t)
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] = nntile::fp32_fast_tf32_t(static_cast<float>(data[i]));
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

        // Acquire the single tile and copy data (converting to fp32_fast_fp16_t)
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] = nntile::fp32_fast_fp16_t(static_cast<float>(data[i]));
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

        // Acquire the single tile and copy data (converting to fp32_fast_bf16_t)
        auto tile = tensor.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(size_t i = 0; i < count; ++i)
        {
            tile_local[i] = nntile::fp32_fast_bf16_t(static_cast<float>(data[i]));
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
void CompiledGraph::bind_data(const std::string& name, const std::vector<T>& data)
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

        // Acquire the single tile and copy data out (converting from fp32_fast_tf32_t)
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

        // Acquire the single tile and copy data out (converting from fp32_fast_fp16_t)
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

        // Acquire the single tile and copy data out (converting from fp32_fast_bf16_t)
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

//! Get raw pointer to output (must call wait() first)
template<typename T>
const T* CompiledGraph::get_output_ptr(const std::string& name)
{
    // This is a simplified implementation - in a real implementation,
    // we'd need to manage the tile acquisition/release properly
    throw std::runtime_error("get_output_ptr not implemented in minimal version");
}

// Template instantiations
template nntile::tensor::Tensor<nntile::fp32_t>& CompiledGraph::get_tensor<nntile::fp32_t>(
        const std::string& name);
template nntile::tensor::Tensor<nntile::fp32_fast_tf32_t>& CompiledGraph::get_tensor<nntile::fp32_fast_tf32_t>(
        const std::string& name);
template nntile::tensor::Tensor<nntile::fp32_fast_fp16_t>& CompiledGraph::get_tensor<nntile::fp32_fast_fp16_t>(
        const std::string& name);
template nntile::tensor::Tensor<nntile::fp32_fast_bf16_t>& CompiledGraph::get_tensor<nntile::fp32_fast_bf16_t>(
        const std::string& name);
template nntile::tensor::Tensor<nntile::fp64_t>& CompiledGraph::get_tensor<nntile::fp64_t>(
        const std::string& name);
template nntile::tensor::Tensor<nntile::fp16_t>& CompiledGraph::get_tensor<nntile::fp16_t>(
        const std::string& name);
template nntile::tensor::Tensor<nntile::bf16_t>& CompiledGraph::get_tensor<nntile::bf16_t>(
        const std::string& name);
template nntile::tensor::Tensor<nntile::int64_t>& CompiledGraph::get_tensor<nntile::int64_t>(
        const std::string& name);
template nntile::tensor::Tensor<nntile::bool_t>& CompiledGraph::get_tensor<nntile::bool_t>(
        const std::string& name);

template void CompiledGraph::bind_data<float>(const std::string& name, const float* data,
        size_t count);
template void CompiledGraph::bind_data<double>(const std::string& name, const double* data,
        size_t count);
template void CompiledGraph::bind_data<long long>(const std::string& name,
        const long long* data, size_t count);
template void CompiledGraph::bind_data<float>(const std::string& name,
        const std::vector<float>& data);
template void CompiledGraph::bind_data<double>(const std::string& name,
        const std::vector<double>& data);
template void CompiledGraph::bind_data<long long>(const std::string& name,
        const std::vector<long long>& data);

template std::vector<float> CompiledGraph::get_output<float>(const std::string& name);
template std::vector<double> CompiledGraph::get_output<double>(const std::string& name);

template const float* CompiledGraph::get_output_ptr<float>(const std::string& name);
template const double* CompiledGraph::get_output_ptr<double>(const std::string& name);

} // namespace nntile::graph
