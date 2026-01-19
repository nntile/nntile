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
#include <starpu.h>  // For STARPU_W, STARPU_R
#include <stdexcept>
#include <memory>
#include <algorithm>

namespace nntile::graph {

//! Compile a logical graph
CompiledGraph CompiledGraph::compile(const LogicalGraph& logical) {
    CompiledGraph cg;
    cg.logical_ = &logical;
    cg.allocate_tensors();
    cg.compute_execution_order();
    return cg;
}

//! Allocate NNTile tensors for all graph tensors
void CompiledGraph::allocate_tensors() {
    for (const auto& node : logical_->tensors()) {
        const auto& spec = node->spec();
        tensor_dtypes_[node->name()] = spec.dtype();

        // Create tensor with single tile (no tiling)
        std::vector<Index> shape = spec.shape();
        std::vector<Index> tile_shape = shape;  // Same as shape = 1 tile

        switch (spec.dtype()) {
            case DataType::FP32: {
                auto t = std::make_shared<nntile::tensor::Tensor<float>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            case DataType::FP64: {
                auto t = std::make_shared<nntile::tensor::Tensor<double>>(
                    nntile::tensor::TensorTraits(shape, tile_shape)
                );
                tensors_[node->name()] = t;
                break;
            }
            default:
                throw std::runtime_error("Unsupported data type");
        }
    }
}

//! Compute topological order of operations
void CompiledGraph::compute_execution_order() {
    execution_order_.clear();
    std::set<NodeId> executed_tensors;

    // Mark all input tensors (no producer) as executed
    for (const auto& t : logical_->tensors()) {
        if (!t->has_producer()) {
            executed_tensors.insert(t->id());
        }
    }

    // Keep adding ops whose inputs are all ready
    std::set<NodeId> executed_ops;
    while (execution_order_.size() < logical_->num_ops()) {
        bool added = false;
        for (const auto& op : logical_->ops()) {
            if (executed_ops.count(op->id())) continue;

            // Check if all inputs are ready
            bool ready = true;
            for (const auto* input : op->inputs()) {
                if (!executed_tensors.count(input->id())) {
                    ready = false;
                    break;
                }
            }

            if (ready) {
                execution_order_.push_back(op.get());
                executed_ops.insert(op->id());
                for (const auto* output : op->outputs()) {
                    executed_tensors.insert(output->id());
                }
                added = true;
            }
        }
        if (!added) {
            throw std::runtime_error("Graph contains cycles or invalid dependencies");
        }
    }
}

//! Execute the graph
void CompiledGraph::execute() {
    for (const OpNode* op : execution_order_) {
        execute_op(op);
    }
}

//! Wait for all operations to complete
void CompiledGraph::wait() {
    // In single-threaded StarPU, operations are synchronous
    // For now, do nothing
}

//! Execute a single operation
void CompiledGraph::execute_op(const OpNode* op) {
    switch (op->type()) {
        case OpType::MATMUL:
            execute_matmul(op);
            break;
        case OpType::GELU:
            execute_gelu(op);
            break;
    }
}

//! Execute matmul operation
void CompiledGraph::execute_matmul(const OpNode* op) {
    const auto& attrs = std::get<MatmulAttrs>(op->attrs());

    const std::string& a_name = op->input(0)->name();
    const std::string& b_name = op->input(1)->name();
    const std::string& c_name = op->output(0)->name();

    DataType dtype = tensor_dtypes_[a_name];

    if (dtype == DataType::FP32) {
        auto& a = get_tensor<float>(a_name);
        auto& b = get_tensor<float>(b_name);
        auto& c = get_tensor<float>(c_name);

        // Use nntile::tensor::gemm
        nntile::tensor::gemm<float>(
            static_cast<float>(attrs.alpha),
            attrs.trans_a ? nntile::TransOp(nntile::TransOp::Trans) : nntile::TransOp(nntile::TransOp::NoTrans),
            a,
            attrs.trans_b ? nntile::TransOp(nntile::TransOp::Trans) : nntile::TransOp(nntile::TransOp::NoTrans),
            b,
            static_cast<float>(attrs.beta),
            c,
            2,  // ndim = 2 for matrices
            0   // batch_ndim = 0
        );
    } else if (dtype == DataType::FP64) {
        auto& a = get_tensor<double>(a_name);
        auto& b = get_tensor<double>(b_name);
        auto& c = get_tensor<double>(c_name);

        nntile::tensor::gemm<double>(
            static_cast<double>(attrs.alpha),
            attrs.trans_a ? nntile::TransOp(nntile::TransOp::Trans) : nntile::TransOp(nntile::TransOp::NoTrans),
            a,
            attrs.trans_b ? nntile::TransOp(nntile::TransOp::Trans) : nntile::TransOp(nntile::TransOp::NoTrans),
            b,
            static_cast<double>(attrs.beta),
            c,
            2,  // ndim = 2 for matrices
            0   // batch_ndim = 0
        );
    } else {
        throw std::runtime_error("Unsupported data type for matmul");
    }
}

//! Execute gelu operation
void CompiledGraph::execute_gelu(const OpNode* op) {
    const std::string& x_name = op->input(0)->name();
    const std::string& y_name = op->output(0)->name();

    DataType dtype = tensor_dtypes_[x_name];

    if (dtype == DataType::FP32) {
        auto& x = get_tensor<float>(x_name);
        auto& y = get_tensor<float>(y_name);

        // Use nntile::tensor::gelu
        nntile::tensor::gelu<float>(x, y);
    } else if (dtype == DataType::FP64) {
        auto& x = get_tensor<double>(x_name);
        auto& y = get_tensor<double>(y_name);

        nntile::tensor::gelu<double>(x, y);
    } else {
        throw std::runtime_error("Unsupported data type for gelu");
    }
}

//! Get typed tensor pointer
template<typename T>
nntile::tensor::Tensor<T>& CompiledGraph::get_tensor(const std::string& name) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return *static_cast<nntile::tensor::Tensor<T>*>(it->second.get());
}

//! Bind data to a tensor (copies data)
template<typename T>
void CompiledGraph::bind_data(const std::string& name, const T* data, size_t count) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }

    auto& tensor = get_tensor<T>(name);

    // Check count matches tensor size
    if (count != static_cast<size_t>(tensor.nelems)) {
        throw std::runtime_error("Data size mismatch for tensor " + name);
    }

    // Acquire the single tile and copy data
    auto tile = tensor.get_tile(0);
    auto tile_local = tile.acquire(STARPU_W);
    std::copy(data, data + count, tile_local.get_ptr());
    tile_local.release();
}

//! Bind data from vector
template<typename T>
void CompiledGraph::bind_data(const std::string& name, const std::vector<T>& data) {
    bind_data(name, data.data(), data.size());
}

//! Get output data (copies data out)
template<typename T>
std::vector<T> CompiledGraph::get_output(const std::string& name) {
    auto& tensor = get_tensor<T>(name);
    std::vector<T> result(tensor.nelems);

    // Acquire the single tile and copy data out
    auto tile = tensor.get_tile(0);
    auto tile_local = tile.acquire(STARPU_R);
    std::copy(tile_local.get_ptr(), tile_local.get_ptr() + tensor.nelems, result.begin());
    tile_local.release();

    return result;
}

//! Get raw pointer to output (must call wait() first)
template<typename T>
const T* CompiledGraph::get_output_ptr(const std::string& name) {
    // This is a simplified implementation - in a real implementation,
    // we'd need to manage the tile acquisition/release properly
    throw std::runtime_error("get_output_ptr not implemented in minimal version");
}

// Template instantiations
template nntile::tensor::Tensor<float>& CompiledGraph::get_tensor<float>(const std::string& name);
template nntile::tensor::Tensor<double>& CompiledGraph::get_tensor<double>(const std::string& name);

template void CompiledGraph::bind_data<float>(const std::string& name, const float* data, size_t count);
template void CompiledGraph::bind_data<double>(const std::string& name, const double* data, size_t count);
template void CompiledGraph::bind_data<float>(const std::string& name, const std::vector<float>& data);
template void CompiledGraph::bind_data<double>(const std::string& name, const std::vector<double>& data);

template std::vector<float> CompiledGraph::get_output<float>(const std::string& name);
template std::vector<double> CompiledGraph::get_output<double>(const std::string& name);

template const float* CompiledGraph::get_output_ptr<float>(const std::string& name);
template const double* CompiledGraph::get_output_ptr<double>(const std::string& name);

} // namespace nntile::graph