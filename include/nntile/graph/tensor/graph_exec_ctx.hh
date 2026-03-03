/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/graph_exec_ctx.hh
 * TensorGraph::ExecutionContext - maps TensorNode to runtime tensor::Tensor data.
 *
 * @version 1.1.0
 * */

#pragma once

#include <map>
#include <memory>
#include <stdexcept>

#include <nntile/graph/dtype.hh>
#include <nntile/tensor/tensor.hh>

#include <nntile/graph/tensor/graph_data_node.hh>

namespace nntile::graph
{

//! Maps TensorGraph::TensorNode to runtime tensor::Tensor data.
//! Built during compilation; used by TensorGraph::OpNode::execute().
class TensorGraph::ExecutionContext
{
public:
    //! Register a data node with its runtime tensor
    template<typename T>
    void register_tensor(const TensorNode* node,
                         std::shared_ptr<tensor::Tensor<T>> tensor)
    {
        tensor_map_[node] = tensor;
    }

    //! Get typed runtime tensor for a data node
    template<typename T>
    tensor::Tensor<T>& get_tensor(const TensorNode* node)
    {
        auto it = tensor_map_.find(node);
        if(it == tensor_map_.end())
        {
            throw std::runtime_error(
                "ExecutionContext::get_tensor: node not found");
        }
        auto ptr = std::static_pointer_cast<tensor::Tensor<T>>(it->second);
        if(!ptr)
        {
            throw std::runtime_error(
                "ExecutionContext::get_tensor: wrong type");
        }
        return *ptr;
    }

    //! Get data type for a data node (from TensorNode::dtype())
    DataType get_dtype(const TensorNode* node) const
    {
        return node->dtype();
    }

private:
    std::map<const TensorNode*, std::shared_ptr<void>> tensor_map_;
};

} // namespace nntile::graph
