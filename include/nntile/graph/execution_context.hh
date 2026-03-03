/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/execution_context.hh
 * ExecutionContext - maps DataNodes to runtime data.
 *
 * @version 1.1.0
 * */

#pragma once

#include <map>
#include <memory>
#include <stdexcept>

#include <nntile/graph/dtype.hh>
#include <nntile/tensor/tensor.hh>

namespace nntile::graph
{

//! Maps DataNodes to runtime data.
//! Built during compilation; used by OpNode::execute().
//! @tparam DataNode The data node type (e.g. TensorGraphNode)
//! @tparam TensorT The tensor type template (e.g. tensor::Tensor). Default: tensor::Tensor.
template<typename DataNode, template<typename> class TensorT = tensor::Tensor>
class ExecutionContext
{
public:
    //! Register a data node with its runtime tensor
    //! DataNode must have dtype() method (dtype is part of the node)
    template<typename T>
    void register_tensor(const DataNode* node,
                         std::shared_ptr<TensorT<T>> tensor)
    {
        tensor_map_[node] = tensor;
    }

    //! Get typed runtime tensor for a data node
    template<typename T>
    TensorT<T>& get_tensor(const DataNode* node)
    {
        auto it = tensor_map_.find(node);
        if(it == tensor_map_.end())
        {
            throw std::runtime_error("ExecutionContext::get_tensor: node not found");
        }
        auto ptr = std::static_pointer_cast<TensorT<T>>(it->second);
        if(!ptr)
        {
            throw std::runtime_error("ExecutionContext::get_tensor: wrong type");
        }
        return *ptr;
    }

    //! Get data type for a data node (from DataNode::dtype())
    DataType get_dtype(const DataNode* node) const
    {
        return node->dtype();
    }

private:
    std::map<const DataNode*, std::shared_ptr<void>> tensor_map_;
};

} // namespace nntile::graph
