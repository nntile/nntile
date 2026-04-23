/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/subtract_indexed_outputs.cc
 * TensorGraph subtract_indexed_outputs operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/subtract_indexed_outputs.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/subtract_indexed_outputs.hh"

namespace nntile::graph::tensor
{



void subtract_indexed_outputs(Scalar val,
                             TensorGraph::TensorNode* labels,
                             TensorGraph::TensorNode* dst,
                             Index ignore_index)
{
    if(labels == nullptr || dst == nullptr)
        throw std::invalid_argument(
            "subtract_indexed_outputs: tensors must be non-null");
    if(labels->graph() != dst->graph())
        throw std::invalid_argument(
            "subtract_indexed_outputs: tensors must belong to same graph");
    if(labels->dtype() != DataType::INT64)
        throw std::invalid_argument(
            "subtract_indexed_outputs: labels must have INT64 dtype");
    // labels.dim[i] == dst.dim[i+1]: labels index the batch dims of dst
    if(labels->ndim() + 1 != dst->ndim())
    {
        throw std::invalid_argument(
            "subtract_indexed_outputs: dst must have ndim = labels.ndim + 1");
    }
    for(Index i = 0; i < labels->ndim(); ++i)
    {
        if(labels->shape()[i] != dst->shape()[i + 1])
        {
            throw std::invalid_argument(
                "subtract_indexed_outputs: labels.dim[" +
                std::to_string(i) + "] must match dst.dim[" +
                std::to_string(i + 1) + "] (" +
                std::to_string(labels->shape()[i]) + " vs " +
                std::to_string(dst->shape()[i + 1]) + ")");
        }
        merge_axis(labels->mutable_axes()[i],
                   dst->mutable_axes()[i + 1]);
    }

    auto op = std::make_shared<TensorSubtractIndexedOutputsOp>(
        val, labels, dst, ignore_index);
    dst->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
