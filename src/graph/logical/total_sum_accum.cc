/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/total_sum_accum.cc
 * Logical graph total_sum_accum operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/total_sum_accum.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Total sum accumulation: val = alpha * sum(logsumexp * src) + beta * val
void total_sum_accum(
    LogicalGraph::TensorNode& logsumexp,
    LogicalGraph::TensorNode& src,
    LogicalGraph::TensorNode& class_labels,
    LogicalGraph::TensorNode& val,
    Scalar alpha,
    Index ignore_index)
{
    if(&logsumexp.graph() != &src.graph() || &src.graph() != &class_labels.graph() || &class_labels.graph() != &val.graph())
    {
        throw std::invalid_argument(
            "total_sum_accum: all tensors must belong to the same graph");
    }

    if(logsumexp.dtype() != src.dtype())
    {
        throw std::invalid_argument(
            "total_sum_accum: logsumexp and src must have the same dtype");
    }

    if(class_labels.dtype() != DataType::INT64)
    {
        throw std::invalid_argument(
            "total_sum_accum: class_labels must be INT64");
    }

    if(val.dtype() != DataType::FP32)
    {
        throw std::invalid_argument(
            "total_sum_accum: val must be FP32");
    }

    OpAttrs attrs = TotalSumAccumAttrs{alpha, ignore_index};
    logsumexp.graph().add_op(
        OpType::TOTAL_SUM_ACCUM,
        attrs,
        {&logsumexp, &src, &class_labels, &val},
        {&val}
    );
}

} // namespace nntile::graph