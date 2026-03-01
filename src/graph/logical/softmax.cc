/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/softmax.cc
 * Logical graph softmax operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/softmax.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Softmax operation: y = softmax(maxsumexp, x, alpha)
LogicalGraph::TensorNode& softmax(
    LogicalGraph::TensorNode& maxsumexp,
    LogicalGraph::TensorNode& x,
    const std::string& output_name,
    Scalar alpha,
    Index axis)
{
    if(&maxsumexp.graph() != &x.graph())
    {
        throw std::invalid_argument(
            "softmax: tensors must belong to the same graph");
    }

    if(maxsumexp.dtype() != x.dtype())
    {
        throw std::invalid_argument(
            "softmax: tensors must have the same dtype");
    }

    if(axis < 0)
    {
        axis += x.ndim();
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "softmax: axis out of bounds");
    }

    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    auto attrs = std::make_shared<LogSumExpAttrs>(LogSumExpAttrs{alpha, 1.0, axis});
    x.graph().add_op(
        OpType::SOFTMAX,
        attrs,
        {&maxsumexp, &x},
        {&output}
    );

    return output;
}

} // namespace nntile::graph
