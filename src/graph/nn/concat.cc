/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/concat.cc
 * NNGraph concat operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/concat.hh"
#include "nntile/graph/nn/graph_data_node.hh"
#include "nntile/graph/tensor/concat.hh"

#include <stdexcept>

namespace nntile::graph
{

NNGraph::TensorNode* concat(
    NNGraph::TensorNode* a,
    NNGraph::TensorNode* b,
    Index axis,
    const std::string& output_name)
{
    if(a == nullptr || b == nullptr)
        throw std::invalid_argument("concat: input tensors must be non-null");

    TensorGraph::TensorNode* out_data = graph::tensor::concat(
        a->data(), b->data(), axis, output_name);
    NNGraph* graph = a->graph();
    return graph->tensor(out_data, false);  // no grad for inference
}

} // namespace nntile::graph
