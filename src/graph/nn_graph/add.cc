/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn_graph/add.cc
 * NNGraph add operation implementation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/nn_graph/add.hh"

// Include standard headers
#include <stdexcept>

namespace nntile::graph
{

NNGraph::TensorNode& add(
    NNGraph& graph,
    Scalar alpha,
    NNGraph::TensorNode& x,
    Scalar beta,
    NNGraph::TensorNode& y,
    const std::string& output_name)
{
    if(&x.data().graph() != &graph.logical_graph())
    {
        throw std::invalid_argument(
            "add: x must belong to the given graph");
    }
    if(&y.data().graph() != &graph.logical_graph())
    {
        throw std::invalid_argument(
            "add: y must belong to the given graph");
    }

    LogicalGraph::TensorNode& z_data = add(
        alpha, x.data(), beta, y.data(), output_name);
    bool out_requires_grad = x.requires_grad() || y.requires_grad();
    return graph.tensor(z_data, out_requires_grad);
}

} // namespace nntile::graph
