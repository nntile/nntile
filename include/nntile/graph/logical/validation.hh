/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/validation.hh
 * Logical graph validation utilities.
 *
 * @version 1.1.0
 * */

// #pragma once

// // Include standard headers
// #include <stdexcept>

// // Include other NNTile headers
// #include <nntile/graph/logical_graph.hh>

// namespace nntile::graph
// {

// //! Validate inputs for binary operations
// void validate_binary_inputs(
//     LogicalGraph::TensorNode& x,
//     LogicalGraph::TensorNode& y,
//     LogicalGraph& expected_graph)
// {
//     if(&x.graph() != &expected_graph || &y.graph() != &expected_graph)
//     {
//         throw std::invalid_argument(
//             "Binary operation: input tensors must belong to the same graph");
//     }

//     if(x.dtype() != y.dtype())
//     {
//         throw std::invalid_argument(
//             "Binary operation: input tensors must have the same dtype");
//     }

//     if(x.shape() != y.shape())
//     {
//         throw std::invalid_argument(
//             "Binary operation: input tensors must have the same shape");
//     }
// }

// } // namespace nntile::graph
