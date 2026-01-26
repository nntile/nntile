/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/gelutanh_inplace.cc
 * Logical graph GeLUTanh in-place operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/gelutanh_inplace.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! GeLUTanh in-place: x = gelutanh(x)
void gelutanh_inplace(LogicalGraph::TensorNode& x)
{
    OpAttrs attrs = GelutanhAttrs{};
    x.graph().add_op(
        OpType::GELUTANH_INPLACE,
        attrs,
        {&x},
        {&x}
    );
}

} // namespace nntile::graph