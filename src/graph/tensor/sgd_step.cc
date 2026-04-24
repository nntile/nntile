/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sgd_step.cc
 * TensorGraph sgd_step operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sgd_step.hh"

#include <memory>
#include <stdexcept>
#include <vector>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tile/graph_ops.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/tensor/sgd_step.hh"

namespace nntile::graph::tensor
{



void sgd_step(Index num_iter, Scalar momentum, Scalar lr,
              Scalar weight_decay, Scalar dampening, bool nesterov,
              TensorGraph::TensorNode* grad,
              TensorGraph::TensorNode* velocity,
              TensorGraph::TensorNode* p)
{
    if(grad == nullptr || velocity == nullptr || p == nullptr)
        throw std::invalid_argument("sgd_step: tensors must be non-null");
    if(grad->graph() != velocity->graph() || velocity->graph() != p->graph())
        throw std::invalid_argument("sgd_step: tensors must belong to same graph");
    if(grad->dtype() != velocity->dtype() || velocity->dtype() != p->dtype())
        throw std::invalid_argument("sgd_step: tensors must have same dtype");
    validate_same_shape_and_merge(grad, velocity, "sgd_step");
    validate_same_shape_and_merge(grad, p, "sgd_step");
    auto op = std::make_shared<TensorSgdStepOp>(
        num_iter, momentum, lr, weight_decay, dampening, nesterov,
        grad, velocity, p);
    p->graph()->add_op(op);
}

void TensorSgdStepOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& m = ctx.tile_map;
    const auto& vg = tile_lower::tiles_of(m, grad);
    const auto& vv = tile_lower::tiles_of(m, velocity);
    const auto& vp = tile_lower::tiles_of(m, p);
    if(vg.size() != vv.size() || vg.size() != vp.size())
    {
        throw std::runtime_error("lower_to_tile SGD_STEP: tile count mismatch");
    }
    tile_lower::assert_same_elementwise_layout(grad, velocity, "SGD_STEP");
    tile_lower::assert_same_elementwise_layout(grad, p, "SGD_STEP");
    auto step_iter = std::make_shared<Index>(num_iter);
    const size_t n = vg.size();
    for(size_t i = 0; i < n; ++i)
    {
        tile_graph::sgd_step(step_iter, (i + 1 == n), momentum, lr, weight_decay,
            dampening, nesterov, vg[i], vv[i], vp[i]);
    }
}

} // namespace nntile::graph::tensor
