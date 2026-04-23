/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/adam_step.cc
 * TensorGraph adam_step operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/adam_step.hh"

#include <memory>
#include <stdexcept>
#include <vector>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tile/graph_ops.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/tensor/adam_step.hh"

namespace nntile::graph::tensor
{



void adam_step(Index num_iter, Scalar beta_1, Scalar beta_2,
               Scalar eps, Scalar lr, Scalar weight_decay,
               TensorGraph::TensorNode* grad,
               TensorGraph::TensorNode* first_moment,
               TensorGraph::TensorNode* second_moment,
               TensorGraph::TensorNode* p)
{
    if(grad == nullptr || first_moment == nullptr ||
       second_moment == nullptr || p == nullptr)
        throw std::invalid_argument("adam_step: tensors must be non-null");
    if(grad->graph() != first_moment->graph() ||
       first_moment->graph() != second_moment->graph() ||
       second_moment->graph() != p->graph())
        throw std::invalid_argument("adam_step: tensors must belong to same graph");
    if(grad->dtype() != first_moment->dtype() ||
       first_moment->dtype() != second_moment->dtype() ||
       second_moment->dtype() != p->dtype())
        throw std::invalid_argument("adam_step: tensors must have same dtype");
    validate_same_shape_and_merge(grad, first_moment, "adam_step");
    validate_same_shape_and_merge(grad, second_moment, "adam_step");
    validate_same_shape_and_merge(grad, p, "adam_step");
    auto op = std::make_shared<TensorAdamStepOp>(
        num_iter, beta_1, beta_2, eps, lr, weight_decay,
        grad, first_moment, second_moment, p);
    p->graph()->add_op(op);
}

void TensorAdamStepOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& m = ctx.tile_map;
    const auto& vg = tile_lower::tiles_of(m, grad);
    const auto& vm = tile_lower::tiles_of(m, first_moment);
    const auto& vv = tile_lower::tiles_of(m, second_moment);
    const auto& vp = tile_lower::tiles_of(m, p);
    if(vg.size() != vm.size() || vg.size() != vv.size() || vg.size() != vp.size())
    {
        throw std::runtime_error(
            "lower_to_tile ADAM_STEP: tile count mismatch");
    }
    tile_lower::assert_same_elementwise_layout(grad, first_moment, "ADAM_STEP");
    tile_lower::assert_same_elementwise_layout(grad, second_moment, "ADAM_STEP");
    tile_lower::assert_same_elementwise_layout(grad, p, "ADAM_STEP");
    auto step_iter = std::make_shared<Index>(num_iter);
    const size_t n = vg.size();
    for(size_t i = 0; i < n; ++i)
    {
        tile_graph::adam_step(step_iter, (i + 1 == n), beta_1, beta_2, eps, lr,
            weight_decay, vg[i], vm[i], vv[i], vp[i]);
    }
}

} // namespace nntile::graph::tensor
