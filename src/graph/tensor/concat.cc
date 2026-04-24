/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/concat.cc
 * TensorGraph concat operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/concat.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/clear.hh"
#include "nntile/graph/tensor/copy_intersection.hh"

#include <stdexcept>
#include <vector>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tile/lowering_context.hh"

namespace nntile::graph::tensor
{

void TensorConcatOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Same decomposition as tensor-level concat: zero dst, then copy each
    // operand into its slice of the output (reuse virtual lowering of CLEAR
    // and COPY_INTERSECTION).
    TensorClearOp clear_op(output);
    clear_op.lower_to_tile(ctx);

    const Index ndim = output->ndim();
    std::vector<Index> zero(static_cast<size_t>(ndim), 0);
    TensorCopyIntersectionOp copy_a(a, zero, output, zero);
    copy_a.lower_to_tile(ctx);

    std::vector<Index> src_b_off(zero);
    src_b_off[static_cast<size_t>(axis)] =
        a->shape()[static_cast<size_t>(axis)];
    TensorCopyIntersectionOp copy_b(b, src_b_off, output, zero);
    copy_b.lower_to_tile(ctx);
}

TensorGraph::TensorNode* concat(
    TensorGraph::TensorNode* a,
    TensorGraph::TensorNode* b,
    Index axis,
    const std::string& output_name)
{
    if(a == nullptr || b == nullptr)
        throw std::invalid_argument("concat: input tensors must be non-null");
    if(a->graph() != b->graph())
        throw std::invalid_argument(
            "concat: tensors must belong to same graph");
    if(a->dtype() != b->dtype())
        throw std::invalid_argument(
            "concat: tensors must have same dtype");
    if(a->ndim() != b->ndim())
        throw std::invalid_argument(
            "concat: tensors must have same number of dimensions");

    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();
    Index n_dim = static_cast<Index>(a_shape.size());

    if(axis < 0 || axis >= n_dim)
        throw std::invalid_argument("concat: axis out of range");

    // Output shape: same as a and b except on axis
    std::vector<Index> output_shape = a_shape;
    output_shape[axis] = a_shape[axis] + b_shape[axis];
    for(Index i = 0; i < n_dim; ++i)
    {
        if(i != axis && a_shape[i] != b_shape[i])
        {
            throw std::invalid_argument(
                "concat: non-concat dimensions must match");
        }
    }

    TensorGraph* graph = a->graph();
    TensorGraph::TensorNode* output = graph->data(
        output_shape, output_name, a->dtype());

    auto op = std::make_shared<TensorConcatOp>(a, b, output, axis);
    graph->add_op(op);

    return output;
}

} // namespace nntile::graph::tensor
