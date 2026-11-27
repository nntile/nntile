/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_broadcast.h
 * Broadcast helpers built from chained ``scale_slice`` ops.
 */

#pragma once

#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>

#include <nntile/base_types.hh>
#include <nntile/tensor/graph.hh>
#include <vector>

namespace torch_nntile
{

void tensor_repeat_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    c10::IntArrayRef repeats);

void tensor_broadcast_scalar_fp32(
    const at::Tensor &scalar,
    at::Tensor &out);

nntile::TensorGraph::TensorNode *broadcast_scale_slice_chain(
    nntile::TensorGraph::TensorNode *src,
    nntile::TensorGraph::TensorNode *dst,
    nntile::TensorGraph &graph,
    const std::vector<nntile::Index> &dst_shape);

} // namespace torch_nntile
