/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_broadcast.h
 * Broadcast helpers built from chained ``scale_slice`` ops.
 */

#pragma once

#include <c10/util/ArrayRef.h>

#ifdef TORCH_NNTILE_USE_LIBNNTILE
#include <nntile/base_types.hh>
#include <nntile/tensor/graph.hh>
#include <vector>
#endif

namespace torch_nntile
{

void tensor_repeat_fp32(
    const float *input_data,
    float *out_data,
    c10::IntArrayRef input_shape,
    c10::IntArrayRef repeats,
    c10::IntArrayRef out_shape);

void tensor_broadcast_scalar_fp32(
    const float *scalar_data,
    float *out_data,
    c10::IntArrayRef out_shape);

#ifdef TORCH_NNTILE_USE_LIBNNTILE
nntile::TensorGraph::TensorNode *broadcast_scale_slice_chain(
    nntile::TensorGraph::TensorNode *src,
    nntile::TensorGraph::TensorNode *dst,
    nntile::TensorGraph &graph,
    const std::vector<nntile::Index> &dst_shape);
#endif

} // namespace torch_nntile
