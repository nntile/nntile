/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_graph_recorder_impl.h
 * Internal graph recording helpers (libnntile TensorGraph).
 */

#pragma once

#include "nntile_tensor_gc.h"

#include <string>
#include <unordered_map>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/dtype.hh>
#include <nntile/tensor/graph.hh>

namespace at
{
class Tensor;
}

namespace torch_nntile
{

void on_tensor_impl_released(TensorImplKey key);

void init_nntile_input_from_cpu(
    const at::Tensor &cpu_src,
    at::Tensor &nntile_dst);

nntile::TensorGraph::TensorNode *get_or_create_data_node(
    const at::Tensor &tensor,
    const std::vector<nntile::Index> &shape,
    nntile::DataType dtype,
    bool mark_as_input);

void register_data_node(
    const at::Tensor &tensor,
    nntile::TensorGraph::TensorNode *node);

nntile::TensorGraph::TensorNode *lookup_data_node(
    const at::Tensor &tensor,
    const std::vector<nntile::Index> &shape);

void register_param_grad_node(
    const at::Tensor &param,
    nntile::TensorGraph::TensorNode *grad_node);

nntile::TensorGraph::TensorNode *lookup_param_grad_node(
    const at::Tensor &param);

void register_grad_alias_for_host_copy(
    at::Tensor &grad,
    nntile::TensorGraph::TensorNode *grad_node);

void record_view_alias(const at::Tensor &self, const at::Tensor &view);

//! Record-path timing buckets (printed by print_info).
void note_record_linear_bwd(double seconds);
void note_record_ce_bwd(double seconds);
void note_record_relu_bwd(double seconds);
void note_record_gemm(double seconds);
void note_record_narrow_copy(std::uint64_t nelems);
void note_record_transpose_copy(std::uint64_t nelems);

} // namespace torch_nntile
