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

#ifdef TORCH_NNTILE_USE_LIBNNTILE
#include <nntile/base_types.hh>
#include <nntile/dtype.hh>
#include <nntile/tensor/graph.hh>
#endif

namespace at
{
class Tensor;
}

namespace torch_nntile
{

//! Pin graph inputs and persistent params for the current capture.
void pin_graph_op_inputs(const std::vector<at::Tensor> &inputs);

//! Pin output when it is user-visible across compile/run.
void pin_graph_op_output(const at::Tensor &output, bool is_user_visible);

void on_tensor_impl_released(TensorImplKey key);

void mark_persistent_graph_tensor(const at::Tensor &tensor);

bool read_nntile_staging_to_host(const at::Tensor &tensor, void *host_ptr);

void init_nntile_input_from_cpu(
    const at::Tensor &cpu_src,
    at::Tensor &nntile_dst);

bool can_read_nntile_tensor_from_staging(const at::Tensor &tensor);

#ifdef TORCH_NNTILE_USE_LIBNNTILE

nntile::TensorGraph &recorder_graph();

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

void push_relu_preactivation_node(nntile::TensorGraph::TensorNode *node);

nntile::TensorGraph::TensorNode *pop_relu_preactivation_node(
    const std::vector<nntile::Index> &shape);

void track_graph_node(nntile::TensorGraph::TensorNode *node);

void record_view_alias(const at::Tensor &self, const at::Tensor &view);

#endif // TORCH_NNTILE_USE_LIBNNTILE

} // namespace torch_nntile
