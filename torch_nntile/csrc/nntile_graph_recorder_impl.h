/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_graph_recorder_impl.h
 * Internal graph recording helpers (libnntile TensorGraph).
 */

#pragma once

#include <nntile/dtype.hh>
#include <nntile/tensor/graph.hh>

#include <vector>

namespace at
{
class Tensor;
}

namespace torch_nntile
{

nntile::TensorGraph &recorder_graph();

nntile::TensorGraph::TensorNode *get_or_create_data_node(
    void *data_ptr,
    const std::vector<nntile::Index> &shape,
    nntile::DataType dtype,
    bool mark_as_input);

void register_data_node(
    void *data_ptr,
    nntile::TensorGraph::TensorNode *node);

nntile::TensorGraph::TensorNode *lookup_data_node(void *data_ptr);

//! Keep tensor storage alive until execute() (CUDA record_stream analog).
void pin_tensor_for_graph(const at::Tensor &tensor);

//! Pin op inputs and user-held outputs. Do not pin backward return buffers
//! that autograd will steal into leaf .grad (extra refs block stealing).
void pin_graph_op_inputs(const std::vector<at::Tensor> &inputs);

void pin_graph_op_output(const at::Tensor &output, bool is_user_visible);

} // namespace torch_nntile
