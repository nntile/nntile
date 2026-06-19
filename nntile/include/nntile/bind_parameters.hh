/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/bind_parameters.hh
 * Copy staged host parameter bytes into runtime tiles via ``bind_data``.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/nn/graph_decl.hh>

namespace nntile
{

class Runtime;

//! ``bind_data`` from host bytes previously stored on the tensor (e.g. after
//! ``Module::load``). Throws if no host bytes are present.
void bind_tensor_host_data(Runtime &rt, NNGraph::TensorNode *tensor);

} // namespace nntile
