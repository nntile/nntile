/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_tensor_gc.h
 * Tensor metadata tracking for GC (0-byte nntile storage).
 */

#pragma once

#include <c10/core/TensorImpl.h>
#include <c10/util/ArrayRef.h>

#include <cstdint>

namespace at
{
class Tensor;
}

namespace torch_nntile
{

using TensorImplKey = c10::TensorImpl *;

TensorImplKey tensor_impl_key(const at::Tensor &tensor);

//! True when a nntile tensor has 0-byte storage (no host payload).
bool is_metadata_only_tensor(const at::Tensor &tensor);

void clear_tensor_gc_state();

void on_host_storage_released(void *storage_ctx);

void register_metadata_tensor_storage(const at::Tensor &tensor);

at::Tensor empty_metadata_tensor(
    c10::IntArrayRef size,
    c10::ScalarType dtype,
    c10::Device device);

} // namespace torch_nntile
