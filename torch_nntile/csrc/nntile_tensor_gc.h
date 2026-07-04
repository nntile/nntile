/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_tensor_gc.h
 * Tensor staging / metadata-only tracking for GC.
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

bool is_metadata_only_tensor(const at::Tensor &tensor);
bool is_staged_input_tensor(const at::Tensor &tensor);
bool is_staged_input_impl(TensorImplKey impl_key);
bool has_host_staging(const at::Tensor &tensor);

void mark_metadata_only_tensor(const at::Tensor &tensor);
void mark_staged_input_tensor(const at::Tensor &tensor);

void clear_tensor_gc_state();

void on_host_storage_released(void *host_data_ptr, void *storage_ctx);

at::Tensor empty_metadata_tensor(
    c10::IntArrayRef size,
    c10::ScalarType dtype,
    c10::Device device);

void ensure_host_staging(at::Tensor &tensor);

} // namespace torch_nntile
