/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_executor.h
 * @version 2.0.0
 */

#pragma once

#include "nntile_gemm_layout.h"

#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>

#include <cstdint>
#include <array>
#include <optional>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/tensor/graph_fill_timer.hh>

namespace torch_nntile
{

//! Densify ``src`` (view OK) into contiguous ``dst`` (same shape).
void tensor_copy_fp32(const at::Tensor &src, at::Tensor &dst);
void tensor_copy_i64(const at::Tensor &src, at::Tensor &dst);
void tensor_copy_bool(const at::Tensor &src, at::Tensor &dst);

void tensor_fill_fp32(at::Tensor &self, float value);

void tensor_norm_fp32(
    const at::Tensor &x,
    at::Tensor &out);

void tensor_norm_slice_fp32(
    const at::Tensor &x,
    at::Tensor &out,
    int64_t axis,
    bool keepdim);





} // namespace torch_nntile
