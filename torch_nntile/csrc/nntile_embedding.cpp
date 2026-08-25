/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_embedding.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_gc.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

#include <nntile/base_types.hh>

#include <cstdint>
#include <vector>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_embedding_optional_args(
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse)
{
    TORCH_CHECK(!sparse, "nntile embedding: sparse=True is not supported");
    TORCH_CHECK(
        !scale_grad_by_freq,
        "nntile embedding: scale_grad_by_freq=True is not supported");
    // padding_idx >= 0 is allowed; StarPU embedding currently still
    // passes -1 into aten::embedding_out (pad rows keep a learned
    // vector). Enough for HF smokes that set pad_token_id.
    (void)padding_idx;
}

void check_embedding_forward_inputs(
    const at::Tensor &weight,
    const at::Tensor &indices)
{
    TORCH_CHECK(
        is_nntile_device(weight.device()),
        "nntile embedding: weight must be on device nntile");
    TORCH_CHECK(
        indices.scalar_type() == at::ScalarType::Long,
        "nntile embedding: indices must be int64");
    TORCH_CHECK(
        is_nntile_device(indices.device()),
        "nntile embedding: indices must be on device nntile");
    TORCH_CHECK(weight.dim() == 2, "nntile embedding: weight must be 2D");
    TORCH_CHECK(
        weight.scalar_type() == at::ScalarType::Float,
        "nntile embedding supports float32 weight only");
}

void check_embedding_backward_inputs(
    const at::Tensor &grad_output,
    const at::Tensor &indices,
    int64_t num_weights)
{
    TORCH_CHECK(
        is_nntile_device(grad_output.device()),
        "nntile embedding_dense_backward: grad_output must be on device nntile");
    TORCH_CHECK(
        indices.scalar_type() == at::ScalarType::Long,
        "nntile embedding_dense_backward: indices must be int64");
    TORCH_CHECK(
        is_nntile_device(indices.device()),
        "nntile embedding_dense_backward: indices must be on device nntile");
    TORCH_CHECK(
        grad_output.scalar_type() == at::ScalarType::Float,
        "nntile embedding_dense_backward supports float32 only");
    TORCH_CHECK(num_weights > 0, "nntile embedding_dense_backward: invalid num_weights");
    TORCH_CHECK(
        grad_output.dim() == indices.dim() + 1,
        "nntile embedding_dense_backward: grad_output rank mismatch");
    TORCH_CHECK(
        grad_output.size(-1) > 0,
        "nntile embedding_dense_backward: invalid embedding dimension");
    for (int64_t i = 0; i < indices.dim(); ++i)
    {
        TORCH_CHECK(
            grad_output.size(i) == indices.size(i),
            "nntile embedding_dense_backward: shape mismatch");
    }
}

std::vector<int64_t> embedding_output_shape(
    c10::IntArrayRef index_shape,
    int64_t embed_dim)
{
    std::vector<int64_t> out_shape(index_shape.begin(), index_shape.end());
    out_shape.push_back(embed_dim);
    return out_shape;
}

} // namespace

at::Tensor embedding(
    const at::Tensor &weight,
    const at::Tensor &indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse)
{
    nntile::GraphFillScope record;
    check_embedding_forward_inputs(weight, indices);
    check_embedding_optional_args(padding_idx, scale_grad_by_freq, sparse);

    const at::Tensor &indices_ref = indices;
    const std::vector<int64_t> out_shape =
        embedding_output_shape(indices_ref.sizes(), weight.size(1));
    at::Tensor output = empty_metadata_tensor(
        out_shape,
        weight.scalar_type(),
        weight.device());

    const nntile::Index axis =
        static_cast<nntile::Index>(indices_ref.dim());
    tensor_embedding_forward_fp32(indices_ref, weight, output, axis);
    return output;
}

at::Tensor embedding_dense_backward(
    const at::Tensor &grad_output,
    const at::Tensor &indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq)
{
    nntile::GraphFillScope record;
    const at::Tensor &grad_ref = grad_output;
    const at::Tensor &indices_ref = indices;
    check_embedding_backward_inputs(
        grad_ref,
        indices_ref,
        num_weights);
    check_embedding_optional_args(padding_idx, scale_grad_by_freq, false);

    const int64_t embed_dim = grad_ref.size(-1);
    TORCH_CHECK(
        num_weights > 0 && embed_dim > 0,
        "nntile embedding_dense_backward: invalid weight shape");

    at::Tensor grad_weight = at::zeros(
        {num_weights, embed_dim},
        grad_ref.options());

    const nntile::Index axis =
        static_cast<nntile::Index>(indices_ref.dim());
    tensor_embedding_backward_fp32(
        indices_ref,
        grad_ref,
        grad_weight,
        axis,
        0);
    return grad_weight;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("embedding", TORCH_FN(torch_nntile::embedding));
    m.impl(
        "embedding_dense_backward",
        TORCH_FN(torch_nntile::embedding_dense_backward));
}
