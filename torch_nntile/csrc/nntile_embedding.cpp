/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_embedding.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

#ifdef TORCH_NNTILE_USE_LIBNNTILE
#include <nntile/base_types.hh>
#endif

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
    TORCH_CHECK(
        padding_idx < 0,
        "nntile embedding: padding_idx >= 0 is not supported");
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
    TORCH_CHECK(
        weight.is_contiguous() && indices.is_contiguous(),
        "nntile embedding requires contiguous tensors");
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
    TORCH_CHECK(
        grad_output.is_contiguous() && indices.is_contiguous(),
        "nntile embedding_dense_backward requires contiguous tensors");
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

at::Tensor prepare_indices(const at::Tensor &indices)
{
    return indices.is_contiguous() ? indices : indices.contiguous();
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
    check_embedding_forward_inputs(weight, indices);
    check_embedding_optional_args(padding_idx, scale_grad_by_freq, sparse);

    const at::Tensor indices_contig = prepare_indices(indices);
    const std::vector<int64_t> out_shape =
        embedding_output_shape(indices_contig.sizes(), weight.size(1));
    at::Tensor output = at::empty(
        out_shape,
        weight.options().memory_format(at::MemoryFormat::Contiguous));

    const nntile::Index axis =
        static_cast<nntile::Index>(indices_contig.dim());
    pin_graph_op_inputs({weight, indices_contig});
    pin_graph_op_output(output, true);
    tensor_embedding_forward_fp32(
        indices_contig.data_ptr<std::int64_t>(),
        indices_contig.sizes(),
        weight.data_ptr<float>(),
        weight.sizes(),
        output.data_ptr<float>(),
        output.sizes(),
        axis);
    return output;
}

at::Tensor embedding_dense_backward(
    const at::Tensor &grad_output,
    const at::Tensor &indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq)
{
    check_embedding_backward_inputs(grad_output, indices, num_weights);
    check_embedding_optional_args(padding_idx, scale_grad_by_freq, false);

    const at::Tensor indices_contig = prepare_indices(indices);
    const int64_t embed_dim = grad_output.size(-1);
    TORCH_CHECK(
        num_weights > 0 && embed_dim > 0,
        "nntile embedding_dense_backward: invalid weight shape");

    at::Tensor grad_weight = at::zeros(
        {num_weights, embed_dim},
        grad_output.options().memory_format(at::MemoryFormat::Contiguous));

    const nntile::Index axis =
        static_cast<nntile::Index>(indices_contig.dim());
    pin_graph_op_inputs({grad_output, indices_contig});
    pin_graph_op_output(grad_weight, false);
    tensor_embedding_backward_fp32(
        indices_contig.data_ptr<std::int64_t>(),
        indices_contig.sizes(),
        grad_output.data_ptr<float>(),
        grad_output.sizes(),
        grad_weight.data_ptr<float>(),
        grad_weight.sizes(),
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
