/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_gemm_layout.cpp
 */

#include "nntile_gemm_layout.h"

#include <ATen/native/LinearAlgebraUtils.h>

#include <nntile/tensor/ops/gemm.hh>

#include <algorithm>
#include <stdexcept>

namespace torch_nntile
{

namespace
{

std::vector<int64_t> sizes_to_vector(c10::IntArrayRef sizes)
{
    return std::vector<int64_t>(sizes.begin(), sizes.end());
}

void require_gemm_layout(
    const at::Tensor &tensor,
    const GemmMatrixLayout &layout,
    const char *name)
{
    TORCH_CHECK(
        !layout.needs_copy,
        "nntile gemm: ",
        name,
        " must be contiguous or row/column-contiguous");
    (void)tensor;
}

GemmMatrixLayout layout_from_prepared_2d(const at::Tensor &tensor)
{
    GemmMatrixLayout layout;
    TORCH_CHECK(tensor.dim() == 2, "expected a 2D matrix tensor");
    if (tensor.is_contiguous())
    {
        layout.gemm_shape = sizes_to_vector(tensor.sizes());
        layout.trans = false;
        layout.needs_copy = false;
        return layout;
    }
    if (at::native::is_row_or_column_contiguous(tensor))
    {
        const auto sizes = tensor.sizes();
        layout.gemm_shape = {sizes[1], sizes[0]};
        layout.trans = true;
        layout.needs_copy = false;
        return layout;
    }
    layout.gemm_shape = sizes_to_vector(tensor.sizes());
    layout.trans = false;
    layout.needs_copy = true;
    return layout;
}

GemmMatrixLayout layout_from_prepared_batched(const at::Tensor &tensor)
{
    GemmMatrixLayout layout;
    TORCH_CHECK(tensor.dim() == 3, "expected a 3D batched matrix tensor");
    if (at::native::is_blas_compatible_row_major_order(tensor))
    {
        layout.gemm_shape = sizes_to_vector(tensor.sizes());
        layout.trans = false;
        layout.needs_copy = false;
        return layout;
    }
    const int64_t batch = tensor.size(0);
    if (batch > 0)
    {
        GemmMatrixLayout slice_layout =
            layout_from_prepared_2d(tensor.select(0, 0));
        if (!slice_layout.needs_copy)
        {
            layout.trans = slice_layout.trans;
            const auto sizes = tensor.sizes();
            if (layout.trans)
            {
                layout.gemm_shape = {sizes[0], sizes[2], sizes[1]};
            }
            else
            {
                layout.gemm_shape = sizes_to_vector(sizes);
            }
            layout.needs_copy = false;
            return layout;
        }
    }
    layout.gemm_shape = sizes_to_vector(tensor.sizes());
    layout.trans = false;
    layout.needs_copy = true;
    return layout;
}

void validate_gemm_contraction(
    c10::IntArrayRef a_shape,
    c10::IntArrayRef b_shape,
    int64_t ndim,
    int64_t batch_ndim,
    bool trans_a,
    bool trans_b)
{
    const int64_t a_rank = static_cast<int64_t>(a_shape.size());
    const int64_t b_rank = static_cast<int64_t>(b_shape.size());

    TORCH_CHECK(ndim > 0, "nntile gemm: ndim must be positive");
    TORCH_CHECK(
        batch_ndim >= 0 && batch_ndim <= a_rank && batch_ndim <= b_rank,
        "nntile gemm: invalid batch_ndim");

    const int64_t a_k_begin = trans_a ? batch_ndim : (a_rank - ndim);
    const int64_t a_k_end = trans_a ? (batch_ndim + ndim) : a_rank;
    const int64_t b_k_begin = trans_b ? (b_rank - ndim) : batch_ndim;
    const int64_t b_k_end = trans_b ? b_rank : (batch_ndim + ndim);

    TORCH_CHECK(
        a_k_end - a_k_begin == ndim && b_k_end - b_k_begin == ndim,
        "nntile gemm: ndim does not fit operand ranks");

    for (int64_t k = 0; k < ndim; ++k)
    {
        TORCH_CHECK(
            a_shape[a_k_begin + k] == b_shape[b_k_begin + k],
            "nntile gemm: contraction dimension mismatch at axis ",
            k);
    }

    for (int64_t b = 0; b < batch_ndim; ++b)
    {
        TORCH_CHECK(
            a_shape[b] == b_shape[b],
            "nntile gemm: batch dimension mismatch at axis ",
            b);
    }
}

} // namespace

GemmMatrixLayout analyze_matrix_layout_for_nntile(const at::Tensor &tensor)
{
    return layout_from_prepared_2d(tensor);
}

GemmMatrixLayout analyze_batched_gemm_operand_layout(const at::Tensor &tensor)
{
    return layout_from_prepared_batched(tensor);
}

GemmMatrixLayout layout_from_nd_contiguous(const at::Tensor &tensor)
{
    GemmMatrixLayout layout;
    layout.gemm_shape = sizes_to_vector(tensor.sizes());
    layout.trans = false;
    layout.needs_copy = !tensor.is_contiguous();
    return layout;
}

std::vector<int64_t> gemm_output_shape_pytorch(
    const std::vector<int64_t> &a_shape,
    const std::vector<int64_t> &b_shape,
    const GemmParams &params)
{
    std::vector<nntile::Index> a_graph(a_shape.begin(), a_shape.end());
    std::vector<nntile::Index> b_graph(b_shape.begin(), b_shape.end());
    const auto out_graph = nntile::tensor::gemm_output_shape(
        a_graph,
        b_graph,
        params.trans_a,
        params.trans_b,
        static_cast<nntile::Index>(params.ndim),
        static_cast<nntile::Index>(params.batch_ndim));
    return std::vector<int64_t>(out_graph.begin(), out_graph.end());
}

PreparedGemmOperands prepare_mm_operands(const at::Tensor &a, const at::Tensor &b)
{
    at::Tensor a_use = a.is_contiguous() ? a : a.contiguous();
    at::Tensor b_use = b.is_contiguous() ? b : b.contiguous();
    // Re-analyze after densify: contiguous matrices may still be
    // row/column-contiguous views with an inferred transpose.
    GemmMatrixLayout a_layout = analyze_matrix_layout_for_nntile(a_use);
    GemmMatrixLayout b_layout = analyze_matrix_layout_for_nntile(b_use);

    PreparedGemmOperands prepared;
    require_gemm_layout(a_use, a_layout, "operand a");
    require_gemm_layout(b_use, b_layout, "operand b");
    prepared.a = a_use;
    prepared.b = b_use;

    prepared.a_gemm_shape = a_layout.gemm_shape;
    prepared.b_gemm_shape = b_layout.gemm_shape;
    prepared.params.trans_a = a_layout.trans;
    prepared.params.trans_b = b_layout.trans;
    prepared.params.ndim = 1;
    prepared.params.batch_ndim = 0;
    prepared.out_shape = gemm_output_shape_pytorch(
        prepared.a_gemm_shape,
        prepared.b_gemm_shape,
        prepared.params);
    return prepared;
}

PreparedGemmOperands prepare_bmm_operands(const at::Tensor &a, const at::Tensor &b)
{
    at::Tensor a_use = a.is_contiguous() ? a : a.contiguous();
    at::Tensor b_use = b.is_contiguous() ? b : b.contiguous();
    GemmMatrixLayout a_layout = analyze_batched_gemm_operand_layout(a_use);
    GemmMatrixLayout b_layout = analyze_batched_gemm_operand_layout(b_use);

    PreparedGemmOperands prepared;
    require_gemm_layout(a_use, a_layout, "operand a");
    require_gemm_layout(b_use, b_layout, "operand b");
    prepared.a = a_use;
    prepared.b = b_use;

    prepared.a_gemm_shape = a_layout.gemm_shape;
    prepared.b_gemm_shape = b_layout.gemm_shape;
    prepared.params.trans_a = a_layout.trans;
    prepared.params.trans_b = b_layout.trans;
    prepared.params.ndim = 1;
    prepared.params.batch_ndim = 1;
    prepared.out_shape = gemm_output_shape_pytorch(
        prepared.a_gemm_shape,
        prepared.b_gemm_shape,
        prepared.params);
    return prepared;
}

std::pair<int64_t, int64_t> infer_gemm_params(
    c10::IntArrayRef a_shape,
    c10::IntArrayRef b_shape)
{
    const int64_t a_rank = static_cast<int64_t>(a_shape.size());
    const int64_t b_rank = static_cast<int64_t>(b_shape.size());

    // Prefer torch.matmul semantics: (...batch, M, K) @ (...batch, K, N)
    // with a single contracted K (ndim=1). The greedy scan below can
    // over-contract when M==N (e.g. attention [B,H,S,D]@[B,H,D,S]).
    if (a_rank >= 2 && b_rank >= 2 && a_rank == b_rank)
    {
        const int64_t batch_ndim = a_rank - 2;
        bool prefix_ok = true;
        for (int64_t b = 0; b < batch_ndim; ++b)
        {
            if (a_shape[b] != b_shape[b])
            {
                prefix_ok = false;
                break;
            }
        }
        if (prefix_ok && a_shape[a_rank - 1] == b_shape[b_rank - 2])
        {
            return {1, batch_ndim};
        }
    }

    const int64_t max_batch = std::min(a_rank, b_rank);

    for (int64_t batch_ndim = 0; batch_ndim <= max_batch; ++batch_ndim)
    {
        bool batch_ok = true;
        for (int64_t b = 0; b < batch_ndim; ++b)
        {
            if (a_shape[b] != b_shape[b])
            {
                batch_ok = false;
                break;
            }
        }
        if (!batch_ok)
        {
            continue;
        }

        // Only contract one trailing K for matmul-like shapes once a
        // batch prefix matches (avoid M==N over-contraction).
        if (a_rank >= batch_ndim + 2 && b_rank >= batch_ndim + 2 &&
            a_shape[a_rank - 1] == b_shape[batch_ndim])
        {
            // b layout (...batch, K, N): K at batch_ndim when N follows.
            if (b_rank == batch_ndim + 2)
            {
                return {1, batch_ndim};
            }
        }

        int64_t ndim = 0;
        while (ndim < a_rank - batch_ndim && batch_ndim + ndim < b_rank &&
               a_shape[a_rank - 1 - ndim] == b_shape[batch_ndim + ndim])
        {
            ++ndim;
        }
        if (ndim == 1)
        {
            return {ndim, batch_ndim};
        }
    }

    TORCH_CHECK(
        false,
        "nntile gemm: no matching contraction dimensions between operands");
}

PreparedGemmOperands prepare_gemm_operands(
    const at::Tensor &a,
    const at::Tensor &b,
    int64_t ndim,
    int64_t batch_ndim,
    bool trans_a,
    bool trans_b)
{
    at::Tensor a_use = a.is_contiguous() ? a : a.contiguous();
    at::Tensor b_use = b.is_contiguous() ? b : b.contiguous();
    GemmMatrixLayout a_layout = layout_from_nd_contiguous(a_use);
    GemmMatrixLayout b_layout = layout_from_nd_contiguous(b_use);

    PreparedGemmOperands prepared;
    require_gemm_layout(a_use, a_layout, "operand a");
    require_gemm_layout(b_use, b_layout, "operand b");
    prepared.a = a_use;
    prepared.b = b_use;

    prepared.a_gemm_shape = a_layout.gemm_shape;
    prepared.b_gemm_shape = b_layout.gemm_shape;
    // Explicit transpose flags override stride-inferred layout.trans.
    prepared.params.trans_a = trans_a;
    prepared.params.trans_b = trans_b;
    prepared.params.ndim = ndim;
    prepared.params.batch_ndim = batch_ndim;
    validate_gemm_contraction(
        prepared.a_gemm_shape,
        prepared.b_gemm_shape,
        ndim,
        batch_ndim,
        prepared.params.trans_a,
        prepared.params.trans_b);
    prepared.out_shape = gemm_output_shape_pytorch(
        prepared.a_gemm_shape,
        prepared.b_gemm_shape,
        prepared.params);
    return prepared;
}

PreparedGemmOperands prepare_gemm_operands_inferred(
    const at::Tensor &a,
    const at::Tensor &b)
{
    const auto [ndim, batch_ndim] =
        infer_gemm_params(a.sizes(), b.sizes());
    return prepare_gemm_operands(a, b, ndim, batch_ndim);
}

PreparedGemmOperands prepare_linear_operands(
    const at::Tensor &input,
    const at::Tensor &weight)
{
    GemmMatrixLayout input_layout;
    if (input.dim() == 1)
    {
        input_layout.gemm_shape = {1, input.size(0)};
        input_layout.trans = false;
        input_layout.needs_copy = !input.is_contiguous();
    }
    else
    {
        input_layout.gemm_shape = sizes_to_vector(input.sizes());
        input_layout.trans = false;
        input_layout.needs_copy = !input.is_contiguous();
    }

    GemmMatrixLayout weight_layout = analyze_matrix_layout_for_nntile(weight);

    PreparedGemmOperands prepared;
    require_gemm_layout(input, input_layout, "input");
    require_gemm_layout(weight, weight_layout, "weight");
    prepared.a = input;
    prepared.b = weight;

    prepared.a_gemm_shape = input_layout.gemm_shape;
    prepared.b_gemm_shape = weight_layout.gemm_shape;
    prepared.params.trans_a = input_layout.trans;
    prepared.params.trans_b = !weight_layout.trans;
    prepared.params.ndim = 1;
    prepared.params.batch_ndim = 0;
    prepared.out_shape = gemm_output_shape_pytorch(
        prepared.a_gemm_shape,
        prepared.b_gemm_shape,
        prepared.params);
    return prepared;
}

GemmParams infer_gemm_backward_grad_a_params(
    const GemmParams &forward,
    int64_t b_rank)
{
    GemmParams params;
    params.trans_a = false;
    params.trans_b = !forward.trans_b;
    params.ndim = b_rank - forward.batch_ndim - forward.ndim;
    params.batch_ndim = forward.batch_ndim;
    return params;
}

GemmParams infer_gemm_backward_grad_b_params(
    const GemmParams &forward,
    int64_t a_rank)
{
    GemmParams params;
    params.trans_a = !forward.trans_a;
    params.trans_b = false;
    params.ndim = a_rank - forward.batch_ndim - forward.ndim;
    params.batch_ndim = forward.batch_ndim;
    return params;
}

GemmParams infer_mm_backward_grad_a_params(const GemmParams &forward)
{
    return infer_gemm_backward_grad_a_params(forward, 2);
}

GemmParams infer_mm_backward_grad_b_params(const GemmParams &forward)
{
    return infer_gemm_backward_grad_b_params(forward, 2);
}

GemmParams infer_linear_backward_grad_input_params(const GemmParams &forward)
{
    GemmParams params;
    params.trans_a = false;
    params.trans_b = !forward.trans_b;
    params.ndim = forward.ndim;
    params.batch_ndim = forward.batch_ndim;
    return params;
}

GemmParams infer_linear_backward_grad_weight_params(const GemmParams &forward)
{
    GemmParams params;
    params.trans_a = true;
    params.trans_b = false;
    params.ndim = forward.ndim;
    params.batch_ndim = forward.batch_ndim;
    return params;
}

std::vector<int64_t> pytorch_sizes_vector(c10::IntArrayRef sizes)
{
    return std::vector<int64_t>(sizes.begin(), sizes.end());
}

} // namespace torch_nntile
