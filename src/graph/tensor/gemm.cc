/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/gemm.cc
 * TensorGraph GEMM operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/gemm.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/constants.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/gemm.hh"

namespace nntile::graph::tensor
{

std::vector<Index> gemm_output_shape(
    const std::vector<Index>& a_shape,
    const std::vector<Index>& b_shape,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    Index a_ndim = static_cast<Index>(a_shape.size());
    Index b_ndim = static_cast<Index>(b_shape.size());

    std::vector<Index> output_shape;
    output_shape.reserve(a_ndim + b_ndim - 2 * ndim);

    Index a_batch_start = a_ndim - batch_ndim;
    Index b_batch_start = b_ndim - batch_ndim;

    Index a_m_begin = trans_a ? ndim : 0;
    Index a_m_end = trans_a ? a_batch_start : a_batch_start - ndim;
    Index b_n_begin = trans_b ? 0 : ndim;
    Index b_n_end = trans_b ? b_batch_start - ndim : b_batch_start;

    output_shape.insert(output_shape.end(),
                        a_shape.begin() + a_m_begin,
                        a_shape.begin() + a_m_end);
    output_shape.insert(output_shape.end(),
                        b_shape.begin() + b_n_begin,
                        b_shape.begin() + b_n_end);
    output_shape.insert(output_shape.end(),
                        a_shape.begin() + a_batch_start,
                        a_shape.end());

    return output_shape;
}

namespace
{
constexpr Scalar gemm_new_output_beta = 0.0;

//! Validate A-B, A-C, B-C shapes and merge axes (inline check+merge per dimension).
void validate_gemm_shape_and_merge(
    TensorGraph::TensorNode* a,
    TensorGraph::TensorNode* b,
    TensorGraph::TensorNode* c,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    Index a_ndim = a->ndim();
    Index b_ndim = b->ndim();
    Index c_ndim = c->ndim();
    if(a_ndim < ndim + batch_ndim)
    {
        throw std::invalid_argument(
            "gemm: A must have ndim >= ndim + batch_ndim (" +
            std::to_string(a_ndim) + " vs " +
            std::to_string(ndim + batch_ndim) + ")");
    }
    if(b_ndim < ndim + batch_ndim)
    {
        throw std::invalid_argument(
            "gemm: B must have ndim >= ndim + batch_ndim (" +
            std::to_string(b_ndim) + " vs " +
            std::to_string(ndim + batch_ndim) + ")");
    }
    Index a_batch_start = a_ndim - batch_ndim;
    Index b_batch_start = b_ndim - batch_ndim;
    Index a_m_begin = trans_a ? ndim : 0;
    Index a_m_end = trans_a ? a_batch_start : a_batch_start - ndim;
    Index a_k_begin = trans_a ? 0 : a_m_end;
    Index b_k_begin = trans_b ? (b_batch_start - ndim) : 0;
    Index b_n_begin = trans_b ? 0 : ndim;
    Index b_n_end = trans_b ? b_batch_start - ndim : b_batch_start;
    Index num_m = a_m_end - a_m_begin;
    Index num_n = b_n_end - b_n_begin;
    Index c_batch_start = num_m + num_n;
    if(c_ndim != c_batch_start + batch_ndim)
    {
        throw std::invalid_argument(
            "gemm: C ndim must equal num_m + num_n + batch_ndim (" +
            std::to_string(c_ndim) + " vs " +
            std::to_string(c_batch_start + batch_ndim) + ")");
    }
    // A-B: contracted (K) dimensions
    for(Index i = 0; i < ndim; ++i)
    {
        if(a->shape()[a_k_begin + i] != b->shape()[b_k_begin + i])
        {
            throw std::invalid_argument(
                "gemm: contracted dimension " + std::to_string(i) +
                " must match (A: " + std::to_string(a->shape()[a_k_begin + i]) +
                " vs B: " + std::to_string(b->shape()[b_k_begin + i]) + ")");
        }
        merge_axis(a->mutable_axes()[a_k_begin + i],
                   b->mutable_axes()[b_k_begin + i]);
    }
    // A-C: M dimensions
    for(Index i = 0; i < num_m; ++i)
    {
        if(a->shape()[a_m_begin + i] != c->shape()[i])
        {
            throw std::invalid_argument(
                "gemm: M dimension " + std::to_string(i) +
                " must match (A: " + std::to_string(a->shape()[a_m_begin + i]) +
                " vs C: " + std::to_string(c->shape()[i]) + ")");
        }
        merge_axis(a->mutable_axes()[a_m_begin + i], c->mutable_axes()[i]);
    }
    // B-C: N dimensions
    for(Index i = 0; i < num_n; ++i)
    {
        if(b->shape()[b_n_begin + i] != c->shape()[num_m + i])
        {
            throw std::invalid_argument(
                "gemm: N dimension " + std::to_string(i) +
                " must match (B: " + std::to_string(b->shape()[b_n_begin + i]) +
                " vs C: " + std::to_string(c->shape()[num_m + i]) + ")");
        }
        merge_axis(b->mutable_axes()[b_n_begin + i],
                   c->mutable_axes()[num_m + i]);
    }
    // A-B-C: batch dimensions
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(a->shape()[a_batch_start + i] != c->shape()[c_batch_start + i])
        {
            throw std::invalid_argument(
                "gemm: batch dimension " + std::to_string(i) +
                " must match (A: " + std::to_string(a->shape()[a_batch_start + i]) +
                " vs C: " + std::to_string(c->shape()[c_batch_start + i]) + ")");
        }
        merge_axis(a->mutable_axes()[a_batch_start + i],
                   c->mutable_axes()[c_batch_start + i]);
        if(b->shape()[b_batch_start + i] != c->shape()[c_batch_start + i])
        {
            throw std::invalid_argument(
                "gemm: batch dimension " + std::to_string(i) +
                " must match (B: " + std::to_string(b->shape()[b_batch_start + i]) +
                " vs C: " + std::to_string(c->shape()[c_batch_start + i]) + ")");
        }
        merge_axis(b->mutable_axes()[b_batch_start + i],
                   c->mutable_axes()[c_batch_start + i]);
    }
}

template<typename T>
void run_gemm(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    bool trans_a, bool trans_b,
    Index ndim, Index batch_ndim,
    TensorGraph::TensorNode* a,
    TensorGraph::TensorNode* b,
    TensorGraph::TensorNode* c)
{
    auto& a_t = runtime.get_tensor<T>(a);
    auto& b_t = runtime.get_tensor<T>(b);
    auto& c_t = runtime.get_tensor<T>(c);

    const auto trans_a_op = trans_a ?
        nntile::TransOp(nntile::TransOp::Trans) :
        nntile::TransOp(nntile::TransOp::NoTrans);
    const auto trans_b_op = trans_b ?
        nntile::TransOp(nntile::TransOp::Trans) :
        nntile::TransOp(nntile::TransOp::NoTrans);

    nntile::tensor::gemm<T>(
        alpha, trans_a_op, a_t, trans_b_op, b_t,
        beta, c_t, ndim, batch_ndim, 0);
}

} // namespace

TensorGraph::TensorNode* gemm(
    TensorGraph::TensorNode* a,
    TensorGraph::TensorNode* b,
    const std::string& output_name,
    Scalar alpha,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    if(a == nullptr || b == nullptr)
    {
        throw std::invalid_argument("gemm: input tensors must be non-null");
    }
    if(a->graph() != b->graph())
    {
        throw std::invalid_argument(
            "gemm: input tensors must belong to the same graph");
    }
    if(a->dtype() != b->dtype())
    {
        throw std::invalid_argument(
            "gemm: input tensors must have the same dtype");
    }
    if(a->ndim() < ndim + batch_ndim)
    {
        throw std::invalid_argument(
            "gemm: A must have ndim >= ndim + batch_ndim (" +
            std::to_string(a->ndim()) + " vs " +
            std::to_string(ndim + batch_ndim) + ")");
    }
    if(b->ndim() < ndim + batch_ndim)
    {
        throw std::invalid_argument(
            "gemm: B must have ndim >= ndim + batch_ndim (" +
            std::to_string(b->ndim()) + " vs " +
            std::to_string(ndim + batch_ndim) + ")");
    }

    std::vector<Index> output_shape = gemm_output_shape(
        a->shape(), b->shape(), trans_a, trans_b, ndim, batch_ndim);

    TensorGraph::TensorNode* output = a->graph()->data(
        std::move(output_shape),
        output_name,
        a->dtype());

    gemm(a, b, output, alpha, gemm_new_output_beta, trans_a, trans_b, ndim,
         batch_ndim);

    return output;
}

void gemm(
    TensorGraph::TensorNode* a,
    TensorGraph::TensorNode* b,
    TensorGraph::TensorNode* c,
    Scalar alpha,
    Scalar beta,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    if(a == nullptr || b == nullptr || c == nullptr)
    {
        throw std::invalid_argument("gemm: input tensors must be non-null");
    }
    if(a->graph() != b->graph() || a->graph() != c->graph())
    {
        throw std::invalid_argument(
            "gemm: input tensors must belong to the same graph");
    }
    if(a->dtype() != b->dtype() || a->dtype() != c->dtype())
    {
        throw std::invalid_argument(
            "gemm: input tensors must have the same dtype");
    }

    validate_gemm_shape_and_merge(a, b, c, trans_a, trans_b, ndim, batch_ndim);

    auto op = std::make_shared<TensorGemmOp>(
        a, b, c, alpha, beta, trans_a, trans_b, ndim, batch_ndim);

    a->graph()->add_op(op);
}

void TensorGemmOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(a);

    switch(dtype)
    {
        case DataType::FP32:
            run_gemm<nntile::fp32_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim,
                a, b, c);
            break;
        case DataType::FP32_FAST_TF32:
            run_gemm<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim,
                a, b, c);
            break;
        case DataType::FP32_FAST_FP16:
            run_gemm<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim,
                a, b, c);
            break;
        case DataType::FP32_FAST_BF16:
            run_gemm<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim,
                a, b, c);
            break;
        case DataType::FP64:
            run_gemm<nntile::fp64_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim,
                a, b, c);
            break;
        case DataType::FP16:
            run_gemm<nntile::fp16_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim,
                a, b, c);
            break;
        case DataType::BF16:
            run_gemm<nntile::bf16_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim,
                a, b, c);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gemm operation");
        default:
            throw std::runtime_error("Unsupported data type for gemm");
    }
}

} // namespace nntile::graph::tensor
