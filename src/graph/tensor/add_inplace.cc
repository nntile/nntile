/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/add_inplace.cc
 * TensorGraph add_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/add_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/add_inplace.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_add_inplace(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y)
{
    auto& x_t = runtime.get_tensor<T>(x);
    auto& y_t = runtime.get_tensor<T>(y);
    nntile::tensor::add_inplace<T>(alpha, x_t, beta, y_t);
}

} // namespace

void add_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* x,
    Scalar beta,
    TensorGraph::TensorNode* y)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument(
            "add_inplace: input tensors must be non-null");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "add_inplace: input tensors must belong to the same graph");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "add_inplace: input tensors must have the same dtype");
    }
    if(x == y)
    {
        throw std::invalid_argument(
            "add_inplace: x and y must be distinct tensors");
    }
    validate_same_shape_and_merge(x, y, "add_inplace");

    auto op = std::make_shared<TensorAddInplaceOp>(x, y, alpha, beta);
    x->graph()->add_op(op);
}

void TensorAddInplaceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_add_inplace<nntile::fp32_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP32_FAST_TF32:
            run_add_inplace<nntile::fp32_fast_tf32_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP32_FAST_FP16:
            run_add_inplace<nntile::fp32_fast_fp16_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP32_FAST_BF16:
            run_add_inplace<nntile::fp32_fast_bf16_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP64:
            run_add_inplace<nntile::fp64_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP16:
            run_add_inplace<nntile::fp16_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::BF16:
            run_add_inplace<nntile::bf16_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for add_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for add_inplace");
    }
}

} // namespace nntile::graph::tensor
