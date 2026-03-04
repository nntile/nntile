/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/norm.cc
 * TensorGraph norm operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/norm.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/norm.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_norm(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y)
{
    auto& x_t = runtime.get_tensor<T>(x);
    auto& y_t = runtime.get_tensor<T>(y);
    nntile::tensor::norm<T>(alpha, x_t, beta, y_t);
}

} // namespace

void norm(TensorGraph::TensorNode* x, TensorGraph::TensorNode* y,
          Scalar alpha, Scalar beta)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument("norm: input tensors must be non-null");
    }
    if(x == y)
    {
        throw std::invalid_argument(
            "norm: x and y must be distinct tensors");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "norm: tensors must belong to the same graph");
    }
    if(y->ndim() != 0)
    {
        throw std::invalid_argument(
            "norm: output tensor must be scalar (shape [])");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "norm: input and output tensors must have the same dtype");
    }

    auto op = std::make_shared<TensorNormOp>(x, y, alpha, beta);
    x->graph()->add_op(op);
}

void TensorNormOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_norm<nntile::fp32_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP32_FAST_TF32:
            run_norm<nntile::fp32_fast_tf32_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP32_FAST_FP16:
            run_norm<nntile::fp32_fast_fp16_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP32_FAST_BF16:
            run_norm<nntile::fp32_fast_bf16_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP64:
            run_norm<nntile::fp64_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP16:
            run_norm<nntile::fp16_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::BF16:
            run_norm<nntile::bf16_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                "INT64/BOOL data type not supported for norm operation");
        default:
            throw std::runtime_error("Unsupported data type for norm");
    }
}

} // namespace nntile::graph::tensor
