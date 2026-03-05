/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/adam_step.cc
 * TensorGraph adam_step operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/adam_step.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/adam_step.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_adam_step(TensorGraph::Runtime& runtime,
                   Index num_iter, Scalar beta_1, Scalar beta_2,
                   Scalar eps, Scalar lr, Scalar weight_decay,
                   TensorGraph::TensorNode* grad,
                   TensorGraph::TensorNode* first_moment,
                   TensorGraph::TensorNode* second_moment,
                   TensorGraph::TensorNode* p)
{
    auto& grad_t = runtime.get_tensor<T>(grad);
    auto& first_moment_t = runtime.get_tensor<T>(first_moment);
    auto& second_moment_t = runtime.get_tensor<T>(second_moment);
    auto& p_t = runtime.get_tensor<T>(p);
    nntile::tensor::adam_step<T>(num_iter, beta_1, beta_2, eps, lr,
                                 weight_decay, grad_t, first_moment_t,
                                 second_moment_t, p_t);
}

} // namespace

void adam_step(Index num_iter, Scalar beta_1, Scalar beta_2,
               Scalar eps, Scalar lr, Scalar weight_decay,
               TensorGraph::TensorNode* grad,
               TensorGraph::TensorNode* first_moment,
               TensorGraph::TensorNode* second_moment,
               TensorGraph::TensorNode* p)
{
    if(grad == nullptr || first_moment == nullptr ||
       second_moment == nullptr || p == nullptr)
        throw std::invalid_argument("adam_step: tensors must be non-null");
    if(grad->graph() != first_moment->graph() ||
       first_moment->graph() != second_moment->graph() ||
       second_moment->graph() != p->graph())
        throw std::invalid_argument("adam_step: tensors must belong to same graph");
    if(grad->dtype() != first_moment->dtype() ||
       first_moment->dtype() != second_moment->dtype() ||
       second_moment->dtype() != p->dtype())
        throw std::invalid_argument("adam_step: tensors must have same dtype");
    auto op = std::make_shared<TensorAdamStepOp>(
        num_iter, beta_1, beta_2, eps, lr, weight_decay,
        grad, first_moment, second_moment, p);
    p->graph()->add_op(op);
}

void TensorAdamStepOp::execute(TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(grad);
    switch(dtype)
    {
        case DataType::FP32:
            run_adam_step<nntile::fp32_t>(runtime, num_iter, beta_1, beta_2,
                eps, lr, weight_decay, grad, first_moment, second_moment, p);
            break;
        case DataType::FP32_FAST_TF32:
            run_adam_step<nntile::fp32_fast_tf32_t>(runtime, num_iter, beta_1, beta_2,
                eps, lr, weight_decay, grad, first_moment, second_moment, p);
            break;
        case DataType::FP32_FAST_FP16:
            run_adam_step<nntile::fp32_fast_fp16_t>(runtime, num_iter, beta_1, beta_2,
                eps, lr, weight_decay, grad, first_moment, second_moment, p);
            break;
        case DataType::FP32_FAST_BF16:
            run_adam_step<nntile::fp32_fast_bf16_t>(runtime, num_iter, beta_1, beta_2,
                eps, lr, weight_decay, grad, first_moment, second_moment, p);
            break;
        case DataType::FP64:
            run_adam_step<nntile::fp64_t>(runtime, num_iter, beta_1, beta_2,
                eps, lr, weight_decay, grad, first_moment, second_moment, p);
            break;
        case DataType::FP16:
            run_adam_step<nntile::fp16_t>(runtime, num_iter, beta_1, beta_2,
                eps, lr, weight_decay, grad, first_moment, second_moment, p);
            break;
        case DataType::BF16:
            run_adam_step<nntile::bf16_t>(runtime, num_iter, beta_1, beta_2,
                eps, lr, weight_decay, grad, first_moment, second_moment, p);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for adam_step");
        default:
            throw std::runtime_error("Unsupported data type for adam_step");
    }
}

} // namespace nntile::graph::tensor
