/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sgd_step.cc
 * TensorGraph sgd_step operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sgd_step.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sgd_step.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_sgd_step(TensorGraph::Runtime& runtime,
                  Index num_iter, Scalar momentum, Scalar lr,
                  Scalar weight_decay, Scalar dampening, bool nesterov,
                  TensorGraph::TensorNode* grad,
                  TensorGraph::TensorNode* velocity,
                  TensorGraph::TensorNode* p)
{
    auto& grad_t = runtime.get_tensor<T>(grad);
    auto& velocity_t = runtime.get_tensor<T>(velocity);
    auto& p_t = runtime.get_tensor<T>(p);
    nntile::tensor::sgd_step<T>(num_iter, momentum, lr, weight_decay,
                                dampening, nesterov, grad_t, velocity_t, p_t);
}

} // namespace

void sgd_step(Index num_iter, Scalar momentum, Scalar lr,
              Scalar weight_decay, Scalar dampening, bool nesterov,
              TensorGraph::TensorNode* grad,
              TensorGraph::TensorNode* velocity,
              TensorGraph::TensorNode* p)
{
    sgd_step(std::make_shared<Index>(num_iter), momentum, lr,
             weight_decay, dampening, nesterov, grad, velocity, p);
}

void sgd_step(std::shared_ptr<Index> num_iter, Scalar momentum, Scalar lr,
              Scalar weight_decay, Scalar dampening, bool nesterov,
              TensorGraph::TensorNode* grad,
              TensorGraph::TensorNode* velocity,
              TensorGraph::TensorNode* p)
{
    if(grad == nullptr || velocity == nullptr || p == nullptr)
        throw std::invalid_argument("sgd_step: tensors must be non-null");
    if(grad->graph() != velocity->graph() || velocity->graph() != p->graph())
        throw std::invalid_argument("sgd_step: tensors must belong to same graph");
    if(grad->dtype() != velocity->dtype() || velocity->dtype() != p->dtype())
        throw std::invalid_argument("sgd_step: tensors must have same dtype");
    auto op = std::make_shared<TensorSgdStepOp>(
        std::move(num_iter), momentum, lr, weight_decay, dampening, nesterov,
        grad, velocity, p);
    p->graph()->add_op(op);
}

void TensorSgdStepOp::execute(TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(grad);
    switch(dtype)
    {
        case DataType::FP32:
            run_sgd_step<nntile::fp32_t>(runtime, *num_iter, momentum, lr,
                weight_decay, dampening, nesterov, grad, velocity, p);
            break;
        case DataType::FP32_FAST_TF32:
            run_sgd_step<nntile::fp32_fast_tf32_t>(runtime, *num_iter, momentum, lr,
                weight_decay, dampening, nesterov, grad, velocity, p);
            break;
        case DataType::FP32_FAST_FP16:
            run_sgd_step<nntile::fp32_fast_fp16_t>(runtime, *num_iter, momentum, lr,
                weight_decay, dampening, nesterov, grad, velocity, p);
            break;
        case DataType::FP32_FAST_BF16:
            run_sgd_step<nntile::fp32_fast_bf16_t>(runtime, *num_iter, momentum, lr,
                weight_decay, dampening, nesterov, grad, velocity, p);
            break;
        case DataType::FP64:
            run_sgd_step<nntile::fp64_t>(runtime, *num_iter, momentum, lr,
                weight_decay, dampening, nesterov, grad, velocity, p);
            break;
        case DataType::FP16:
            run_sgd_step<nntile::fp16_t>(runtime, *num_iter, momentum, lr,
                weight_decay, dampening, nesterov, grad, velocity, p);
            break;
        case DataType::BF16:
            run_sgd_step<nntile::bf16_t>(runtime, *num_iter, momentum, lr,
                weight_decay, dampening, nesterov, grad, velocity, p);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for sgd_step");
        default:
            throw std::runtime_error("Unsupported data type for sgd_step");
    }
}

} // namespace nntile::graph::tensor
