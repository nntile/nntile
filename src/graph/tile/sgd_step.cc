/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/sgd_step.cc
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/sgd_step.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/sgd_step.hh>

namespace nntile::graph::tile_graph
{

namespace
{

template<typename T>
void run_sgd(
    TileGraph::Runtime& runtime,
    Index num_iter,
    Scalar momentum,
    Scalar lr,
    Scalar weight_decay,
    Scalar dampening,
    bool nesterov,
    TileGraph::TileNode* g,
    TileGraph::TileNode* vel,
    TileGraph::TileNode* pn)
{
    nntile::tile::sgd_step<T>(num_iter, momentum, lr, weight_decay, dampening,
        nesterov, runtime.get_tile<T>(g), runtime.get_tile<T>(vel),
        runtime.get_tile<T>(pn));
}

} // namespace

TileSgdStepOp::TileSgdStepOp(const std::shared_ptr<Index>& step_iter_,
    bool bump_after_, Scalar momentum_, Scalar lr_, Scalar weight_decay_,
    Scalar dampening_, bool nesterov_, TileGraph::TileNode* grad_,
    TileGraph::TileNode* velocity_, TileGraph::TileNode* p_)
    : step_iter(step_iter_),
      bump_after(bump_after_),
      momentum(momentum_),
      lr(lr_),
      weight_decay(weight_decay_),
      dampening(dampening_),
      nesterov(nesterov_),
      grad(grad_),
      velocity(velocity_),
      p(p_)
{
    if(!step_iter)
    {
        throw std::invalid_argument("TileSgdStepOp: step_iter must be non-null");
    }
    if(grad == nullptr || velocity == nullptr || p == nullptr)
    {
        throw std::invalid_argument("TileSgdStepOp: tile pointers must be non-null");
    }
    if(grad->graph() != velocity->graph() || grad->graph() != p->graph())
    {
        throw std::invalid_argument(
            "TileSgdStepOp: tiles must belong to the same graph");
    }
    if(grad->dtype() != velocity->dtype() || grad->dtype() != p->dtype())
    {
        throw std::invalid_argument("TileSgdStepOp: dtype mismatch");
    }
    if(grad->shape() != velocity->shape() || grad->shape() != p->shape())
    {
        throw std::invalid_argument("TileSgdStepOp: shape mismatch");
    }
    inputs_ = {grad, velocity, p};
    outputs_ = {velocity, p};
}

void sgd_step(const std::shared_ptr<Index>& step_iter, bool bump_after,
    Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening,
    bool nesterov, TileGraph::TileNode* grad, TileGraph::TileNode* velocity,
    TileGraph::TileNode* p)
{
    auto op = std::make_shared<TileSgdStepOp>(step_iter, bump_after, momentum,
        lr, weight_decay, dampening, nesterov, grad, velocity, p);
    grad->graph()->add_op(op);
}

void TileSgdStepOp::execute(TileGraph::Runtime& runtime) const
{
    const Index cur = *step_iter;
    const DataType dtype = runtime.get_dtype(grad);
    switch(dtype)
    {
        case DataType::FP32:
            run_sgd<nntile::fp32_t>(runtime, cur, momentum, lr, weight_decay,
                dampening, nesterov, grad, velocity, p);
            break;
        case DataType::FP32_FAST_TF32:
            run_sgd<nntile::fp32_fast_tf32_t>(runtime, cur, momentum, lr,
                weight_decay, dampening, nesterov, grad, velocity, p);
            break;
        case DataType::FP32_FAST_FP16:
            run_sgd<nntile::fp32_fast_fp16_t>(runtime, cur, momentum, lr,
                weight_decay, dampening, nesterov, grad, velocity, p);
            break;
        case DataType::FP32_FAST_BF16:
            run_sgd<nntile::fp32_fast_bf16_t>(runtime, cur, momentum, lr,
                weight_decay, dampening, nesterov, grad, velocity, p);
            break;
        case DataType::FP64:
            run_sgd<nntile::fp64_t>(runtime, cur, momentum, lr, weight_decay,
                dampening, nesterov, grad, velocity, p);
            break;
        case DataType::FP16:
            run_sgd<nntile::fp16_t>(runtime, cur, momentum, lr, weight_decay,
                dampening, nesterov, grad, velocity, p);
            break;
        case DataType::BF16:
            run_sgd<nntile::bf16_t>(runtime, cur, momentum, lr, weight_decay,
                dampening, nesterov, grad, velocity, p);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for TileSgdStepOp");
        default:
            throw std::runtime_error("Unsupported data type for TileSgdStepOp");
    }
    if(bump_after)
    {
        ++(*step_iter);
    }
}

} // namespace nntile::graph::tile_graph
