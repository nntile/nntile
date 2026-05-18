/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/adamw_step.cc
 * TileGraph adamw step operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/adamw_step.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/adamw_step.hh>

namespace nntile::graph::tile_graph
{

namespace
{

template<typename T>
void run_adamw(
    TileGraph::Runtime& runtime,
    Index num_iter,
    Scalar beta_1,
    Scalar beta_2,
    Scalar eps,
    Scalar lr,
    Scalar weight_decay,
    TileGraph::TileNode* g,
    TileGraph::TileNode* m,
    TileGraph::TileNode* v,
    TileGraph::TileNode* pn)
{
    nntile::tile::adamw_step<T>(num_iter, beta_1, beta_2, eps, lr, weight_decay,
        runtime.get_tile<T>(g), runtime.get_tile<T>(m), runtime.get_tile<T>(v),
        runtime.get_tile<T>(pn));
}

} // namespace

TileAdamwStepOp::TileAdamwStepOp(const std::shared_ptr<Index>& step_iter_,
    bool bump_after_, Scalar beta_1_, Scalar beta_2_, Scalar eps_, Scalar lr_,
    Scalar weight_decay_, TileGraph::TileNode* grad_,
    TileGraph::TileNode* first_moment_, TileGraph::TileNode* second_moment_,
    TileGraph::TileNode* p_)
    : step_iter(step_iter_),
      bump_after(bump_after_),
      beta_1(beta_1_),
      beta_2(beta_2_),
      eps(eps_),
      lr(lr_),
      weight_decay(weight_decay_),
      grad(grad_),
      first_moment(first_moment_),
      second_moment(second_moment_),
      p(p_)
{
    if(!step_iter)
    {
        throw std::invalid_argument("TileAdamwStepOp: step_iter must be non-null");
    }
    if(grad == nullptr || first_moment == nullptr || second_moment == nullptr ||
        p == nullptr)
    {
        throw std::invalid_argument("TileAdamwStepOp: tile pointers must be non-null");
    }
    if(grad->graph() != first_moment->graph() ||
        grad->graph() != second_moment->graph() || grad->graph() != p->graph())
    {
        throw std::invalid_argument(
            "TileAdamwStepOp: tiles must belong to the same graph");
    }
    if(grad->dtype() != first_moment->dtype() ||
        grad->dtype() != second_moment->dtype() || grad->dtype() != p->dtype())
    {
        throw std::invalid_argument("TileAdamwStepOp: dtype mismatch");
    }
    if(grad->shape() != first_moment->shape() ||
        grad->shape() != second_moment->shape() || grad->shape() != p->shape())
    {
        throw std::invalid_argument("TileAdamwStepOp: shape mismatch");
    }
    inputs_ = {grad, first_moment, second_moment, p};
    outputs_ = {first_moment, second_moment, p};
}

void adamw_step(const std::shared_ptr<Index>& step_iter, bool bump_after,
    Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    TileGraph::TileNode* grad, TileGraph::TileNode* first_moment,
    TileGraph::TileNode* second_moment, TileGraph::TileNode* p)
{
    auto op = std::make_shared<TileAdamwStepOp>(step_iter, bump_after, beta_1,
        beta_2, eps, lr, weight_decay, grad, first_moment, second_moment, p);
    grad->graph()->add_op(op);
}

void TileAdamwStepOp::execute(TileGraph::Runtime& runtime) const
{
    const Index cur = *step_iter;
    const DataType dtype = runtime.get_dtype(grad);
    switch(dtype)
    {
        case DataType::FP32:
            run_adamw<nntile::fp32_t>(runtime, cur, beta_1, beta_2, eps, lr,
                weight_decay, grad, first_moment, second_moment, p);
            break;
        case DataType::FP32_FAST_TF32:
            run_adamw<nntile::fp32_fast_tf32_t>(runtime, cur, beta_1, beta_2,
                eps, lr, weight_decay, grad, first_moment, second_moment, p);
            break;
        case DataType::FP32_FAST_FP16:
            run_adamw<nntile::fp32_fast_fp16_t>(runtime, cur, beta_1, beta_2,
                eps, lr, weight_decay, grad, first_moment, second_moment, p);
            break;
        case DataType::FP32_FAST_BF16:
            run_adamw<nntile::fp32_fast_bf16_t>(runtime, cur, beta_1, beta_2,
                eps, lr, weight_decay, grad, first_moment, second_moment, p);
            break;
        case DataType::FP64:
            run_adamw<nntile::fp64_t>(runtime, cur, beta_1, beta_2, eps, lr,
                weight_decay, grad, first_moment, second_moment, p);
            break;
        case DataType::FP16:
            run_adamw<nntile::fp16_t>(runtime, cur, beta_1, beta_2, eps, lr,
                weight_decay, grad, first_moment, second_moment, p);
            break;
        case DataType::BF16:
            run_adamw<nntile::bf16_t>(runtime, cur, beta_1, beta_2, eps, lr,
                weight_decay, grad, first_moment, second_moment, p);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for TileAdamwStepOp");
        default:
            throw std::runtime_error("Unsupported data type for TileAdamwStepOp");
    }
    if(bump_after)
    {
        ++(*step_iter);
    }
}

} // namespace nntile::graph::tile_graph
