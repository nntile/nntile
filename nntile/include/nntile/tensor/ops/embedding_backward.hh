/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/ops/embedding_backward.hh
 * TensorGraph embedding_backward: vocab = beta*vocab + alpha*scatter(embed)
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/tensor/graph.hh>

namespace nntile
{
struct LoweringContext;
}

namespace nntile::tensor
{

//! Embedding backward with alpha/beta
/*! beta=0: overwrite each vocab tile (kernel zeros then accumulates;
 *  STARPU_W on first write per tile).
 *  beta=1: accumulate only (vocab += alpha*scatter).
 *  Tiling across vocab_size (axis 0) is disallowed.
 */
struct TensorEmbeddingBackwardOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* index = nullptr;
    TensorGraph::TensorNode* embed = nullptr;
    TensorGraph::TensorNode* vocab = nullptr;
    Index axis;
    Scalar alpha;
    Scalar beta;
    int redux;

    TensorEmbeddingBackwardOp() = default;
    TensorEmbeddingBackwardOp(TensorGraph::TensorNode* index_,
                             TensorGraph::TensorNode* embed_,
                             TensorGraph::TensorNode* vocab_,
                             Index axis_,
                             Scalar alpha_,
                             Scalar beta_,
                             int redux_)
        : index(index_), embed(embed_), vocab(vocab_), axis(axis_),
          alpha(alpha_), beta(beta_), redux(redux_)
    {
        inputs_ = {index, embed, vocab};
        outputs_ = {vocab};
    }

    std::string op_name() const override { return "EMBEDDING_BACKWARD"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorEmbeddingBackwardOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

//! vocab = beta*vocab + alpha*scatter(embed, index); beta must be 0 or 1
void embedding_backward(TensorGraph::TensorNode* index,
                        TensorGraph::TensorNode* embed,
                        TensorGraph::TensorNode* vocab,
                        Index axis,
                        Scalar alpha,
                        Scalar beta,
                        int redux);

} // namespace nntile::tensor
