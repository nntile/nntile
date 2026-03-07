/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/optim/adam.hh
 * Adam optimizer (like PyTorch's torch.optim.Adam).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/optim/optimizer.hh>

namespace nntile::graph::optim
{

class Adam : public Optimizer
{
    Scalar lr_;
    Scalar beta_1_;
    Scalar beta_2_;
    Scalar eps_;
    Scalar weight_decay_;

public:
    Adam(NNGraph* graph,
         module::Module* module,
         Scalar lr = 0.001,
         Scalar beta_1 = 0.9,
         Scalar beta_2 = 0.999,
         Scalar eps = 1e-8,
         Scalar weight_decay = 0.0);

    void step() override;

    void save_config(const std::string& path) const override;
    void load_config(const std::string& path) override;

    std::string repr() const override;

    Scalar lr() const { return lr_; }
    Scalar beta_1() const { return beta_1_; }
    Scalar beta_2() const { return beta_2_; }
    Scalar eps() const { return eps_; }
    Scalar weight_decay() const { return weight_decay_; }
};

} // namespace nntile::graph::optim
