/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/optim/sgd.hh
 * SGD optimizer with optional momentum (like PyTorch's torch.optim.SGD).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/optim/optimizer.hh>

namespace nntile::graph::optim
{

class SGD : public Optimizer
{
    Scalar lr_;
    Scalar momentum_;
    Scalar weight_decay_;
    Scalar dampening_;
    bool nesterov_;

public:
    SGD(NNGraph* graph,
        module::Module* module,
        Scalar lr = 0.01,
        Scalar momentum = 0.0,
        Scalar weight_decay = 0.0,
        Scalar dampening = 0.0,
        bool nesterov = false);

    void step() override;

    void save_config(const std::string& path) const override;
    void load_config(const std::string& path) override;

    std::string repr() const override;

    Scalar lr() const { return lr_; }
    Scalar momentum() const { return momentum_; }
    Scalar weight_decay() const { return weight_decay_; }
    Scalar dampening() const { return dampening_; }
    bool nesterov() const { return nesterov_; }
};

} // namespace nntile::graph::optim
