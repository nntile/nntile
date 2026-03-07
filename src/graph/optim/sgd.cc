/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/optim/sgd.cc
 * SGD optimizer implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/optim/sgd.hh"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "nntile/graph/nn/sgd_step.hh"

namespace nntile::graph::optim
{

SGD::SGD(NNGraph* graph,
         module::Module* module,
         Scalar lr,
         Scalar momentum,
         Scalar weight_decay,
         Scalar dampening,
         bool nesterov)
    : Optimizer(graph, module)
    , lr_(lr)
    , momentum_(momentum)
    , weight_decay_(weight_decay)
    , dampening_(dampening)
    , nesterov_(nesterov)
{
}

void SGD::step()
{
    for(auto& ps : param_states_)
    {
        std::string vel_name = ps.name + "_velocity";
        auto* velocity = graph_->tensor(
            ps.param->shape(), vel_name,
            ps.param->dtype(), false);
        velocity->mark_input(true);
        velocity->mark_output(true);

        ps.param->mark_input(true);
        ps.param->mark_output(true);

        sgd_step(ps.param, ps.grad, velocity,
                 1, momentum_, lr_, weight_decay_, dampening_, nesterov_);

        ps.buffers.emplace_back(vel_name, velocity);
    }
}

void SGD::save_config(const std::string& path) const
{
    nlohmann::json j;
    j["optimizer"] = "SGD";
    j["num_iter"] = num_iter_;
    j["lr"] = lr_;
    j["momentum"] = momentum_;
    j["weight_decay"] = weight_decay_;
    j["dampening"] = dampening_;
    j["nesterov"] = nesterov_;

    nlohmann::json params = nlohmann::json::array();
    for(const auto& ps : param_states_)
    {
        params.push_back(ps.name);
    }
    j["param_names"] = params;

    std::ofstream f(path);
    if(!f.is_open())
    {
        throw std::runtime_error(
            "SGD::save_config: cannot open '" + path + "'");
    }
    f << j.dump(2);
}

void SGD::load_config(const std::string& path)
{
    std::ifstream f(path);
    if(!f.is_open())
    {
        throw std::runtime_error(
            "SGD::load_config: cannot open '" + path + "'");
    }
    nlohmann::json j = nlohmann::json::parse(f);
    if(j.at("optimizer").get<std::string>() != "SGD")
    {
        throw std::runtime_error(
            "SGD::load_config: optimizer type mismatch");
    }
    num_iter_ = j.at("num_iter").get<Index>();
    lr_ = j.at("lr").get<Scalar>();
    momentum_ = j.at("momentum").get<Scalar>();
    weight_decay_ = j.at("weight_decay").get<Scalar>();
    dampening_ = j.at("dampening").get<Scalar>();
    nesterov_ = j.at("nesterov").get<bool>();
}

std::string SGD::repr() const
{
    return "SGD(lr=" + std::to_string(lr_) +
           ", momentum=" + std::to_string(momentum_) +
           ", weight_decay=" + std::to_string(weight_decay_) +
           ", dampening=" + std::to_string(dampening_) +
           ", nesterov=" + (nesterov_ ? "true" : "false") +
           ", num_params=" + std::to_string(param_states_.size()) + ")";
}

} // namespace nntile::graph::optim
