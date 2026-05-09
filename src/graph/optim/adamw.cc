/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/optim/adamw.cc
 * AdamW optimizer implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/optim/adamw.hh"

#include "nntile/graph/nn/ops/adamw_step.hh"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace nntile::graph::optim
{

AdamW::AdamW(NNGraph *graph,
    module::Module *module,
    Scalar lr,
    Scalar beta_1,
    Scalar beta_2,
    Scalar eps,
    Scalar weight_decay) :
    Optimizer(graph, module),
    lr_(lr),
    beta_1_(beta_1),
    beta_2_(beta_2),
    eps_(eps),
    weight_decay_(weight_decay)
{
}

void AdamW::step_lr(std::optional<Scalar> lr_override)
{
    Scalar const lr_rec = lr_override.value_or(lr_);
    for (auto &ps : param_states_)
    {
        ps.grad = ps.param->grad();
        if (ps.grad == nullptr)
        {
            throw std::runtime_error(
                "AdamW::step: parameter '" + ps.name +
                "' has no gradient tensor; call backward() before step()");
        }

        NNGraph::TensorNode *first_moment = nullptr;
        NNGraph::TensorNode *second_moment = nullptr;
        std::string m1_name = ps.name + "_first_moment";
        std::string m2_name = ps.name + "_second_moment";

        if (ps.buffers.empty())
        {
            first_moment =
                graph_->tensor(ps.param->shape(), ps.param->dtype(), false)
                    ->set_name(m1_name);
            first_moment->mark_input(true);
            first_moment->mark_output(true);

            second_moment =
                graph_->tensor(ps.param->shape(), ps.param->dtype(), false)
                    ->set_name(m2_name);
            second_moment->mark_input(true);
            second_moment->mark_output(true);

            ps.buffers.emplace_back(m1_name, first_moment);
            ps.buffers.emplace_back(m2_name, second_moment);
        }
        else
        {
            first_moment = ps.buffers[0].second;
            second_moment = ps.buffers[1].second;
        }

        ps.param->mark_input(true);
        ps.param->mark_output(true);

        adamw_step(ps.param,
            ps.grad,
            first_moment,
            second_moment,
            num_iter_ + 1,
            beta_1_,
            beta_2_,
            eps_,
            lr_rec,
            weight_decay_);
    }
    ++num_iter_;
}

void AdamW::save_config(const std::string &path) const
{
    nlohmann::json j;
    j["optimizer"] = "AdamW";
    j["num_iter"] = num_iter_;
    j["lr"] = lr_;
    j["beta_1"] = beta_1_;
    j["beta_2"] = beta_2_;
    j["eps"] = eps_;
    j["weight_decay"] = weight_decay_;

    nlohmann::json params = nlohmann::json::array();
    for (const auto &ps : param_states_)
    {
        params.push_back(ps.name);
    }
    j["param_names"] = params;

    std::ofstream f(path);
    if (!f.is_open())
    {
        throw std::runtime_error(
            "AdamW::save_config: cannot open '" + path + "'");
    }
    f << j.dump(2);
}

void AdamW::load_config(const std::string &path)
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        throw std::runtime_error(
            "AdamW::load_config: cannot open '" + path + "'");
    }
    nlohmann::json j = nlohmann::json::parse(f);
    if (j.at("optimizer").get<std::string>() != "AdamW")
    {
        throw std::runtime_error(
            "AdamW::load_config: optimizer type mismatch");
    }
    num_iter_ = j.at("num_iter").get<Index>();
    lr_ = j.at("lr").get<Scalar>();
    beta_1_ = j.at("beta_1").get<Scalar>();
    beta_2_ = j.at("beta_2").get<Scalar>();
    eps_ = j.at("eps").get<Scalar>();
    weight_decay_ = j.at("weight_decay").get<Scalar>();
}

std::string AdamW::repr() const
{
    return "AdamW(lr=" + std::to_string(lr_) +
           ", beta_1=" + std::to_string(beta_1_) +
           ", beta_2=" + std::to_string(beta_2_) +
           ", eps=" + std::to_string(eps_) +
           ", weight_decay=" + std::to_string(weight_decay_) +
           ", num_params=" + std::to_string(param_states_.size()) + ")";
}

} // namespace nntile::graph::optim
