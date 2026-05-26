/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file examples/roberta_config_json.hh
 * Load/save RobertaConfig JSON for C++ examples.
 *
 * @version 1.1.0
 * */

#pragma once

#include "json_config_helpers.hh"

#include <nlohmann/json.hpp>
#include <nntile/graph/model/roberta/roberta_config.hh>

#include <fstream>
#include <stdexcept>
#include <string>

namespace nntile::examples
{

inline model::roberta::RobertaConfig load_roberta_config_json(
    std::string const &path)
{
    std::ifstream f(path);
    if (!f.good())
    {
        throw std::runtime_error("Cannot open config: " + path);
    }
    nlohmann::json j = nlohmann::json::parse(f);

    model::roberta::RobertaConfig cfg;
    cfg.vocab_size = config_get_int(j, "vocab_size", 50265);
    cfg.hidden_size = config_get_int(j, "hidden_size", 768);
    cfg.num_hidden_layers = config_get_int(j, "num_hidden_layers", 12);
    cfg.num_attention_heads = config_get_int(j, "num_attention_heads", 12);
    cfg.max_position_embeddings = config_get_int(
        j, "max_position_embeddings", 514);
    cfg.type_vocab_size = config_get_int(j, "type_vocab_size", 1);
    cfg.pad_token_id = config_get_int(j, "pad_token_id", 1);
    cfg.layer_norm_eps = config_get_float(j, "layer_norm_eps", 1e-5f);
    cfg.intermediate_size = config_get_int(j, "intermediate_size", 0);
    if (cfg.intermediate_size <= 0)
    {
        cfg.intermediate_size = 4 * cfg.hidden_size;
    }
    if (j.contains("name") && j["name"].is_string())
    {
        cfg.name = j["name"].get<std::string>();
    }
    if (j.contains("hidden_act") && j["hidden_act"].is_string())
    {
        cfg.hidden_act = j["hidden_act"].get<std::string>();
    }
    cfg.validate();
    return cfg;
}

} // namespace nntile::examples
