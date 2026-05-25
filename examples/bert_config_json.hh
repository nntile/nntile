/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file examples/bert_config_json.hh
 * Load/save BertConfig JSON for C++ examples.
 *
 * @version 1.1.0
 * */

#pragma once

#include "json_config_helpers.hh"

#include <nlohmann/json.hpp>
#include <nntile/graph/model/bert/bert_config.hh>

#include <fstream>
#include <stdexcept>
#include <string>

namespace nntile::examples
{

inline model::bert::BertConfig load_bert_config_json(std::string const &path)
{
    std::ifstream f(path);
    if (!f.good())
    {
        throw std::runtime_error("Cannot open config: " + path);
    }
    nlohmann::json j = nlohmann::json::parse(f);

    model::bert::BertConfig cfg;
    cfg.vocab_size = config_get_int(j, "vocab_size", 30522);
    cfg.hidden_size = config_get_int(j, "hidden_size", 768);
    cfg.num_hidden_layers = config_get_int(j, "num_hidden_layers", 12);
    cfg.num_attention_heads = config_get_int(j, "num_attention_heads", 12);
    cfg.max_position_embeddings = config_get_int(
        j, "max_position_embeddings", 512);
    cfg.type_vocab_size = config_get_int(j, "type_vocab_size", 2);
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
    cfg.validate();
    return cfg;
}

inline void save_bert_config_json(
    model::bert::BertConfig const &cfg,
    std::string const &path)
{
    nlohmann::json j;
    j["vocab_size"] = cfg.vocab_size;
    j["hidden_size"] = cfg.hidden_size;
    j["intermediate_size"] = cfg.intermediate_size;
    j["num_hidden_layers"] = cfg.num_hidden_layers;
    j["num_attention_heads"] = cfg.num_attention_heads;
    j["max_position_embeddings"] = cfg.max_position_embeddings;
    j["type_vocab_size"] = cfg.type_vocab_size;
    j["layer_norm_eps"] = cfg.layer_norm_eps;
    j["name"] = cfg.name;
    std::ofstream f(path);
    if (!f.good())
    {
        throw std::runtime_error("Cannot write config: " + path);
    }
    f << j.dump(2) << "\n";
}

} // namespace nntile::examples
