/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file examples/gptneox_config_json.hh
 * Load/save GptneoxConfig JSON for C++ examples.
 *
 * @version 1.1.0
 * */

#pragma once

#include "json_config_helpers.hh"

#include <nlohmann/json.hpp>
#include <nntile/model/gptneox/gptneox_config.hh>

#include <fstream>
#include <stdexcept>
#include <string>

namespace nntile::examples
{

inline model::gptneox::GptneoxConfig load_gptneox_config_json(
    std::string const &path)
{
    std::ifstream f(path);
    if(!f.good())
    {
        throw std::runtime_error("Cannot open config: " + path);
    }
    nlohmann::json j = nlohmann::json::parse(f);

    model::gptneox::GptneoxConfig cfg;
    cfg.vocab_size = config_get_int(j, "vocab_size", 50280);
    cfg.hidden_size = config_get_int(
        j, "hidden_size", config_get_int(j, "n_embd", 1024));
    cfg.num_hidden_layers = config_get_int(
        j, "num_hidden_layers", config_get_int(j, "n_layer", 24));
    cfg.num_attention_heads = config_get_int(
        j, "num_attention_heads", config_get_int(j, "n_head", 16));
    cfg.max_position_embeddings = config_get_int(
        j, "max_position_embeddings", config_get_int(j, "n_positions", 2048));
    cfg.layer_norm_eps = config_get_float(j, "layer_norm_eps", 1e-5f);
    cfg.intermediate_size = config_get_int(
        j, "intermediate_size", config_get_int(j, "n_inner", 0));
    if(cfg.intermediate_size <= 0)
    {
        cfg.intermediate_size = 4 * cfg.hidden_size;
    }
    cfg.rotary_pct = config_get_float(j, "rotary_pct", 0.25f);
    cfg.rotary_emb_base = config_get_float(j, "rotary_emb_base", 10000.0f);
    cfg.use_parallel_residual =
        config_get_bool(j, "use_parallel_residual", true);
    cfg.attention_bias = config_get_bool(j, "attention_bias", false);
    cfg.eos_token_id = config_get_int(j, "eos_token_id", 50256);
    cfg.bos_token_id = config_get_int(j, "bos_token_id", 50256);
    if(j.contains("name") && j["name"].is_string())
    {
        cfg.name = j["name"].get<std::string>();
    }
    cfg.compute_head_dim();
    cfg.validate();
    return cfg;
}

inline void save_gptneox_config_json(
    model::gptneox::GptneoxConfig const &cfg,
    std::string const &path)
{
    nlohmann::json j;
    j["vocab_size"] = cfg.vocab_size;
    j["hidden_size"] = cfg.hidden_size;
    j["intermediate_size"] = cfg.intermediate_size;
    j["num_hidden_layers"] = cfg.num_hidden_layers;
    j["num_attention_heads"] = cfg.num_attention_heads;
    j["max_position_embeddings"] = cfg.max_position_embeddings;
    j["head_dim"] = cfg.head_dim;
    j["layer_norm_eps"] = cfg.layer_norm_eps;
    j["rotary_pct"] = cfg.rotary_pct;
    j["rotary_emb_base"] = cfg.rotary_emb_base;
    j["use_parallel_residual"] = cfg.use_parallel_residual;
    j["attention_bias"] = cfg.attention_bias;
    j["eos_token_id"] = cfg.eos_token_id;
    j["bos_token_id"] = cfg.bos_token_id;
    j["name"] = cfg.name;
    std::ofstream f(path);
    if(!f.good())
    {
        throw std::runtime_error("Cannot write config: " + path);
    }
    f << j.dump(2) << "\n";
}

} // namespace nntile::examples
