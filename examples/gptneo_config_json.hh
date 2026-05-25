/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/gptneo_config_json.hh
 * Load/save GptneoConfig JSON for C++ examples (HF + NNTile key names).
 *
 * @version 1.1.0
 * */

#pragma once

#include "json_config_helpers.hh"

#include <nlohmann/json.hpp>
#include <nntile/graph/model/gptneo/gptneo_config.hh>

#include <fstream>
#include <stdexcept>
#include <string>

namespace nntile::examples
{

//! Load ``GptneoConfig`` from JSON (``gptneo_generate.py``, training save, HF).
inline model::gptneo::GptneoConfig load_gptneo_config_json(std::string const &path)
{
    std::ifstream f(path);
    if (!f.good())
    {
        throw std::runtime_error("Cannot open config: " + path);
    }
    nlohmann::json j = nlohmann::json::parse(f);

    model::gptneo::GptneoConfig cfg;
    cfg.vocab_size = config_get_int(j, "vocab_size", 50257);
    cfg.hidden_size = config_get_int(
        j, "hidden_size", config_get_int(j, "n_embd", 768));
    cfg.num_hidden_layers = config_get_int(
        j, "num_hidden_layers", config_get_int(j, "n_layer", 12));
    cfg.num_attention_heads = config_get_int(
        j, "num_attention_heads", config_get_int(j, "n_head", 12));
    cfg.max_position_embeddings = config_get_int(
        j,
        "max_position_embeddings",
        config_get_int(j, "n_positions", 1024));
    cfg.layer_norm_eps = config_get_float(
        j, "layer_norm_eps", config_get_float(j, "layer_norm_epsilon", 1e-5f));
    cfg.intermediate_size = config_get_int(
        j, "intermediate_size", config_get_int(j, "n_inner", 0));
    if (cfg.intermediate_size <= 0)
    {
        cfg.intermediate_size = 4 * cfg.hidden_size;
    }
    cfg.head_dim = config_get_int(j, "head_dim", 0);
    cfg.window_size = config_get_int(j, "window_size", 256);
    cfg.eos_token_id = config_get_int(j, "eos_token_id", 50256);
    cfg.bos_token_id = config_get_int(j, "bos_token_id", 50256);
    if (j.contains("name") && j["name"].is_string())
    {
        cfg.name = j["name"].get<std::string>();
    }
    cfg.compute_head_dim();
    cfg.validate();
    return cfg;
}

//! Write ``GptneoConfig`` for training checkpoints (HF ``layer_norm_epsilon`` key).
inline void save_gptneo_config_json(
    model::gptneo::GptneoConfig const &cfg,
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
    j["window_size"] = cfg.window_size;
    j["layer_norm_epsilon"] = cfg.layer_norm_eps;
    j["eos_token_id"] = cfg.eos_token_id;
    j["bos_token_id"] = cfg.bos_token_id;
    j["name"] = cfg.name;
    std::ofstream f(path);
    if (!f.good())
    {
        throw std::runtime_error("Cannot write config: " + path);
    }
    f << j.dump(2) << "\n";
}

} // namespace nntile::examples
