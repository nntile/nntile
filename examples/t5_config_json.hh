/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/t5_config_json.hh
 * Load/save T5Config JSON for C++ examples (HF + NNTile key names).
 *
 * @version 1.1.0
 * */

#pragma once

#include "json_config_helpers.hh"

#include <nlohmann/json.hpp>
#include <nntile/graph/model/t5/t5_config.hh>

#include <fstream>
#include <stdexcept>
#include <string>

namespace nntile::examples
{

//! Load ``T5Config`` from JSON (``t5_generate.py``, training save, HF).
inline graph::model::t5::T5Config load_t5_config_json(std::string const &path)
{
    std::ifstream f(path);
    if (!f.good())
    {
        throw std::runtime_error("Cannot open config: " + path);
    }
    nlohmann::json j = nlohmann::json::parse(f);

    graph::model::t5::T5Config cfg;
    cfg.vocab_size = config_get_int(j, "vocab_size", 32100);
    cfg.d_model = config_get_int(j, "d_model", 512);
    cfg.d_kv = config_get_int(j, "d_kv", 64);
    cfg.d_ff = config_get_int(j, "d_ff", 1024);
    cfg.num_layers = config_get_int(j, "num_layers", 6);
    cfg.num_decoder_layers = config_get_int(
        j, "num_decoder_layers", config_get_int(j, "num_layers", 6));
    cfg.num_heads = config_get_int(j, "num_heads", 8);
    cfg.layer_norm_epsilon = config_get_float(
        j, "layer_norm_epsilon", config_get_float(j, "layer_norm_eps", 1e-5f));
    cfg.pad_token_id = config_get_int(j, "pad_token_id", 0);
    cfg.eos_token_id = config_get_int(j, "eos_token_id", 1);
    cfg.decoder_start_token_id = config_get_int(
        j, "decoder_start_token_id", cfg.pad_token_id);
    if (j.contains("name") && j["name"].is_string())
    {
        cfg.name = j["name"].get<std::string>();
    }
    cfg.validate();
    return cfg;
}

//! Write ``T5Config`` for training checkpoints.
inline void save_t5_config_json(
    graph::model::t5::T5Config const &cfg,
    std::string const &path)
{
    nlohmann::json j;
    j["vocab_size"] = cfg.vocab_size;
    j["d_model"] = cfg.d_model;
    j["d_kv"] = cfg.d_kv;
    j["d_ff"] = cfg.d_ff;
    j["num_layers"] = cfg.num_layers;
    j["num_decoder_layers"] = cfg.num_decoder_layers;
    j["num_heads"] = cfg.num_heads;
    j["layer_norm_epsilon"] = cfg.layer_norm_epsilon;
    j["pad_token_id"] = cfg.pad_token_id;
    j["eos_token_id"] = cfg.eos_token_id;
    j["decoder_start_token_id"] = cfg.decoder_start_token_id;
    j["name"] = cfg.name;
    std::ofstream f(path);
    if (!f.good())
    {
        throw std::runtime_error("Cannot write config: " + path);
    }
    f << j.dump(2) << "\n";
}

} // namespace nntile::examples
