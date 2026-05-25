/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/graph/model/gptneox/gptneox_config_json.hh
 * Parse ``attention_layers`` / HF ``attention_types`` into ``GptneoxConfig``.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nlohmann/json.hpp>
#include <nntile/graph/model/gptneox/gptneox_config.hh>

#include <stdexcept>
#include <string>
#include <vector>

namespace nntile::model::gptneox
{

//! Expand HF ``attention_types`` like ``GPTNeoXConfig.expand_attention_types``.
inline void parse_gptneox_attention_layers(
    nlohmann::json const &j,
    GptneoxConfig &cfg)
{
    if (j.contains("attention_layers") && j["attention_layers"].is_array())
    {
        cfg.attention_layers.clear();
        for (auto const &entry : j["attention_layers"])
        {
            if (!entry.is_string())
            {
                throw std::runtime_error(
                    "GptneoxConfig JSON: attention_layers entries must be "
                    "strings");
            }
            cfg.attention_layers.push_back(entry.get<std::string>());
        }
        return;
    }
    if (!j.contains("attention_types") || !j["attention_types"].is_array())
    {
        return;
    }
    cfg.attention_layers.clear();
    for (auto const &group : j["attention_types"])
    {
        if (!group.is_array() || group.size() != 2)
        {
            throw std::runtime_error(
                "GptneoxConfig JSON: attention_types must be [[types], count], "
                "...");
        }
        auto const &types_json = group[0];
        int count = group.at(1).get<int>();
        if (!types_json.is_array() || count <= 0)
        {
            throw std::runtime_error(
                "GptneoxConfig JSON: invalid attention_types group");
        }
        std::vector<std::string> types;
        types.reserve(types_json.size());
        for (auto const &entry : types_json)
        {
            if (!entry.is_string())
            {
                throw std::runtime_error(
                    "GptneoxConfig JSON: attention_types type must be string");
            }
            types.push_back(entry.get<std::string>());
        }
        if (types.empty())
        {
            throw std::runtime_error(
                "GptneoxConfig JSON: attention_types type list is empty");
        }
        for (int rep = 0; rep < count; ++rep)
        {
            for (auto const &layer_type : types)
            {
                cfg.attention_layers.push_back(layer_type);
            }
        }
    }
}

} // namespace nntile::model::gptneox
