/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file examples/tiling_config_json.hh
 * Load/save tiling.json (``default`` + ``layers``) for graph training examples.
 *
 * @version 1.1.0
 * */

#pragma once

#include "json_config_helpers.hh"

#include <nlohmann/json.hpp>
#include <nntile/base_types.hh>
#include <nntile/tensor/axis_descriptor.hh>
#include <nntile/tensor/graph.hh>

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace nntile::examples
{

//! Parsed tiling.json (canonical: ``default`` + ``layers``).
struct FlatTilingSpec
{
    std::map<std::string, std::vector<Index>> defaults;
    std::map<Index, std::map<std::string, std::vector<Index>>> per_layer;
};

inline bool is_tiling_axis_key(std::string const &key)
{
    return key == "vocab_size" || key == "hidden_size" ||
           key == "intermediate_size" || key == "num_attention_heads" ||
           key == "max_position_embeddings" || key == "seq_len" ||
           key == "batch_size";
}

inline std::string normalize_tiling_axis_key(std::string key)
{
    if (key == "n_embd")
    {
        return "hidden_size";
    }
    if (key == "n_inner")
    {
        return "intermediate_size";
    }
    if (key == "n_head")
    {
        return "num_attention_heads";
    }
    if (key == "n_positions")
    {
        return "max_position_embeddings";
    }
    return key;
}

inline std::vector<Index> parse_tile_sizes_json(
    nlohmann::json const &v,
    Index extent,
    char const *context)
{
    if (v.is_number_integer())
    {
        Index const t = v.get<Index>();
        if (t <= 0)
        {
            throw std::runtime_error(
                std::string(context) + ": tile size must be positive");
        }
        if (extent > 0 && t > extent)
        {
            throw std::runtime_error(
                std::string(context) + ": uniform tile larger than extent");
        }
        std::vector<Index> sizes;
        if (extent <= 0)
        {
            sizes.push_back(t);
            return sizes;
        }
        for (Index rem = extent; rem > 0; rem -= t)
        {
            sizes.push_back(std::min(t, rem));
        }
        return sizes;
    }
    if (v.is_array())
    {
        std::vector<Index> sizes;
        sizes.reserve(v.size());
        Index total = 0;
        for (auto const &el : v)
        {
            if (!el.is_number_integer())
            {
                throw std::runtime_error(
                    std::string(context) + ": tile array must be integers");
            }
            Index const t = el.get<Index>();
            if (t <= 0)
            {
                throw std::runtime_error(
                    std::string(context) + ": tile size must be positive");
            }
            sizes.push_back(t);
            total += t;
        }
        if (extent > 0 && total != extent)
        {
            throw std::runtime_error(
                std::string(context) + ": tile sizes sum (" +
                std::to_string(total) + ") != extent (" +
                std::to_string(extent) + ")");
        }
        return sizes;
    }
    throw std::runtime_error(
        std::string(context) + ": tile value must be int or int array");
}

inline Index resolve_layer_key(
    std::string const &key,
    Index num_hidden_layers)
{
    std::string k = key;
    std::string const prefix = "transformer.";
    if (k.rfind(prefix, 0) == 0)
    {
        k = k.substr(prefix.size());
    }
    if (k.rfind("h_", 0) == 0)
    {
        k = k.substr(2);
    }
    int idx = 0;
    try
    {
        idx = std::stoi(k);
    }
    catch (...)
    {
        throw std::runtime_error(
            "tiling layers: invalid layer key '" + key + "'");
    }
    if (idx < 0 || static_cast<Index>(idx) >= num_hidden_layers)
    {
        throw std::runtime_error(
            "tiling layers: layer index " + std::to_string(idx) +
            " out of range [0, " + std::to_string(num_hidden_layers) + ")");
    }
    return static_cast<Index>(idx);
}

inline FlatTilingSpec load_tiling_from_json(
    nlohmann::json const &j,
    Index num_hidden_layers)
{
    FlatTilingSpec spec;
    if (!j.contains("default") || !j["default"].is_object())
    {
        throw std::runtime_error(
            "tiling.json: missing or invalid top-level \"default\" object");
    }
    if (!j.contains("layers") || !j["layers"].is_object())
    {
        throw std::runtime_error(
            "tiling.json: missing or invalid top-level \"layers\" object");
    }
    for (auto it = j["default"].begin(); it != j["default"].end(); ++it)
    {
        std::string const axis = normalize_tiling_axis_key(it.key());
        if (!is_tiling_axis_key(axis))
        {
            throw std::runtime_error(
                "tiling.json default: unknown axis key '" + it.key() + "'");
        }
        spec.defaults.emplace(
            axis,
            parse_tile_sizes_json(it.value(), 0, "tiling.json default"));
    }
    for (auto it = j["layers"].begin(); it != j["layers"].end(); ++it)
    {
        Index const layer = resolve_layer_key(it.key(), num_hidden_layers);
        if (!it.value().is_object())
        {
            throw std::runtime_error(
                "tiling.json layers." + it.key() +
                ": expected object of axis tiling");
        }
        auto &layer_map = spec.per_layer[layer];
        for (auto ax = it.value().begin(); ax != it.value().end(); ++ax)
        {
            std::string const axis = normalize_tiling_axis_key(ax.key());
            if (axis != "intermediate_size" && axis != "num_attention_heads")
            {
                throw std::runtime_error(
                    "tiling.json layers." + it.key() +
                    ": only intermediate_size and num_attention_heads "
                    "allowed in layers");
            }
            layer_map.emplace(
                axis,
                parse_tile_sizes_json(
                    ax.value(), 0, "tiling.json layers"));
        }
    }
    return spec;
}

inline FlatTilingSpec load_tiling_json(
    std::string const &path,
    Index num_hidden_layers)
{
    std::ifstream f(path);
    if (!f.good())
    {
        throw std::runtime_error("Cannot open tiling file: " + path);
    }
    nlohmann::json j = nlohmann::json::parse(f);
    return load_tiling_from_json(j, num_hidden_layers);
}

inline nlohmann::json tile_sizes_to_json(std::vector<Index> const &sizes)
{
    if (sizes.size() == 1)
    {
        return sizes[0];
    }
    nlohmann::json arr = nlohmann::json::array();
    for (Index s : sizes)
    {
        arr.push_back(s);
    }
    return arr;
}

inline nlohmann::json flat_tiling_spec_to_json(FlatTilingSpec const &spec)
{
    nlohmann::json j;
    nlohmann::json def = nlohmann::json::object();
    for (auto const &[axis, sizes] : spec.defaults)
    {
        def[axis] = tile_sizes_to_json(sizes);
    }
    j["default"] = def;
    nlohmann::json layers = nlohmann::json::object();
    for (auto const &[layer, roles] : spec.per_layer)
    {
        std::string const key = "h_" + std::to_string(layer);
        nlohmann::json lob = nlohmann::json::object();
        for (auto const &[axis, sizes] : roles)
        {
            lob[axis] = tile_sizes_to_json(sizes);
        }
        layers[key] = lob;
    }
    j["layers"] = layers;
    return j;
}

inline void save_tiling_json(
    FlatTilingSpec const &spec,
    std::string const &path)
{
    std::ofstream f(path);
    if (!f.good())
    {
        throw std::runtime_error("Cannot write tiling file: " + path);
    }
    f << flat_tiling_spec_to_json(spec).dump(2) << "\n";
}

inline bool parse_layer_axis_group_name(
    std::string const &name,
    Index &layer_out,
    std::string &axis_out)
{
    std::string const prefix = "layer.";
    if (name.rfind(prefix, 0) != 0)
    {
        return false;
    }
    std::string rest = name.substr(prefix.size());
    auto const dot = rest.find('.');
    if (dot == std::string::npos)
    {
        return false;
    }
    try
    {
        layer_out = static_cast<Index>(std::stoi(rest.substr(0, dot)));
    }
    catch (...)
    {
        return false;
    }
    axis_out = rest.substr(dot + 1);
    return true;
}

inline void apply_tiling_to_axis(
    AxisDescriptor *ad,
    std::vector<Index> const &sizes)
{
    if (sizes.size() == 1)
    {
        ad->set_tiling(sizes[0]);
    }
    else
    {
        std::vector<Index> sized = sizes;
        Index total = 0;
        for (Index s : sized)
        {
            total += s;
        }
        if (total != ad->extent)
        {
            throw std::runtime_error(
                "tiling: sum of tile sizes for axis '" + ad->name +
                "' (" + std::to_string(total) + ") != extent (" +
                std::to_string(ad->extent) + ")");
        }
        ad->set_tiling(sized);
    }
}

inline std::vector<Index> tile_sizes_for_axis_extent(
    std::vector<Index> const &pattern,
    Index extent)
{
    if (pattern.size() == 1)
    {
        Index const t = pattern[0];
        std::vector<Index> sizes;
        for (Index rem = extent; rem > 0; rem -= t)
        {
            sizes.push_back(std::min(t, rem));
        }
        return sizes;
    }
    Index total = 0;
    for (Index s : pattern)
    {
        total += s;
    }
    if (total != extent)
    {
        throw std::runtime_error(
            "tiling: tile sizes sum (" + std::to_string(total) +
            ") != extent (" + std::to_string(extent) + ")");
    }
    return pattern;
}

inline void apply_flat_tiling_spec(
    TensorGraph &tg,
    FlatTilingSpec const &spec,
    Index num_hidden_layers)
{
    (void) num_hidden_layers;
    for (AxisDescriptor *ad : tg.axis_groups())
    {
        if (ad->name.empty())
        {
            continue;
        }
        Index layer = 0;
        std::string axis_key;
        std::vector<Index> const *sizes = nullptr;
        if (parse_layer_axis_group_name(ad->name, layer, axis_key))
        {
            auto lit = spec.per_layer.find(layer);
            if (lit != spec.per_layer.end())
            {
                auto ait = lit->second.find(axis_key);
                if (ait != lit->second.end())
                {
                    sizes = &ait->second;
                }
            }
            if (sizes == nullptr)
            {
                auto dit = spec.defaults.find(axis_key);
                if (dit != spec.defaults.end())
                {
                    sizes = &dit->second;
                }
            }
        }
        else
        {
            auto dit = spec.defaults.find(ad->name);
            if (dit != spec.defaults.end())
            {
                sizes = &dit->second;
            }
        }
        if (sizes == nullptr)
        {
            continue;
        }
        std::vector<Index> const resolved =
            tile_sizes_for_axis_extent(*sizes, ad->extent);
        apply_tiling_to_axis(ad, resolved);
    }
}

} // namespace nntile::examples
