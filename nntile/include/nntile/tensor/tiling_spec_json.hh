/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/tensor/tiling_spec_json.hh
 * Load/save tiling.json (``default`` + ``layers``) for graph training.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nlohmann/json.hpp>
#include <nntile/base_types.hh>
#include <nntile/tensor/axis_descriptor.hh>
#include <nntile/tensor/graph.hh>

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace nntile
{

struct FlatTilingSpec
{
    std::map<std::string, std::vector<Index>> defaults;
    std::map<Index, std::map<std::string, std::vector<Index>>> per_layer;
};

bool is_tiling_axis_key(std::string const &key);
std::string normalize_tiling_axis_key(std::string key);
std::vector<Index> parse_tile_sizes_json(
    nlohmann::json const &v,
    Index extent,
    char const *context);
Index resolve_tiling_layer_key(std::string const &key, Index num_hidden_layers);
FlatTilingSpec load_tiling_from_json(
    nlohmann::json const &j,
    Index num_hidden_layers);
FlatTilingSpec load_tiling_json(std::string const &path, Index num_hidden_layers);
nlohmann::json tile_sizes_to_json(std::vector<Index> const &sizes);
nlohmann::json flat_tiling_spec_to_json(FlatTilingSpec const &spec);
void save_tiling_json(FlatTilingSpec const &spec, std::string const &path);
bool parse_layer_axis_group_name(
    std::string const &name,
    Index &layer_out,
    std::string &axis_out);
void apply_tiling_to_axis(AxisDescriptor *ad, std::vector<Index> const &sizes);
std::vector<Index> tile_sizes_for_axis_extent(
    std::vector<Index> const &pattern,
    Index extent);
void apply_flat_tiling_spec(TensorGraph &tg, FlatTilingSpec const &spec);

} // namespace nntile
