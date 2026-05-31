/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file examples/tiling_config_json.hh
 * Backward-compatible include for example code (see tensor/tiling_spec_json.hh).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tiling_spec_json.hh>

namespace nntile::examples
{

using FlatTilingSpec = nntile::FlatTilingSpec;
using nntile::apply_flat_tiling_spec;
using nntile::apply_tiling_to_axis;
using nntile::flat_tiling_spec_to_json;
using nntile::is_tiling_axis_key;
using nntile::load_tiling_from_json;
using nntile::load_tiling_json;
using nntile::normalize_tiling_axis_key;
using nntile::parse_layer_axis_group_name;
using nntile::parse_tile_sizes_json;
using nntile::save_tiling_json;
using nntile::tile_sizes_for_axis_extent;
using nntile::tile_sizes_to_json;

inline Index resolve_layer_key(std::string const &key, Index num_hidden_layers)
{
    return resolve_tiling_layer_key(key, num_hidden_layers);
}

} // namespace nntile::examples
