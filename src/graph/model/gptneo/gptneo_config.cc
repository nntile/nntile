#include <nntile/graph/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/model/gptneo/gptneo_config.cc
 * GPT-Neo configuration helpers.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gptneo/gptneo_config.hh"

#include <stdexcept>

namespace nntile::graph::model::gptneo
{

void GptneoConfig::build_attention_layers()
{
    if(!attention_layers.empty())
    {
        return;
    }
    attention_layers.resize(static_cast<std::size_t>(num_hidden_layers));
    for(Index i = 0; i < num_hidden_layers; ++i)
    {
        attention_layers[static_cast<std::size_t>(i)] =
            (i % 2 == 1) ? "local" : "global";
    }
}

bool GptneoConfig::is_local_attention_layer(Index layer_id) const
{
    if(layer_id < 0 || layer_id >= num_hidden_layers)
    {
        throw std::out_of_range(
            "GptneoConfig::is_local_attention_layer: layer_id out of range");
    }
    if(attention_layers.empty())
    {
        return layer_id % 2 == 1;
    }
    return attention_layers[static_cast<std::size_t>(layer_id)] == "local";
}

} // namespace nntile::graph::model::gptneo
