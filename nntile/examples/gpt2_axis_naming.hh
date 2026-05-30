/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file examples/gpt2_axis_naming.hh
 * Name TensorGraph axis groups for GPT-2 training + tiling.json keys.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/model/gpt2/gpt2_config.hh>
#include <nntile/tensor/axis_descriptor.hh>
#include <nntile/tensor/graph.hh>

#include <cctype>
#include <optional>
#include <string>

namespace nntile::examples
{

inline std::optional<Index> parse_h_layer_index_from_tensor_name(
    std::string const &tensor_name)
{
    std::string const needle = "_h_";
    auto pos = tensor_name.find(needle);
    if (pos == std::string::npos)
    {
        return std::nullopt;
    }
    pos += needle.size();
    if (pos >= tensor_name.size() || !std::isdigit(tensor_name[pos]))
    {
        return std::nullopt;
    }
    Index idx = 0;
    while (pos < tensor_name.size() && std::isdigit(tensor_name[pos]))
    {
        idx = idx * 10 + static_cast<Index>(tensor_name[pos] - '0');
        ++pos;
    }
    if (pos < tensor_name.size() && tensor_name[pos] != '_')
    {
        return std::nullopt;
    }
    return idx;
}

inline bool axis_group_member_name_contains(
    AxisDescriptor const *ad,
    char const *needle)
{
    for (auto const &[node_ptr, axis_idx] : ad->members)
    {
        (void) axis_idx;
        auto *node = static_cast<TensorGraph::TensorNode const *>(node_ptr);
        if (node->name().find(needle) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

//! Axis index on ``input_ids`` / ``labels`` / ``position_ids`` members, if any.
inline std::optional<size_t> axis_group_training_io_axis_index(
    AxisDescriptor const *ad)
{
    std::optional<size_t> idx;
    for (auto const &[node_ptr, axis_idx] : ad->members)
    {
        auto *node = static_cast<TensorGraph::TensorNode const *>(node_ptr);
        std::string const &tname = node->name();
        if (tname.find("input_ids") == std::string::npos &&
            tname.find("labels") == std::string::npos &&
            tname.find("position_ids") == std::string::npos)
        {
            continue;
        }
        if (idx.has_value() && *idx != axis_idx)
        {
            return std::nullopt;
        }
        idx = axis_idx;
    }
    return idx;
}

inline bool axis_group_looks_like_batch(AxisDescriptor const *ad)
{
    if (axis_group_member_name_contains(ad, "_attn_"))
    {
        return false;
    }
    if (axis_group_member_name_contains(ad, "input_ids") ||
        axis_group_member_name_contains(ad, "labels") ||
        axis_group_member_name_contains(ad, "position_ids"))
    {
        return true;
    }
    return !axis_group_member_name_contains(ad, "_h_");
}

inline bool axis_group_looks_like_seq(AxisDescriptor const *ad)
{
    if (axis_group_member_name_contains(ad, "_h_"))
    {
        return false;
    }
    if (axis_group_member_name_contains(ad, "attn_mask"))
    {
        return true;
    }
    if (axis_group_member_name_contains(ad, "input_ids") ||
        axis_group_member_name_contains(ad, "labels") ||
        axis_group_member_name_contains(ad, "position_ids"))
    {
        return true;
    }
    return false;
}

inline void name_gpt2_layer_local_axis_groups(
    TensorGraph &tg,
    model::gpt2::Gpt2Config const &cfg)
{
    for (AxisDescriptor *ad : tg.axis_groups())
    {
        if (!ad->name.empty())
        {
            continue;
        }
        if (ad->extent != cfg.intermediate_size &&
            ad->extent != cfg.num_attention_heads)
        {
            continue;
        }
        std::optional<Index> layer_idx;
        bool is_mlp = false;
        bool is_attn = false;
        for (auto const &[node_ptr, axis_idx] : ad->members)
        {
            (void) axis_idx;
            auto *node = static_cast<TensorGraph::TensorNode *>(node_ptr);
            std::string const &tname = node->name();
            if (tname.find("_mlp_") != std::string::npos)
            {
                is_mlp = true;
            }
            if (tname.find("_attn_") != std::string::npos)
            {
                is_attn = true;
            }
            auto parsed = parse_h_layer_index_from_tensor_name(tname);
            if (parsed.has_value())
            {
                if (layer_idx.has_value() && *layer_idx != *parsed)
                {
                    layer_idx = std::nullopt;
                    break;
                }
                layer_idx = parsed;
            }
        }
        if (!layer_idx.has_value())
        {
            continue;
        }
        if (is_mlp && ad->extent == cfg.intermediate_size)
        {
            ad->name = "layer." + std::to_string(*layer_idx) +
                       ".intermediate_size";
            continue;
        }
        if (is_attn && ad->extent == cfg.num_attention_heads)
        {
            ad->name = "layer." + std::to_string(*layer_idx) +
                       ".num_attention_heads";
        }
    }
}

inline void name_gpt2_global_axis_groups(
    TensorGraph &tg,
    model::gpt2::Gpt2Config const &cfg,
    Index seq_len,
    Index batch_size)
{
    bool const seq_batch_same =
        seq_len > 0 && batch_size > 0 && seq_len == batch_size;
    for (AxisDescriptor *ad : tg.axis_groups())
    {
        if (!ad->name.empty())
        {
            continue;
        }
        if (seq_batch_same && ad->extent == seq_len)
        {
            if (axis_group_member_name_contains(ad, "attn_mask"))
            {
                ad->name = "seq_len";
                continue;
            }
            auto io_axis = axis_group_training_io_axis_index(ad);
            if (io_axis.has_value())
            {
                if (*io_axis == 0)
                {
                    ad->name = "seq_len";
                    continue;
                }
                if (*io_axis == 1)
                {
                    ad->name = "batch_size";
                    continue;
                }
            }
        }
        if (ad->extent == batch_size && batch_size > 0 &&
            axis_group_looks_like_batch(ad))
        {
            ad->name = "batch_size";
            continue;
        }
        if (ad->extent == cfg.vocab_size)
        {
            ad->name = "vocab_size";
            continue;
        }
        if (ad->extent == cfg.hidden_size &&
            !axis_group_member_name_contains(ad, "_attn_"))
        {
            ad->name = "hidden_size";
            continue;
        }
        if (ad->extent == seq_len && seq_len > 0 &&
            axis_group_looks_like_seq(ad))
        {
            ad->name = "seq_len";
            continue;
        }
        if (ad->extent == cfg.max_position_embeddings)
        {
            ad->name = "max_position_embeddings";
        }
    }
}

inline void name_gpt2_training_axis_groups(
    TensorGraph &tg,
    model::gpt2::Gpt2Config const &cfg,
    Index seq_len,
    Index batch_size)
{
    name_gpt2_layer_local_axis_groups(tg, cfg);
    name_gpt2_global_axis_groups(tg, cfg, seq_len, batch_size);
}

} // namespace nntile::examples
