/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file examples/roberta_example_helpers.hh
 * Shared tiny RoBERTa config and parameter-init helpers for C++ examples.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <nntile/common.hh>
#include <nntile/model/roberta/roberta_config.hh>
#include <nntile/model/roberta/roberta_mlm.hh>

namespace nntile::examples
{

inline graph::model::roberta::RobertaConfig make_tiny_roberta_config(
    graph::Index num_hidden_layers,
    graph::Index max_position_embeddings,
    float layer_norm_eps = 1e-5f)
{
    graph::model::roberta::RobertaConfig c;
    c.vocab_size = 64;
    c.hidden_size = 32;
    c.intermediate_size = 64;
    c.num_hidden_layers = num_hidden_layers;
    c.num_attention_heads = 4;
    c.max_position_embeddings = max_position_embeddings;
    c.type_vocab_size = 1;
    c.pad_token_id = 1;
    c.layer_norm_eps = layer_norm_eps;
    c.validate();
    return c;
}

enum class RobertaParamInitScale
{
    Uniform05,
    FanInSqrt,
};

inline void init_random_roberta_parameter_hints(
    graph::model::roberta::RobertaMlm &model,
    std::mt19937 &gen,
    RobertaParamInitScale scale = RobertaParamInitScale::FanInSqrt)
{
    for (nntile::NNGraph::TensorNode *tensor :
         model.parameters_recursive())
    {
        const auto &shape = tensor->shape();
        graph::Index nelems = 1;
        for (auto d : shape)
        {
            nelems *= d;
        }

        std::vector<float> data(static_cast<std::size_t>(nelems));
        if (scale == RobertaParamInitScale::Uniform05)
        {
            std::uniform_real_distribution<float> wdist(-0.05f, 0.05f);
            for (auto &v : data)
            {
                v = wdist(gen);
            }
        }
        else
        {
            float fan_in = static_cast<float>(shape[0]);
            if (fan_in < 1.f)
            {
                fan_in = 1.f;
            }
            float limit = std::sqrt(1.0f / fan_in);
            std::uniform_real_distribution<float> wdist(-limit, limit);
            for (auto &v : data)
            {
                v = wdist(gen);
            }
        }
        std::vector<std::uint8_t> bytes(data.size() * sizeof(float));
        std::memcpy(bytes.data(), data.data(), bytes.size());
        tensor->data()->set_bind_hint(std::move(bytes));
    }
    model.mark_parameters_input_recursive();
}

inline void fill_roberta_position_ids(
    std::vector<std::int64_t> &pos,
    graph::Index n_seq,
    graph::Index n_batch,
    std::int64_t pad_token_id)
{
    const std::int64_t offset = pad_token_id + 1;
    for (graph::Index b = 0; b < n_batch; ++b)
    {
        for (graph::Index s = 0; s < n_seq; ++s)
        {
            pos[s + n_seq * b] = offset + static_cast<std::int64_t>(s);
        }
    }
}

} // namespace nntile::examples
