/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file examples/bert_graph_training.cc
 * BERT masked-LM training on the graph API (tiny demo).
 *
 * Uses a small in-memory toy dataset and verifies loss decreases.
 *
 * @version 1.1.0
 * */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "bert_config_json.hh"
#include <nntile.hh>
#include <nntile/graph/model/bert/bert_mlm.hh>
#include <nntile/graph/tensor/ops/clear.hh>

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::bert;
using namespace nntile::graph::optim;

namespace
{

constexpr int EXIT_OK = 0;
constexpr int EXIT_ERROR = 1;

static BertConfig make_tiny_config()
{
    BertConfig c;
    c.vocab_size = 64;
    c.hidden_size = 32;
    c.intermediate_size = 64;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.max_position_embeddings = 32;
    c.type_vocab_size = 2;
    c.layer_norm_eps = 1e-5f;
    c.validate();
    return c;
}

static void fill_arange_position_ids(
    std::vector<std::int64_t> &pos, Index n_seq, Index n_batch)
{
    for (Index b = 0; b < n_batch; ++b)
    {
        for (Index s = 0; s < n_seq; ++s)
        {
            pos[s + n_seq * b] = static_cast<std::int64_t>(s);
        }
    }
}

static void init_random_parameter_hints(BertMlm &model, std::mt19937 &gen)
{
    for (NNGraph::TensorNode *tensor : model.parameters_recursive())
    {
        const auto &shape = tensor->shape();
        Index nelems = 1;
        for (auto d : shape)
        {
            nelems *= d;
        }
        float fan_in = static_cast<float>(shape[0]);
        if (fan_in < 1.f)
        {
            fan_in = 1.f;
        }
        float limit = std::sqrt(1.0f / fan_in);
        std::uniform_real_distribution<float> wdist(-limit, limit);

        std::vector<float> data(static_cast<std::size_t>(nelems));
        for (auto &v : data)
        {
            v = wdist(gen);
        }
        std::vector<std::uint8_t> bytes(data.size() * sizeof(float));
        std::memcpy(bytes.data(), data.data(), bytes.size());
        tensor->data()->set_bind_hint(std::move(bytes));
    }
    model.mark_parameters_input_recursive();
}

static void sync_param_hint_from_runtime(
    Runtime &runtime, NNGraph::TensorNode *t)
{
    std::vector<std::uint8_t> bytes;
    auto d = runtime.get_output<float>(t);
    bytes.resize(d.size() * sizeof(float));
    std::memcpy(bytes.data(), d.data(), bytes.size());
    t->data()->set_bind_hint(std::move(bytes));
}

struct ToyMlmBatch
{
    std::vector<std::int64_t> input_ids;
    std::vector<std::int64_t> labels;
};

static ToyMlmBatch make_mlm_batch(
    Index n_seq,
    Index n_batch,
    Index vocab_size,
    std::mt19937 &rng,
    std::int64_t mask_token_id)
{
    ToyMlmBatch b;
    const std::size_t n = static_cast<std::size_t>(n_seq * n_batch);
    b.input_ids.resize(n);
    b.labels.resize(n, -100);
    std::uniform_int_distribution<std::int64_t> tok_dist(1, vocab_size - 1);
    for (auto &t : b.input_ids)
    {
        t = tok_dist(rng);
    }
    std::uniform_int_distribution<Index> pos_dist(0, n_seq - 1);
    for (Index col = 0; col < n_batch; ++col)
    {
        Index s = pos_dist(rng);
        const std::size_t idx = static_cast<std::size_t>(s + n_seq * col);
        b.labels[idx] = b.input_ids[idx];
        b.input_ids[idx] = mask_token_id;
    }
    return b;
}

} // namespace

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    BertConfig config = make_tiny_config();
    const Index n_seq = 8;
    const Index n_batch = 2;
    const std::int64_t mask_token_id = 3;
    const std::size_t num_batches = 8;
    const float learning_rate = 0.01f;

    Context context(1, 0, 0, "/tmp/nntile_ooc", 16777216, 0, "localhost", 5001, 0);

    NNGraph graph("bert_graph_training");
    BertMlm model(&graph, "model", config);

    auto *input_ids = graph.tensor({n_seq, n_batch}, DataType::INT64, false)
                          ->set_name("input_ids");
    auto *token_type_ids =
        graph.tensor({n_seq, n_batch}, DataType::INT64, false)
            ->set_name("token_type_ids");
    auto *position_ids =
        graph.tensor({n_seq, n_batch}, DataType::INT64, false)
            ->set_name("position_ids");
    auto *labels = graph.tensor({n_seq, n_batch}, DataType::INT64, false)
                       ->set_name("labels");
    input_ids->mark_input(true);
    token_type_ids->mark_input(true);
    position_ids->mark_input(true);
    labels->mark_input(true);

    std::mt19937 gen(42);
    init_random_parameter_hints(model, gen);

    auto optimizer = std::make_unique<AdamW>(
        &graph,
        &model,
        static_cast<Scalar>(learning_rate),
        0.9f,
        0.999f,
        1e-8f,
        0.0f);

    std::vector<std::int64_t> pos_data(
        static_cast<std::size_t>(n_seq * n_batch));
    fill_arange_position_ids(pos_data, n_seq, n_batch);
    std::vector<std::int64_t> tt_data(
        static_cast<std::size_t>(n_seq * n_batch), 0);

    const Scalar ce_scale = 1.0f / static_cast<Scalar>(n_seq * n_batch);
    bool bound_optimizer_state = false;

    float first_loss = -1.f;
    float last_loss = -1.f;

    for (std::size_t step = 0; step < num_batches; ++step)
    {
        if (step > 0)
        {
            graph.reset_incremental_tile_state();
        }

        for (NNGraph::TensorNode *p : graph.parameters())
        {
            if (p->grad() != nullptr)
            {
                graph::tensor::clear(p->grad()->data());
            }
        }

        ToyMlmBatch batch =
            make_mlm_batch(n_seq, n_batch, config.vocab_size, gen, mask_token_id);

        auto *logits = model.forward(
            input_ids, token_type_ids, position_ids, nullptr);
        auto *loss = cross_entropy(logits, labels, 0, ce_scale, -100)
                         ->set_name("loss");
        loss->mark_output(true);

        auto [loss_grad, loss_grad_first] =
            graph.get_or_create_grad(loss, "loss_grad");
        (void)loss_grad_first;
        graph::tensor::fill(Scalar(1.0), loss_grad->data());
        loss->backward(true);
        optimizer->step(static_cast<Scalar>(learning_rate));

        graph.finish_phase();
        graph.lower_and_compile();
        Runtime &runtime = graph.runtime();

        if (!bound_optimizer_state)
        {
            for (const auto &[sname, stensor] : optimizer->named_state_tensors())
            {
                (void)sname;
                Index n = 1;
                for (auto d : stensor->shape())
                {
                    n *= d;
                }
                std::vector<float> zeros(static_cast<std::size_t>(n), 0.0f);
                runtime.bind_data(stensor, zeros);
            }
            bound_optimizer_state = true;
        }

        runtime.bind_data(input_ids, batch.input_ids);
        runtime.bind_data(labels, batch.labels);
        runtime.bind_data(position_ids, pos_data);
        runtime.bind_data(token_type_ids, tt_data);

        runtime.execute();
        runtime.wait();

        float loss_val = runtime.get_output<float>(loss)[0];
        if (step == 0)
        {
            first_loss = loss_val;
        }
        last_loss = loss_val;
        std::cout << "Batch " << step << "  loss=" << loss_val << "\n";

        for (NNGraph::TensorNode *ptensor : graph.parameters())
        {
            sync_param_hint_from_runtime(runtime, ptensor);
        }
        for (const auto &[sname, stensor] : optimizer->named_state_tensors())
        {
            (void)sname;
            sync_param_hint_from_runtime(runtime, stensor);
        }
    }

    std::cout << "First loss=" << first_loss << "  last loss=" << last_loss
              << "\n";
    if (!(last_loss < first_loss))
    {
        std::cerr << "bert_graph_training: loss did not decrease\n";
        return EXIT_ERROR;
    }
    return EXIT_OK;
}
