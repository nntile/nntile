/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file examples/bert_graph_training.cc
 * BERT masked-LM training on the graph API (tiny demo).
 *
 * Tiny MLM demo: scratch weights give much higher loss than after training or
 * after loading a saved checkpoint for the next step (loss need not go to zero).
 *
 * @version 1.1.0
 * */

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "bert_config_json.hh"
#include "bert_example_helpers.hh"
#include <nntile.hh>
#include <nntile/model/bert/bert_mlm.hh>
#include <nntile/tensor/ops/clear.hh>

using namespace nntile;
using namespace nntile::examples;
using namespace nntile;
using namespace nntile::model::bert;
using namespace nntile::optim;

namespace
{

constexpr int EXIT_OK = 0;
constexpr int EXIT_ERROR = 1;

static void fill_arange_position_ids(
    std::vector<std::int64_t> &pos, Index n_seq, Index n_batch)
{
    for (Index b = 0; b < n_batch; ++b)
    {
        for (Index s = 0; s < n_seq; ++s)
        {
            pos[b * n_seq + s] = static_cast<std::int64_t>(s);
        }
    }
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

static float run_forward_loss(
    BertMlm &model,
    NNGraph &graph,
    NNGraph::TensorNode *input_ids,
    NNGraph::TensorNode *token_type_ids,
    NNGraph::TensorNode *position_ids,
    NNGraph::TensorNode *labels,
    ToyMlmBatch const &batch,
    std::vector<std::int64_t> const &pos_data,
    std::vector<std::int64_t> const &tt_data,
    Scalar ce_scale)
{
    graph.reset_incremental_tile_state();

    auto *logits = model.forward(
        input_ids, token_type_ids, position_ids, nullptr);
    auto *loss = cross_entropy(logits, labels, 0, ce_scale, -100)
                     ->set_name("eval_loss");
    loss->mark_output(true);

    graph.finish_phase();
    graph.lower_and_compile();
    Runtime &runtime = graph.runtime();

    runtime.bind_data(input_ids, batch.input_ids);
    runtime.bind_data(labels, batch.labels);
    runtime.bind_data(position_ids, pos_data);
    runtime.bind_data(token_type_ids, tt_data);
    runtime.execute();
    runtime.wait();

    return runtime.get_output<float>(loss)[0];
}

} // namespace

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    BertConfig config = make_tiny_bert_config(2, 32, 1e-5f);
    const Index n_seq = 8;
    const Index n_batch = 2;
    const std::int64_t mask_token_id = 3;
    const std::size_t num_batches = 32;
    const float learning_rate = 0.05f;
    const char *const checkpoint_path =
        "/tmp/nntile_bert_graph_training_checkpoint.safetensors";

    Context context(1, 0, 0, "/tmp/nntile_ooc", 16777216, 0, "localhost", 5001, 0);

    NNGraph graph("bert_graph_training");
    graph.enable_auto_tensor_name_phase_suffix(true);
    BertMlm model(&graph, "model", config);

    auto *input_ids = graph.tensor({n_batch, n_seq}, DataType::INT64, false)
                          ->set_name("input_ids");
    auto *token_type_ids =
        graph.tensor({n_batch, n_seq}, DataType::INT64, false)
            ->set_name("token_type_ids");
    auto *position_ids =
        graph.tensor({n_batch, n_seq}, DataType::INT64, false)
            ->set_name("position_ids");
    auto *labels = graph.tensor({n_batch, n_seq}, DataType::INT64, false)
                       ->set_name("labels");
    input_ids->mark_input(true);
    token_type_ids->mark_input(true);
    position_ids->mark_input(true);
    labels->mark_input(true);

    std::mt19937 gen(42);
    init_random_bert_parameter_hints(model, gen);

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
    bool bound_persistent_tensors = false;

    float first_loss = -1.f;
    float best_loss = 1e30f;

    ToyMlmBatch const fixed_batch =
        make_mlm_batch(n_seq, n_batch, config.vocab_size, gen, mask_token_id);

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
                nntile::tensor::clear(p->grad()->data());
            }
        }

        ToyMlmBatch const &batch = fixed_batch;

        auto *logits = model.forward(
            input_ids, token_type_ids, position_ids, nullptr);
        auto *loss = cross_entropy(logits, labels, 0, ce_scale, -100)
                         ->set_name("loss");
        loss->mark_output(true);

        auto [loss_grad, loss_grad_first] =
            graph.get_or_create_grad(loss, "loss_grad");
        (void)loss_grad_first;
        nntile::tensor::fill(Scalar(1.0), loss_grad->data());
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

        if (!bound_persistent_tensors)
        {
            graph.bind_parameters(runtime);
            runtime.bind_data(position_ids, pos_data);
            runtime.bind_data(token_type_ids, tt_data);
            bound_persistent_tensors = true;
        }

        runtime.bind_data(input_ids, batch.input_ids);
        runtime.bind_data(labels, batch.labels);

        runtime.execute();
        runtime.wait();

        float loss_val = runtime.get_output<float>(loss)[0];
        if (step == 0)
        {
            first_loss = loss_val;
        }
        if (loss_val < best_loss)
        {
            best_loss = loss_val;
        }
        std::cout << "Batch " << step << "  loss=" << loss_val << "\n";
    }

    std::cout << "Scratch first loss=" << first_loss
              << "  best training loss=" << best_loss << "\n";
    if (!(best_loss < first_loss))
    {
        std::cerr << "bert_graph_training: training did not lower loss vs scratch\n";
        return EXIT_ERROR;
    }

    if (graph.has_runtime())
    {
        for (NNGraph::TensorNode *ptensor : graph.parameters())
        {
            sync_param_hint_from_runtime(graph.runtime(), ptensor);
        }
    }
    model.save(checkpoint_path);
    std::cout << "Saved checkpoint " << checkpoint_path << "\n";

    // Fresh graph: load checkpoint and evaluate on the same batch (continued
    // training step), not from scratch.
    NNGraph graph_loaded("bert_graph_training_loaded");
    graph_loaded.enable_auto_tensor_name_phase_suffix(true);
    BertMlm model_loaded(&graph_loaded, "model", config);
    model_loaded.load(checkpoint_path);
    model_loaded.mark_parameters_input_recursive();

    auto *input_ids2 =
        graph_loaded.tensor({n_batch, n_seq}, DataType::INT64, false)
            ->set_name("input_ids");
    auto *token_type_ids2 =
        graph_loaded.tensor({n_batch, n_seq}, DataType::INT64, false)
            ->set_name("token_type_ids");
    auto *position_ids2 =
        graph_loaded.tensor({n_batch, n_seq}, DataType::INT64, false)
            ->set_name("position_ids");
    auto *labels2 =
        graph_loaded.tensor({n_batch, n_seq}, DataType::INT64, false)
            ->set_name("labels");
    input_ids2->mark_input(true);
    token_type_ids2->mark_input(true);
    position_ids2->mark_input(true);
    labels2->mark_input(true);

    float const loaded_loss = run_forward_loss(
        model_loaded,
        graph_loaded,
        input_ids2,
        token_type_ids2,
        position_ids2,
        labels2,
        fixed_batch,
        pos_data,
        tt_data,
        ce_scale);

    std::cout << "Loaded checkpoint loss=" << loaded_loss << "\n";

    if (!(loaded_loss < first_loss * 0.5f))
    {
        std::cerr << "bert_graph_training: loaded loss should be much lower "
                     "than scratch ("
                  << loaded_loss << " vs scratch " << first_loss << ")\n";
        return EXIT_ERROR;
    }
    if (!(loaded_loss <= best_loss * 1.05f))
    {
        std::cerr << "bert_graph_training: loaded loss should match end of "
                     "training ("
                  << loaded_loss << " vs best " << best_loss << ")\n";
        return EXIT_ERROR;
    }

    return EXIT_OK;
}
