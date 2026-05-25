/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file examples/bert_mlm_inference.cc
 * BERT MLM forward-only inference demo on the graph API.
 *
 * @version 1.1.0
 * */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include <nntile.hh>
#include <nntile/graph/model/bert/bert_mlm.hh>

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::bert;

namespace
{

static BertConfig make_tiny_config()
{
    BertConfig c;
    c.vocab_size = 64;
    c.hidden_size = 32;
    c.intermediate_size = 64;
    c.num_hidden_layers = 1;
    c.num_attention_heads = 4;
    c.max_position_embeddings = 16;
    c.validate();
    return c;
}

static void init_random_parameter_hints(BertMlm &model, std::mt19937 &gen)
{
    for (NNGraph::TensorNode *tensor : model.parameters_recursive())
    {
        Index nelems = 1;
        for (auto d : tensor->shape())
        {
            nelems *= d;
        }
        std::vector<float> data(static_cast<std::size_t>(nelems), 0.01f);
        std::uniform_real_distribution<float> wdist(-0.05f, 0.05f);
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

} // namespace

int main()
{
    BertConfig config = make_tiny_config();
    const Index n_seq = 4;
    const Index n_batch = 1;

    Context context(1, 0, 0, "/tmp/nntile_ooc", 16777216, 0, "localhost", 5001, 0);

    NNGraph graph("bert_mlm_inference");
    BertMlm model(&graph, "model", config);

    auto *input_ids = graph.tensor({n_seq, n_batch}, DataType::INT64, false)
                          ->set_name("input_ids");
    auto *token_type_ids =
        graph.tensor({n_seq, n_batch}, DataType::INT64, false)
            ->set_name("token_type_ids");
    auto *position_ids =
        graph.tensor({n_seq, n_batch}, DataType::INT64, false)
            ->set_name("position_ids");
    input_ids->mark_input(true);
    token_type_ids->mark_input(true);
    position_ids->mark_input(true);

    std::mt19937 gen(7);
    init_random_parameter_hints(model, gen);

    auto *logits = model.forward(
        input_ids, token_type_ids, position_ids, nullptr);
    logits->mark_output(true);

    graph.finish_phase();
    graph.lower_and_compile();
    Runtime &runtime = graph.runtime();

    std::vector<std::int64_t> ids = {1, 2, 3, 4};
    std::vector<std::int64_t> pos = {0, 1, 2, 3};
    std::vector<std::int64_t> tt = {0, 0, 0, 0};

    runtime.bind_data(input_ids, ids);
    runtime.bind_data(position_ids, pos);
    runtime.bind_data(token_type_ids, tt);
    runtime.execute();
    runtime.wait();

    auto out = runtime.get_output<float>(logits);
    std::cout << "BERT MLM inference ok, logits.size=" << out.size() << "\n";
    return 0;
}
