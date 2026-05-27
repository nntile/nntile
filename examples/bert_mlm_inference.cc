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

#include <iostream>
#include <random>
#include <vector>

#include "bert_example_helpers.hh"
#include <nntile.hh>
#include <nntile/graph/model/bert/bert_mlm.hh>

using namespace nntile::core;
using namespace nntile::examples;
using namespace nntile::graph;
using namespace nntile::graph::model::bert;

int main()
{
    BertConfig config = make_tiny_bert_config(1, 16);
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
    init_random_bert_parameter_hints(
        model, gen, BertParamInitScale::Uniform05);

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
