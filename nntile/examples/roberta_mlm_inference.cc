/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file examples/roberta_mlm_inference.cc
 * RoBERTa MLM forward-only inference demo on the graph API.
 *
 * @version 1.1.0
 * */

#include <iostream>
#include <random>
#include <vector>

#include "roberta_example_helpers.hh"
#include <nntile.hh>
#include <nntile/model/roberta/roberta_mlm.hh>

using namespace nntile;
using namespace nntile::examples;
using namespace nntile;
using namespace nntile::model::roberta;

int main()
{
    RobertaConfig config = make_tiny_roberta_config(1, 16);
    const Index n_seq = 4;
    const Index n_batch = 1;

    Context context(1, 0, 0, "/tmp/nntile_ooc", 16777216, 0, "localhost", 5001, 0);

    NNGraph graph("roberta_mlm_inference");
    RobertaMlm model(&graph, "model", config);

    auto *input_ids = graph.tensor({n_seq, n_batch}, DataType::INT64, false)
                          ->set_name("input_ids");
    auto *position_ids =
        graph.tensor({n_seq, n_batch}, DataType::INT64, false)
            ->set_name("position_ids");
    input_ids->mark_input(true);
    position_ids->mark_input(true);

    std::mt19937 gen(7);
    init_random_roberta_parameter_hints(
        model, gen, RobertaParamInitScale::Uniform05);

    auto *logits = model.forward(input_ids, position_ids, nullptr);
    logits->mark_output(true);

    graph.finish_phase();
    graph.lower_and_compile();
    Runtime &runtime = graph.runtime();

    std::vector<std::int64_t> ids = {2, 3, 4, 5};
    std::vector<std::int64_t> pos(4);
    fill_roberta_position_ids(pos, n_seq, n_batch, config.pad_token_id);

    runtime.bind_data(input_ids, ids);
    runtime.bind_data(position_ids, pos);
    runtime.execute();
    runtime.wait();

    auto out = runtime.get_output<float>(logits);
    std::cout << "RoBERTa MLM inference ok, logits.size=" << out.size()
              << "\n";
    return 0;
}
