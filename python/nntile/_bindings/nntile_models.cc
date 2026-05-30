/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file python/nntile/_bindings/nntile_models.cc
 * Model, module, optimizer, context, and dataset bindings for examples.
 *
 * @version 1.1.0
 * */

#include <cstring>
#include <memory>
#include <nntile/context.hh>
#include <nntile/dataset/causal_lm_mmap.hh>
#include <nntile/model/gpt2/gpt2_causal.hh>
#include <nntile/model/gpt2/gpt2_config.hh>
#include <nntile/module/activation.hh>
#include <nntile/module/linear.hh>
#include <nntile/module/mlp.hh>
#include <nntile/optim/adamw.hh>
#include <nntile/runtime.hh>
#include <random>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using pybind11::literals::operator""_a;
using namespace nntile;

namespace
{

void init_random_parameter_hints(module::Module &mod, unsigned seed)
{
    std::mt19937 gen(seed);
    for (NNGraph::TensorNode *tensor : mod.parameters_recursive())
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
        float const limit = std::sqrt(1.0f / fan_in);
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
    mod.mark_parameters_input_recursive();
}

model::gpt2::Gpt2Config make_tiny_gpt2_config()
{
    model::gpt2::Gpt2Config c;
    c.vocab_size = 256;
    c.hidden_size = 64;
    c.intermediate_size = 128;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.max_position_embeddings = 512;
    c.layer_norm_eps = 1e-5f;
    c.validate();
    return c;
}

} // namespace

void bind_nntile_models(py::module_ &m)
{
    // Context
    py::class_<Context>(m, "Context")
        .def(py::init<int,
                 int,
                 int,
                 const char *,
                 std::size_t,
                 int,
                 const char *,
                 int,
                 int>(),
            "ncpu"_a = -1,
            "ncuda"_a = -1,
            "ooc"_a = 0,
            "ooc_path"_a = "/tmp/nntile_ooc",
            "ooc_size"_a = 16777216,
            "logger"_a = 0,
            "logger_addr"_a = "localhost",
            "logger_port"_a = 5001,
            "verbose"_a = 0)
        .def("shutdown", &Context::shutdown)
        .def("restrict_cpu", &Context::restrict_cpu)
        .def("restrict_cuda", &Context::restrict_cuda)
        .def("restore_where", &Context::restore_where);

    py::enum_<module::ActivationType>(m, "ActivationType")
        .value("GELU", module::ActivationType::GELU)
        .value("GELUTANH", module::ActivationType::GELUTANH)
        .value("RELU", module::ActivationType::RELU)
        .value("SILU", module::ActivationType::SILU);

    py::class_<module::Linear, module::Module>(m, "Linear")
        .def("weight_tensor",
            [](module::Linear &l) { return l.weight_tensor(); },
            py::return_value_policy::reference)
        .def("bias_tensor",
            [](module::Linear &l) { return l.bias_tensor(); },
            py::return_value_policy::reference);

    py::class_<module::Mlp, module::Module>(m, "Mlp")
        .def(py::init<NNGraph *,
                 const std::string &,
                 Index,
                 Index,
                 Index,
                 module::ActivationType,
                 DataType>(),
            "graph"_a,
            "name"_a,
            "input_dim"_a,
            "intermediate_dim"_a,
            "output_dim"_a,
            "activation"_a = module::ActivationType::GELU,
            "dtype"_a = DataType::FP32,
            py::keep_alive<1, 2>())
        .def("forward",
            &module::Mlp::forward,
            "input"_a,
            py::return_value_policy::reference)
        .def(
            "fc1",
            [](module::Mlp &m) -> module::Linear & { return m.fc1(); },
            py::return_value_policy::reference)
        .def(
            "fc2",
            [](module::Mlp &m) -> module::Linear & { return m.fc2(); },
            py::return_value_policy::reference);

    py::class_<model::gpt2::Gpt2Config>(m, "Gpt2Config")
        .def(py::init<>())
        .def_readwrite("vocab_size", &model::gpt2::Gpt2Config::vocab_size)
        .def_readwrite("hidden_size", &model::gpt2::Gpt2Config::hidden_size)
        .def_readwrite(
            "intermediate_size", &model::gpt2::Gpt2Config::intermediate_size)
        .def_readwrite(
            "num_hidden_layers", &model::gpt2::Gpt2Config::num_hidden_layers)
        .def_readwrite("num_attention_heads",
            &model::gpt2::Gpt2Config::num_attention_heads)
        .def_readwrite("max_position_embeddings",
            &model::gpt2::Gpt2Config::max_position_embeddings)
        .def_readwrite("layer_norm_eps", &model::gpt2::Gpt2Config::layer_norm_eps)
        .def("validate", &model::gpt2::Gpt2Config::validate);

    m.def("make_tiny_gpt2_config", &make_tiny_gpt2_config,
        "Built-in tiny GPT-2 config (matches C++ make_tiny_config).");

    py::class_<model::gpt2::Gpt2Causal, module::Module>(m, "Gpt2Causal")
        .def(py::init<NNGraph *, const std::string &, const model::gpt2::Gpt2Config &,
                 DataType>(),
            "graph"_a,
            "name"_a,
            "config"_a,
            "dtype"_a = DataType::FP32,
            py::keep_alive<1, 2>())
        .def("forward",
            &model::gpt2::Gpt2Causal::forward,
            "input_ids"_a,
            "position_ids"_a,
            "mask"_a = nullptr,
            "causal"_a = false,
            py::return_value_policy::reference)
        .def("load", &model::gpt2::Gpt2Causal::load, "path"_a, "strict"_a = true);

    py::class_<optim::AdamW>(m, "AdamW")
        .def(py::init<NNGraph *,
                 module::Module *,
                 Scalar,
                 Scalar,
                 Scalar,
                 Scalar,
                 Scalar>(),
            "graph"_a,
            "module"_a,
            "lr"_a = 0.001,
            "beta_1"_a = 0.9,
            "beta_2"_a = 0.999,
            "eps"_a = 1e-8,
            "weight_decay"_a = 0.01,
            py::keep_alive<1, 2>(),
            py::keep_alive<1, 3>())
        .def("step",
            py::overload_cast<Scalar>(&optim::AdamW::step),
            "lr"_a)
        .def("step", py::overload_cast<>(&optim::AdamW::step))
        .def("repr", &optim::AdamW::repr)
        .def(
            "named_state_tensors",
            [](const optim::AdamW &opt)
            {
                py::list out;
                for (const auto &entry : opt.named_state_tensors())
                {
                    out.append(py::make_tuple(
                        entry.first,
                        py::cast(
                            entry.second,
                            py::return_value_policy::reference)));
                }
                return out;
            });

    m.def("init_random_parameter_hints", &init_random_parameter_hints,
        "module"_a, "seed"_a = 42u);

    py::class_<dataset::CausalLmBatch>(m, "CausalLmBatch")
        .def(py::init<>())
        .def_readwrite("input_ids", &dataset::CausalLmBatch::input_ids)
        .def_readwrite("target_ids", &dataset::CausalLmBatch::target_ids);

    py::class_<dataset::CausalLmBatchConfig>(m, "CausalLmBatchConfig")
        .def(py::init<>())
        .def_readwrite("n_seq", &dataset::CausalLmBatchConfig::n_seq)
        .def_readwrite("n_batch", &dataset::CausalLmBatchConfig::n_batch)
        .def_readwrite("shuffle", &dataset::CausalLmBatchConfig::shuffle)
        .def_readwrite("seed", &dataset::CausalLmBatchConfig::seed);

    py::class_<dataset::TokenMemoryMap>(m, "TokenMemoryMap")
        .def(py::init<std::string>(), "path"_a)
        .def("num_tokens", &dataset::TokenMemoryMap::num_tokens);

    py::class_<dataset::CausalLmBatchIterator>(m, "CausalLmBatchIterator")
        .def(py::init<const dataset::TokenMemoryMap &,
                 const dataset::CausalLmBatchConfig &,
                 Index>(),
            "tokens"_a,
            "config"_a,
            "vocab_size"_a)
        .def("next",
            [](dataset::CausalLmBatchIterator &it, dataset::CausalLmBatch &batch)
            { return it.next(batch); },
            "batch"_a)
        .def("num_batches", &dataset::CausalLmBatchIterator::num_batches);
}
