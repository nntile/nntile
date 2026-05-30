/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file wrappers/python/nntile/nntile.cc
 * Python extension module for the NNTile Graph API.
 *
 * @version 1.1.0
 * */

#include <cstring>
#include <memory>
#include <nntile/context.hh>
#include <nntile/graph.hh>
#include <nntile/nn/graph_ops.hh>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

void bind_nntile_models(pybind11::module_ &m);
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using pybind11::literals::operator""_a;
using namespace nntile;
using namespace nntile;

//! Owns a TileGraph and its ``nntile::Runtime`` executor.
struct PyGraphRuntime
{
    std::shared_ptr<TileGraph> tile_graph;
    Runtime runtime;
    explicit PyGraphRuntime(std::shared_ptr<TileGraph> g) :
        tile_graph(std::move(g)), runtime(*tile_graph)
    {
    }
};

//! Non-owning view of ``NNGraph::runtime()`` after ``lower_and_compile``.
struct PyRuntimeView
{
    Runtime *runtime = nullptr;
    explicit PyRuntimeView(Runtime &rt) : runtime(&rt)
    {
    }
};

namespace pybind11
{
namespace detail
{
template <> struct is_copy_constructible<TensorGraph> : std::false_type
{
};
template <> struct is_copy_constructible<TileGraph> : std::false_type
{
};
template <> struct is_copy_constructible<NNGraph> : std::false_type
{
};
template <> struct is_copy_constructible<PyGraphRuntime> : std::false_type
{
};
} // namespace detail
} // namespace pybind11

// ---------------------------------------------------------------------------
// Helpers for numpy <-> Runtime data transfer
// ---------------------------------------------------------------------------

template <typename TensorPtr>
static void runtime_bind_numpy(
    Runtime &rt, TensorPtr tensor, py::array arr)
{
    DataType dt = rt.get_dtype(tensor);
    arr = py::array::ensure(arr);
    if (!arr)
        throw std::runtime_error("bind_data: cannot convert to numpy array");
    py::buffer_info buf = arr.request();
    size_t count = static_cast<size_t>(buf.size);

    switch (dt)
    {
    case DataType::FP32:
    case DataType::FP32_FAST_TF32:
    case DataType::FP32_FAST_FP16:
    case DataType::FP32_FAST_BF16:
    case DataType::FP16:
    case DataType::BF16:
    {
        auto f = py::array_t<float>::ensure(arr);
        if (!f)
            throw std::runtime_error(
                "bind_data: cannot convert to float32 array");
        rt.bind_data<float>(tensor, f.data(), count);
        break;
    }
    case DataType::FP64:
    {
        auto d = py::array_t<double>::ensure(arr);
        if (!d)
            throw std::runtime_error(
                "bind_data: cannot convert to float64 array");
        rt.bind_data<double>(tensor, d.data(), count);
        break;
    }
    case DataType::INT64:
    {
        auto i = py::array_t<std::int64_t>::ensure(arr);
        if (!i)
            throw std::runtime_error(
                "bind_data: cannot convert to int64 array");
        rt.bind_data<std::int64_t>(tensor, i.data(), count);
        break;
    }
    case DataType::BOOL:
    {
        auto b = py::array_t<std::uint8_t>::ensure(arr);
        if (!b)
            throw std::runtime_error(
                "bind_data: cannot convert to bool/uint8 array");
        rt.bind_data<std::uint8_t>(tensor, b.data(), count);
        break;
    }
    default:
        throw std::runtime_error("bind_data: unsupported dtype");
    }
}

static void runtime_bind_numpy_nn(
    PyGraphRuntime &s, NNGraph::TensorNode const *tensor, py::array arr)
{
    runtime_bind_numpy(s.runtime, tensor, arr);
}

template <typename TensorPtr>
static py::array runtime_get_numpy(Runtime &rt, TensorPtr tensor)
{
    DataType dt = rt.get_dtype(tensor);
    switch (dt)
    {
    case DataType::FP32:
    case DataType::FP32_FAST_TF32:
    case DataType::FP32_FAST_FP16:
    case DataType::FP32_FAST_BF16:
    case DataType::FP16:
    case DataType::BF16:
    {
        auto v = rt.get_output<float>(tensor);
        auto arr = py::array_t<float>(v.size());
        std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
        return arr;
    }
    case DataType::FP64:
    {
        auto v = rt.get_output<double>(tensor);
        auto arr = py::array_t<double>(v.size());
        std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(double));
        return arr;
    }
    case DataType::INT64:
    {
        auto v = rt.get_output<std::int64_t>(tensor);
        auto arr = py::array_t<std::int64_t>(v.size());
        std::memcpy(
            arr.mutable_data(), v.data(), v.size() * sizeof(std::int64_t));
        return arr;
    }
    case DataType::BOOL:
    {
        auto v = rt.get_output<std::uint8_t>(tensor);
        auto arr = py::array_t<std::uint8_t>(v.size());
        std::memcpy(arr.mutable_data(), v.data(), v.size());
        return arr;
    }
    default:
        throw std::runtime_error("get_output: unsupported dtype");
    }
}

static py::array runtime_get_numpy_nn(
    PyGraphRuntime &s, NNGraph::TensorNode const *tensor)
{
    return runtime_get_numpy(s.runtime, tensor);
}

static void sync_param_hint_from_runtime(
    Runtime &runtime, NNGraph::TensorNode *t)
{
    std::vector<std::uint8_t> bytes;
    switch (t->dtype())
    {
    case DataType::FP64:
    {
        auto d = runtime.get_output<double>(t);
        bytes.resize(d.size() * sizeof(double));
        std::memcpy(bytes.data(), d.data(), bytes.size());
        break;
    }
    case DataType::INT64:
    {
        auto d = runtime.get_output<std::int64_t>(t);
        bytes.resize(d.size() * sizeof(std::int64_t));
        std::memcpy(bytes.data(), d.data(), bytes.size());
        break;
    }
    default:
    {
        auto d = runtime.get_output<float>(t);
        bytes.resize(d.size() * sizeof(float));
        std::memcpy(bytes.data(), d.data(), bytes.size());
        break;
    }
    }
    t->data()->set_bind_hint(std::move(bytes));
}

static void bind_runtime_methods(py::class_<PyRuntimeView> &cls)
{
    cls.def("compile", [](PyRuntimeView &s) { s.runtime->compile(); })
        .def(
            "bind_data",
            [](PyRuntimeView &s, NNGraph::TensorNode const *t, py::array a)
            { runtime_bind_numpy(*s.runtime, t, a); },
            "tensor"_a,
            "data"_a)
        .def(
            "bind_data",
            [](PyRuntimeView &s,
                TensorGraph::TensorNode const *t,
                py::array a) { runtime_bind_numpy(*s.runtime, t, a); },
            "tensor"_a,
            "data"_a)
        .def("execute", [](PyRuntimeView &s) { s.runtime->execute(); })
        .def("wait", [](PyRuntimeView &s) { s.runtime->wait(); })
        .def(
            "get_output",
            [](PyRuntimeView &s, NNGraph::TensorNode const *t)
            { return runtime_get_numpy(*s.runtime, t); },
            "tensor"_a)
        .def(
            "get_output",
            [](PyRuntimeView &s, TensorGraph::TensorNode const *t)
            { return runtime_get_numpy(*s.runtime, t); },
            "tensor"_a)
        .def_property_readonly("is_compiled",
            [](const PyRuntimeView &s) { return s.runtime->is_compiled(); })
        .def(
            "sync_param_hint_from_runtime",
            [](PyRuntimeView &s, NNGraph::TensorNode *tensor)
            { sync_param_hint_from_runtime(*s.runtime, tensor); },
            "tensor"_a);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(nntile, m)
{
    m.doc() =
        "NNTile Graph API - computation graph with autograd. Execute via "
        "NNGraph.tensor_graph() -> TileGraph.from_tensor_graph -> Runtime.";

    // -----------------------------------------------------------------------
    // DataType enum
    // -----------------------------------------------------------------------
    py::enum_<DataType>(m, "DataType")
        .value("FP32", DataType::FP32)
        .value("FP32_FAST_TF32", DataType::FP32_FAST_TF32)
        .value("FP32_FAST_FP16", DataType::FP32_FAST_FP16)
        .value("FP32_FAST_BF16", DataType::FP32_FAST_BF16)
        .value("FP64", DataType::FP64)
        .value("FP16", DataType::FP16)
        .value("BF16", DataType::BF16)
        .value("INT64", DataType::INT64)
        .value("BOOL", DataType::BOOL);

    m.def("dtype_to_string", &dtype_to_string, "name"_a);

    // -----------------------------------------------------------------------
    // TensorGraph::TensorNode (low-level data node)
    // -----------------------------------------------------------------------
    py::class_<TensorGraph::TensorNode>(m, "TensorDataNode")
        .def_property_readonly("name", &TensorGraph::TensorNode::name)
        .def_property_readonly("shape", &TensorGraph::TensorNode::shape)
        .def_property_readonly("dtype", &TensorGraph::TensorNode::dtype)
        .def_property_readonly("ndim", &TensorGraph::TensorNode::ndim)
        .def_property_readonly("nelems", &TensorGraph::TensorNode::nelems)
        .def_property_readonly("is_input", &TensorGraph::TensorNode::is_input)
        .def_property_readonly(
            "is_output", &TensorGraph::TensorNode::is_output)
        .def("mark_input", &TensorGraph::TensorNode::mark_input, "v"_a = true)
        .def(
            "mark_output", &TensorGraph::TensorNode::mark_output, "v"_a = true)
        .def("set_name",
            &TensorGraph::TensorNode::set_name,
            "new_name"_a,
            py::return_value_policy::reference)
        .def("__repr__", &TensorGraph::TensorNode::to_string);

    // -----------------------------------------------------------------------
    // TensorGraph (low-level computation graph)
    // -----------------------------------------------------------------------
    py::class_<TensorGraph, std::unique_ptr<TensorGraph, py::nodelete>>(
        m, "TensorGraph")
        .def_property_readonly("name", &TensorGraph::name)
        .def_property_readonly("num_data", &TensorGraph::num_data)
        .def_property_readonly("num_ops", &TensorGraph::num_ops)
        .def("data_names", &TensorGraph::data_names)
        .def("__repr__", &TensorGraph::to_string)
        .def("to_mermaid", &TensorGraph::to_mermaid);

    // -----------------------------------------------------------------------
    // TileGraph (lowered tile-level graph; use with Runtime)
    // -----------------------------------------------------------------------
    py::class_<TileGraph, std::shared_ptr<TileGraph>>(m, "TileGraph")
        .def(py::init<std::string>(), "name"_a = "")
        .def_static(
            "from_tensor_graph",
            [](const TensorGraph &tg)
            {
                return std::make_shared<TileGraph>(
                    TileGraph::from_tensor_graph(tg));
            },
            "tensor_graph"_a)
        .def_property_readonly("name", &TileGraph::name)
        .def_property_readonly("num_data", &TileGraph::num_data)
        .def_property_readonly("num_ops", &TileGraph::num_ops)
        .def_property_readonly("num_tensors", &TileGraph::num_tensors)
        .def("data_names", &TileGraph::data_names)
        .def("__repr__", &TileGraph::to_string)
        .def("to_mermaid", &TileGraph::to_mermaid);

    // -----------------------------------------------------------------------
    // Graph execution: nntile::Runtime (Python class name ``Runtime``).
    // -----------------------------------------------------------------------
    py::class_<PyGraphRuntime>(m,
        "Runtime",
        "Tile graph executor (C++ ``nntile::Runtime``). "
        "Build: TileGraph.from_tensor_graph(nn_graph.tensor_graph()), then "
        "Runtime(tile_graph).")
        .def(py::init<std::shared_ptr<TileGraph>>(), "tile_graph"_a)
        .def("compile", [](PyGraphRuntime &s) { s.runtime.compile(); })
        .def(
            "bind_data",
            [](PyGraphRuntime &s,
                TensorGraph::TensorNode const *t,
                py::array a) { runtime_bind_numpy(s.runtime, t, a); },
            "tensor"_a,
            "data"_a)
        .def(
            "bind_data",
            [](PyGraphRuntime &s, NNGraph::TensorNode const *t, py::array a)
            { runtime_bind_numpy_nn(s, t, a); },
            "tensor"_a,
            "data"_a)
        .def("execute", [](PyGraphRuntime &s) { s.runtime.execute(); })
        .def("wait", [](PyGraphRuntime &s) { s.runtime.wait(); })
        .def(
            "get_output",
            [](PyGraphRuntime &s, TensorGraph::TensorNode const *t)
            { return runtime_get_numpy(s.runtime, t); },
            "tensor"_a)
        .def(
            "get_output",
            [](PyGraphRuntime &s, NNGraph::TensorNode const *t)
            { return runtime_get_numpy_nn(s, t); },
            "tensor"_a)
        .def_property_readonly("is_compiled",
            [](const PyGraphRuntime &s) { return s.runtime.is_compiled(); });

    {
        py::class_<PyRuntimeView> gr(
            m,
            "GraphRuntime",
            "Executor view from ``NNGraph.runtime()`` after "
            "``lower_and_compile()``.");
        bind_runtime_methods(gr);
    }

    m.def(
        "sync_param_hint_from_runtime",
        [](PyRuntimeView runtime, NNGraph::TensorNode *tensor)
        { sync_param_hint_from_runtime(*runtime.runtime, tensor); },
        "runtime"_a,
        "tensor"_a);
    m.def(
        "sync_param_hint_from_runtime",
        [](PyGraphRuntime &runtime, NNGraph::TensorNode *tensor)
        { sync_param_hint_from_runtime(runtime.runtime, tensor); },
        "runtime"_a,
        "tensor"_a);
    m.def(
        "sync_param_hint_from_runtime",
        [](NNGraph &graph, NNGraph::TensorNode *tensor)
        {
            if (!graph.has_runtime())
            {
                throw std::runtime_error(
                    "sync_param_hint_from_runtime: call lower_and_compile() "
                    "first");
            }
            sync_param_hint_from_runtime(graph.runtime(), tensor);
        },
        "graph"_a,
        "tensor"_a);

    // -----------------------------------------------------------------------
    // NNGraph::TensorNode (autograd-aware tensor node)
    // -----------------------------------------------------------------------
    py::class_<NNGraph::TensorNode>(m, "TensorNode")
        .def_property_readonly("name", &NNGraph::TensorNode::name)
        .def_property_readonly("shape", &NNGraph::TensorNode::shape)
        .def_property_readonly("dtype", &NNGraph::TensorNode::dtype)
        .def_property_readonly("ndim", &NNGraph::TensorNode::ndim)
        .def_property_readonly(
            "requires_grad", &NNGraph::TensorNode::requires_grad)
        .def("set_requires_grad",
            &NNGraph::TensorNode::set_requires_grad,
            "value"_a)
        .def_property_readonly("has_grad", &NNGraph::TensorNode::has_grad)
        .def_property_readonly(
            "grad",
            [](NNGraph::TensorNode &t) -> NNGraph::TensorNode *
            { return t.grad(); },
            py::return_value_policy::reference)
        .def_property_readonly(
            "data",
            [](NNGraph::TensorNode &t) -> TensorGraph::TensorNode *
            { return t.data(); },
            py::return_value_policy::reference)
        .def_property_readonly("is_leaf", &NNGraph::TensorNode::is_leaf)
        .def_property_readonly("is_input", &NNGraph::TensorNode::is_input)
        .def_property_readonly("is_output", &NNGraph::TensorNode::is_output)
        .def("mark_input", &NNGraph::TensorNode::mark_input, "v"_a = true)
        .def("mark_output", &NNGraph::TensorNode::mark_output, "v"_a = true)
        .def("set_name",
            &NNGraph::TensorNode::set_name,
            "new_name"_a,
            py::return_value_policy::reference)
        .def("backward",
            &NNGraph::TensorNode::backward,
            "retain_graph"_a = false)
        .def("__repr__", &NNGraph::TensorNode::to_string);

    // -----------------------------------------------------------------------
    // NNGraph (autograd computation graph)
    // -----------------------------------------------------------------------
    py::class_<NNGraph>(m, "NNGraph")
        .def(py::init<const std::string &>(), "name"_a = "")
        .def_property_readonly("name", &NNGraph::name)
        .def_property_readonly("num_tensors", &NNGraph::num_tensors)
        .def_property_readonly("num_ops", &NNGraph::num_ops)
        .def("tensor",
            static_cast<NNGraph::TensorNode *(
                NNGraph::*) (std::vector<Index>, DataType, bool)>(
                &NNGraph::tensor),
            "shape"_a,
            "dtype"_a = DataType::FP32,
            "requires_grad"_a = true,
            py::return_value_policy::reference)
        .def("get_tensor",
            static_cast<NNGraph::TensorNode *(
                NNGraph::*) (TensorGraph::TensorNode const *)>(
                &NNGraph::get_tensor),
            "tensor_data"_a,
            py::return_value_policy::reference)
        .def("tensor_names", &NNGraph::tensor_names)
        .def(
            "parameters",
            [](const NNGraph &g)
            {
                py::list out;
                for (NNGraph::TensorNode *p : g.parameters())
                {
                    out.append(
                        py::cast(p, py::return_value_policy::reference));
                }
                return out;
            })
        .def(
            "named_parameters",
            [](const NNGraph &g)
            {
                py::list out;
                for (const auto &entry : g.named_parameters())
                {
                    out.append(py::make_tuple(
                        entry.first,
                        py::cast(
                            entry.second,
                            py::return_value_policy::reference)));
                }
                return out;
            })
        .def(
            "tensor_graph",
            [](NNGraph &g) -> TensorGraph * { return &g.tensor_graph(); },
            py::return_value_policy::reference_internal)
        .def(
            "get_or_create_grad",
            [](NNGraph &g,
                NNGraph::TensorNode *t,
                const std::string &grad_name) -> NNGraph::TensorNode *
            {
                auto [grad, is_first] = g.get_or_create_grad(t, grad_name);
                return grad;
            },
            "tensor"_a,
            "grad_name"_a,
            py::return_value_policy::reference)
        .def_property_readonly("grad_enabled", &NNGraph::is_grad_enabled)
        .def("set_grad_enabled", &NNGraph::set_grad_enabled, "enabled"_a)
        .def(
            "finish_phase",
            [](NNGraph &g, bool reset_autograd_state)
            { g.finish_phase(reset_autograd_state); },
            "reset_autograd_state"_a = true)
        .def("lower_and_compile",
            py::overload_cast<>(&NNGraph::lower_and_compile))
        .def("runtime",
            [](NNGraph &g) { return PyRuntimeView(g.runtime()); })
        .def("has_runtime", &NNGraph::has_runtime)
        .def("enable_auto_tensor_name_phase_suffix",
            &NNGraph::enable_auto_tensor_name_phase_suffix,
            "enable"_a = true)
        .def("reset_incremental_tile_state",
            &NNGraph::reset_incremental_tile_state)
        .def("__repr__", &NNGraph::to_string)
        .def("to_mermaid", &NNGraph::to_mermaid);

    // -----------------------------------------------------------------------
    // NN operations (free functions)
    // -----------------------------------------------------------------------
    auto nn = m.def_submodule("nn", "Neural network graph operations");

    nn.def("gemm",
        &nntile::gemm,
        "a"_a,
        "b"_a,
        "alpha"_a = 1.0f,
        "trans_a"_a = false,
        "trans_b"_a = false,
        "ndim"_a = 1,
        "batch_ndim"_a = 0,
        py::return_value_policy::reference);

    nn.def("transpose",
        &nntile::transpose,
        "src"_a,
        "ndim"_a,
        py::return_value_policy::reference);

    nn.def("rope",
        &nntile::rope,
        "sin"_a,
        "cos"_a,
        "x"_a,
        py::return_value_policy::reference);

    nn.def("sdpa_eager",
        &nntile::sdpa_eager,
        "q"_a,
        "k"_a,
        "v"_a,
        "mask"_a = nullptr,
        "batch_ndim"_a = 2,
        "redux"_a = 0,
        py::return_value_policy::reference);

    nn.def("scale_slice",
        &nntile::scale_slice,
        "alpha"_a,
        "src"_a,
        "axis"_a,
        "axis_size"_a,
        py::return_value_policy::reference);

    nn.def("scale",
        &nntile::scale,
        "alpha"_a,
        "src"_a,
        py::return_value_policy::reference);

    nn.def("add",
        &nntile::add,
        "alpha"_a,
        "x"_a,
        "beta"_a,
        "y"_a,
        py::return_value_policy::reference);

    nn.def("multiply",
        &nntile::multiply,
        "x"_a,
        "y"_a,
        "alpha"_a = 1.0f,
        py::return_value_policy::reference);

    nn.def("silu", &nntile::silu, "x"_a, py::return_value_policy::reference);

    nn.def("gelu", &nntile::gelu, "x"_a, py::return_value_policy::reference);

    nn.def("relu", &nntile::relu, "x"_a, py::return_value_policy::reference);

    nn.def("rms_norm",
        &nntile::rms_norm,
        "x"_a,
        "gamma"_a,
        "axis"_a = 0,
        "eps"_a = 1e-6f,
        "redux"_a = 0,
        py::return_value_policy::reference);

    nn.def("softmax",
        &nntile::softmax,
        "x"_a,
        "axis"_a = 0,
        "redux"_a = 0,
        py::return_value_policy::reference);

    nn.def("embedding",
        &nntile::embedding,
        "index"_a,
        "vocab"_a,
        "axis"_a = 0,
        "redux"_a = 0,
        py::return_value_policy::reference);

    nn.def("cross_entropy",
        &nntile::cross_entropy,
        "x"_a,
        "labels"_a,
        "redux"_a = 0,
        "scale"_a = 1.0f,
        "ignore_index"_a = -100,
        py::return_value_policy::reference);

    nn.def("fill", &nntile::fill, "val"_a, "x"_a);

    nn.def("clear", &nntile::clear, "x"_a);

    // -----------------------------------------------------------------------
    // Module base class
    // -----------------------------------------------------------------------
    py::class_<nntile::module::Module>(m, "Module")
        .def_property_readonly("name", &nntile::module::Module::name)
        .def("parameters",
            static_cast<std::vector<NNGraph::TensorNode *> (
                nntile::module::Module::*)() const>(
                &nntile::module::Module::parameters),
            py::return_value_policy::reference)
        .def("named_parameters",
            &nntile::module::Module::named_parameters,
            py::return_value_policy::reference)
        .def("parameters_recursive",
            &nntile::module::Module::parameters_recursive,
            py::return_value_policy::reference)
        .def("named_parameters_recursive",
            &nntile::module::Module::named_parameters_recursive,
            py::return_value_policy::reference)
        .def("parameter_gradients",
            &nntile::module::Module::parameter_gradients,
            py::return_value_policy::reference)
        .def("parameter_gradients_recursive",
            &nntile::module::Module::parameter_gradients_recursive,
            py::return_value_policy::reference)
        .def("children",
            &nntile::module::Module::children,
            py::return_value_policy::reference)
        .def("repr", &nntile::module::Module::repr)
        .def("__repr__", &nntile::module::Module::to_string)
        .def("load",
            &nntile::module::Module::load,
            "path"_a,
            "strict"_a = true)
        .def("mark_parameters_input_recursive",
            &nntile::module::Module::mark_parameters_input_recursive);

    bind_nntile_models(m);
}
