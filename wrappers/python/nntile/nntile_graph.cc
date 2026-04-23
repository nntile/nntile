/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file wrappers/python/nntile/nntile_graph.cc
 * Python extension module for the NNTile Graph API.
 *
 * @version 1.1.0
 * */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nntile/graph.hh>
#include <nntile/graph/nn/graph_ops.hh>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using pybind11::literals::operator""_a;
using namespace nntile;
using namespace nntile::graph;

//! Owns a TileGraph lowered from TensorGraph so Runtime outlives the init call.
struct PyGraphRuntime
{
    TileGraph tile_graph;
    TileGraph::Runtime runtime;
    explicit PyGraphRuntime(const TensorGraph& g)
        : tile_graph(TileGraph::from_tensor_graph(g))
        , runtime(tile_graph)
    {
    }
};

namespace pybind11 { namespace detail {
template<>
struct is_copy_constructible<TensorGraph> : std::false_type {};
template<>
struct is_copy_constructible<NNGraph> : std::false_type {};
template<>
struct is_copy_constructible<PyGraphRuntime> : std::false_type {};
}} // namespace pybind11::detail

// ---------------------------------------------------------------------------
// Helpers for numpy <-> Runtime data transfer
// ---------------------------------------------------------------------------

static void runtime_bind_numpy(TileGraph::Runtime& rt,
                               const std::string& name,
                               py::array arr)
{
    DataType dt = rt.get_dtype(name);
    arr = py::array::ensure(arr);
    if(!arr)
        throw std::runtime_error("bind_data: cannot convert to numpy array");
    py::buffer_info buf = arr.request();
    size_t count = static_cast<size_t>(buf.size);

    switch(dt)
    {
        case DataType::FP32:
        case DataType::FP32_FAST_TF32:
        case DataType::FP32_FAST_FP16:
        case DataType::FP32_FAST_BF16:
        case DataType::FP16:
        case DataType::BF16:
        {
            auto f = py::array_t<float>::ensure(arr);
            if(!f)
                throw std::runtime_error(
                    "bind_data: cannot convert to float32 array");
            rt.bind_data<float>(name, f.data(), count);
            break;
        }
        case DataType::FP64:
        {
            auto d = py::array_t<double>::ensure(arr);
            if(!d)
                throw std::runtime_error(
                    "bind_data: cannot convert to float64 array");
            rt.bind_data<double>(name, d.data(), count);
            break;
        }
        case DataType::INT64:
        {
            auto i = py::array_t<std::int64_t>::ensure(arr);
            if(!i)
                throw std::runtime_error(
                    "bind_data: cannot convert to int64 array");
            rt.bind_data<std::int64_t>(name, i.data(), count);
            break;
        }
        case DataType::BOOL:
        {
            auto b = py::array_t<std::uint8_t>::ensure(arr);
            if(!b)
                throw std::runtime_error(
                    "bind_data: cannot convert to bool/uint8 array");
            rt.bind_data<std::uint8_t>(name, b.data(), count);
            break;
        }
        default:
            throw std::runtime_error("bind_data: unsupported dtype");
    }
}

static py::array runtime_get_numpy(TileGraph::Runtime& rt,
                                   const std::string& name)
{
    DataType dt = rt.get_dtype(name);
    switch(dt)
    {
        case DataType::FP32:
        case DataType::FP32_FAST_TF32:
        case DataType::FP32_FAST_FP16:
        case DataType::FP32_FAST_BF16:
        case DataType::FP16:
        case DataType::BF16:
        {
            auto v = rt.get_output<float>(name);
            auto arr = py::array_t<float>(v.size());
            std::memcpy(arr.mutable_data(), v.data(),
                        v.size() * sizeof(float));
            return arr;
        }
        case DataType::FP64:
        {
            auto v = rt.get_output<double>(name);
            auto arr = py::array_t<double>(v.size());
            std::memcpy(arr.mutable_data(), v.data(),
                        v.size() * sizeof(double));
            return arr;
        }
        case DataType::INT64:
        {
            auto v = rt.get_output<std::int64_t>(name);
            auto arr = py::array_t<std::int64_t>(v.size());
            std::memcpy(arr.mutable_data(), v.data(),
                        v.size() * sizeof(std::int64_t));
            return arr;
        }
        case DataType::BOOL:
        {
            auto v = rt.get_output<std::uint8_t>(name);
            auto arr = py::array_t<std::uint8_t>(v.size());
            std::memcpy(arr.mutable_data(), v.data(), v.size());
            return arr;
        }
        default:
            throw std::runtime_error("get_output: unsupported dtype");
    }
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(nntile_graph, m)
{
    m.doc() = "NNTile Graph API - computation graph with autograd";

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
        .def_property_readonly("is_output",
                               &TensorGraph::TensorNode::is_output)
        .def("mark_input", &TensorGraph::TensorNode::mark_input,
             "v"_a = true)
        .def("mark_output", &TensorGraph::TensorNode::mark_output,
             "v"_a = true)
        .def("__repr__", &TensorGraph::TensorNode::to_string);

    // -----------------------------------------------------------------------
    // TensorGraph (low-level computation graph)
    // -----------------------------------------------------------------------
    py::class_<TensorGraph,
               std::unique_ptr<TensorGraph, py::nodelete>>(m, "TensorGraph")
        .def_property_readonly("name", &TensorGraph::name)
        .def_property_readonly("num_data", &TensorGraph::num_data)
        .def_property_readonly("num_ops", &TensorGraph::num_ops)
        .def("data_names", &TensorGraph::data_names)
        .def("get_tensor_node",
             static_cast<TensorGraph::TensorNode* (TensorGraph::*)(
                 const std::string&)>(&TensorGraph::get_tensor_node),
             "name"_a, py::return_value_policy::reference)
        .def("__repr__", &TensorGraph::to_string)
        .def("to_mermaid", &TensorGraph::to_mermaid);

    // -----------------------------------------------------------------------
    // Graph execution: TileGraph::Runtime (lowers from TensorGraph on init)
    // -----------------------------------------------------------------------
    py::class_<PyGraphRuntime>(m, "Runtime")
        .def(py::init<const TensorGraph&>(), "graph"_a)
        .def("compile",
             [](PyGraphRuntime& s) { s.runtime.compile(); })
        .def("bind_data",
             [](PyGraphRuntime& s, const std::string& n, py::array a) {
                 runtime_bind_numpy(s.runtime, n, a);
             },
             "name"_a, "data"_a)
        .def("execute", [](PyGraphRuntime& s) { s.runtime.execute(); })
        .def("wait", [](PyGraphRuntime& s) { s.runtime.wait(); })
        .def("get_output",
             [](PyGraphRuntime& s, const std::string& n) {
                 return runtime_get_numpy(s.runtime, n);
             },
             "name"_a)
        .def_property_readonly("is_compiled", [](const PyGraphRuntime& s) {
            return s.runtime.is_compiled();
        });

    // -----------------------------------------------------------------------
    // NNGraph::TensorNode (autograd-aware tensor node)
    // -----------------------------------------------------------------------
    py::class_<NNGraph::TensorNode>(m, "TensorNode")
        .def_property_readonly("name", &NNGraph::TensorNode::name)
        .def_property_readonly("shape", &NNGraph::TensorNode::shape)
        .def_property_readonly("dtype", &NNGraph::TensorNode::dtype)
        .def_property_readonly("ndim", &NNGraph::TensorNode::ndim)
        .def_property_readonly("requires_grad",
                               &NNGraph::TensorNode::requires_grad)
        .def("set_requires_grad",
             &NNGraph::TensorNode::set_requires_grad, "value"_a)
        .def_property_readonly("has_grad", &NNGraph::TensorNode::has_grad)
        .def_property_readonly("grad",
             [](NNGraph::TensorNode& t) -> NNGraph::TensorNode* {
                 return t.grad();
             }, py::return_value_policy::reference)
        .def_property_readonly("data",
             [](NNGraph::TensorNode& t) -> TensorGraph::TensorNode* {
                 return t.data();
             }, py::return_value_policy::reference)
        .def_property_readonly("is_leaf", &NNGraph::TensorNode::is_leaf)
        .def_property_readonly("is_input", &NNGraph::TensorNode::is_input)
        .def_property_readonly("is_output", &NNGraph::TensorNode::is_output)
        .def("mark_input", &NNGraph::TensorNode::mark_input, "v"_a = true)
        .def("mark_output", &NNGraph::TensorNode::mark_output, "v"_a = true)
        .def("backward", &NNGraph::TensorNode::backward,
             "retain_graph"_a = false)
        .def("__repr__", &NNGraph::TensorNode::to_string);

    // -----------------------------------------------------------------------
    // NNGraph (autograd computation graph)
    // -----------------------------------------------------------------------
    py::class_<NNGraph>(m, "NNGraph")
        .def(py::init<const std::string&>(), "name"_a = "")
        .def_property_readonly("name", &NNGraph::name)
        .def_property_readonly("num_tensors", &NNGraph::num_tensors)
        .def_property_readonly("num_ops", &NNGraph::num_ops)
        .def("tensor",
             static_cast<NNGraph::TensorNode* (NNGraph::*)(
                 std::vector<Index>, const std::string&, DataType, bool)>(
                     &NNGraph::tensor),
             "shape"_a, "name"_a,
             "dtype"_a = DataType::FP32,
             "requires_grad"_a = true,
             py::return_value_policy::reference)
        .def("get_tensor",
             static_cast<NNGraph::TensorNode* (NNGraph::*)(
                 const std::string&)>(&NNGraph::get_tensor),
             "name"_a, py::return_value_policy::reference)
        .def("tensor_names", &NNGraph::tensor_names)
        .def("tensor_graph",
             [](NNGraph& g) -> TensorGraph* {
                 return &g.tensor_graph();
             }, py::return_value_policy::reference_internal)
        .def("get_or_create_grad",
             [](NNGraph& g, NNGraph::TensorNode* t,
                const std::string& grad_name) -> NNGraph::TensorNode* {
                 auto [grad, is_first] =
                     g.get_or_create_grad(t, grad_name);
                 return grad;
             },
             "tensor"_a, "grad_name"_a,
             py::return_value_policy::reference)
        .def_property_readonly("grad_enabled", &NNGraph::is_grad_enabled)
        .def("set_grad_enabled", &NNGraph::set_grad_enabled, "enabled"_a)
        .def("__repr__", &NNGraph::to_string)
        .def("to_mermaid", &NNGraph::to_mermaid);

    // -----------------------------------------------------------------------
    // NN operations (free functions)
    // -----------------------------------------------------------------------
    auto nn = m.def_submodule("nn", "Neural network graph operations");

    nn.def("gemm", &graph::gemm,
           "a"_a, "b"_a, "output_name"_a,
           "alpha"_a = 1.0f, "trans_a"_a = false, "trans_b"_a = false,
           "ndim"_a = 1, "batch_ndim"_a = 0,
           py::return_value_policy::reference);

    nn.def("transpose", &graph::transpose,
           "src"_a, "output_name"_a, "ndim"_a,
           py::return_value_policy::reference);

    nn.def("rope", &graph::rope,
           "sin"_a, "cos"_a, "x"_a, "output_name"_a,
           py::return_value_policy::reference);

    nn.def("sdpa_eager", &graph::sdpa_eager,
           "q"_a, "k"_a, "v"_a, "output_name"_a,
           "mask"_a = nullptr, "batch_ndim"_a = 2, "redux"_a = 0,
           py::return_value_policy::reference);

    nn.def("scale_slice", &graph::scale_slice,
           "alpha"_a, "src"_a, "output_name"_a, "axis"_a, "axis_size"_a,
           py::return_value_policy::reference);

    nn.def("scale", &graph::scale,
           "alpha"_a, "src"_a, "output_name"_a,
           py::return_value_policy::reference);

    nn.def("add", &graph::add,
           "alpha"_a, "x"_a, "beta"_a, "y"_a, "output_name"_a,
           py::return_value_policy::reference);

    nn.def("multiply", &graph::multiply,
           "x"_a, "y"_a, "output_name"_a, "alpha"_a = 1.0f,
           py::return_value_policy::reference);

    nn.def("silu", &graph::silu,
           "x"_a, "output_name"_a,
           py::return_value_policy::reference);

    nn.def("gelu", &graph::gelu,
           "x"_a, "output_name"_a,
           py::return_value_policy::reference);

    nn.def("relu", &graph::relu,
           "x"_a, "output_name"_a,
           py::return_value_policy::reference);

    nn.def("rms_norm", &graph::rms_norm,
           "x"_a, "gamma"_a, "output_name"_a,
           "axis"_a = 0, "eps"_a = 1e-6f, "redux"_a = 0,
           py::return_value_policy::reference);

    nn.def("softmax", &graph::softmax,
           "x"_a, "output_name"_a, "axis"_a = 0, "redux"_a = 0,
           py::return_value_policy::reference);

    nn.def("embedding", &graph::embedding,
           "index"_a, "vocab"_a, "output_name"_a,
           "axis"_a = 0, "redux"_a = 0,
           py::return_value_policy::reference);

    nn.def("cross_entropy", &graph::cross_entropy,
           "x"_a, "labels"_a, "output_name"_a,
           "redux"_a = 0, "scale"_a = 1.0f, "ignore_index"_a = -100,
           py::return_value_policy::reference);

    nn.def("fill", &graph::fill,
           "val"_a, "x"_a);

    nn.def("clear", &graph::clear,
           "x"_a);

    // -----------------------------------------------------------------------
    // Module base class
    // -----------------------------------------------------------------------
    py::class_<graph::module::Module>(m, "Module")
        .def_property_readonly("name", &graph::module::Module::name)
        .def("parameters",
             static_cast<std::vector<NNGraph::TensorNode*>
                 (graph::module::Module::*)() const>(
                     &graph::module::Module::parameters),
             py::return_value_policy::reference)
        .def("named_parameters",
             &graph::module::Module::named_parameters,
             py::return_value_policy::reference)
        .def("parameters_recursive",
             &graph::module::Module::parameters_recursive,
             py::return_value_policy::reference)
        .def("named_parameters_recursive",
             &graph::module::Module::named_parameters_recursive,
             py::return_value_policy::reference)
        .def("parameter_gradients",
             &graph::module::Module::parameter_gradients,
             py::return_value_policy::reference)
        .def("parameter_gradients_recursive",
             &graph::module::Module::parameter_gradients_recursive,
             py::return_value_policy::reference)
        .def("children", &graph::module::Module::children,
             py::return_value_policy::reference)
        .def("repr", &graph::module::Module::repr)
        .def("__repr__", &graph::module::Module::to_string);
}
