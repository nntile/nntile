/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor/phase_codec.cc
 * TensorGraph phase codec for Flush PhaseIR JSON.
 *
 * @version 1.1.0
 * */

#include <nntile/tensor/phase_codec.hh>

#include <nntile/dtype.hh>
#include <nntile/tensor/graph.hh>
#include <nntile/tensor/ops/add.hh>
#include <nntile/tensor/ops/fill.hh>
#include <nntile/tensor/ops/gemm.hh>
#include <nntile/tensor/ops/multiply.hh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace nntile::tensor
{

namespace
{

void throw_unknown_op(std::string const &name)
{
    throw std::runtime_error("UnknownOp:" + name);
}

std::string wire_op_name(std::string const &tensor_name)
{
    if (tensor_name == "MULTIPLY")
    {
        return "MUL";
    }
    if (tensor_name == "GEMM")
    {
        return "MM";
    }
    return tensor_name;
}

std::string phase_dtype_name(DataType dtype)
{
    switch (dtype)
    {
        case DataType::FP32:
            return "float32";
        case DataType::FP64:
            return "float64";
        case DataType::FP16:
            return "float16";
        case DataType::BF16:
            return "bfloat16";
        case DataType::INT64:
            return "int64";
        case DataType::BOOL:
            return "bool";
        default:
            return dtype_to_string(dtype);
    }
}

void require_v1_ops(nlohmann::json const &ops)
{
    if (!ops.is_array())
    {
        throw_unknown_op("");
    }
    for (auto const &op : ops)
    {
        std::string name;
        if (op.is_object())
        {
            name = op.value("op_name", std::string());
        }
        if (!is_v1_phase_op(name))
        {
            throw_unknown_op(name);
        }
    }
}

nlohmann::json encode_node(TensorGraph::TensorNode const &node)
{
    nlohmann::json shape = nlohmann::json::array();
    for (Index dim : node.shape())
    {
        shape.push_back(dim);
    }
    return {
        {"id", node.id()},
        {"shape", std::move(shape)},
        {"dtype", phase_dtype_name(node.dtype())},
        {"name", node.name()},
    };
}

nlohmann::json encode_op_attrs(TensorGraph::OpNode const &op)
{
    nlohmann::json attrs = nlohmann::json::object();
    if (auto const *fill = dynamic_cast<TensorFillOp const *>(&op))
    {
        attrs["value"] = fill->val;
        return attrs;
    }
    if (auto const *add = dynamic_cast<TensorAddOp const *>(&op))
    {
        attrs["alpha"] = add->alpha;
        attrs["beta"] = add->beta;
        return attrs;
    }
    if (auto const *mul = dynamic_cast<TensorMultiplyOp const *>(&op))
    {
        attrs["alpha"] = mul->alpha;
        return attrs;
    }
    if (auto const *gemm = dynamic_cast<TensorGemmOp const *>(&op))
    {
        attrs["alpha"] = gemm->alpha;
        attrs["beta"] = gemm->beta;
        attrs["trans_a"] = gemm->trans_a;
        attrs["trans_b"] = gemm->trans_b;
        attrs["ndim"] = gemm->ndim;
        attrs["batch_ndim"] = gemm->batch_ndim;
        return attrs;
    }
    return attrs;
}

nlohmann::json encode_op(TensorGraph::OpNode const &op)
{
    std::string const wire = wire_op_name(op.op_name());
    if (!is_v1_phase_op(wire))
    {
        throw_unknown_op(wire);
    }
    nlohmann::json inputs = nlohmann::json::array();
    for (TensorGraph::TensorNode const *in : op.inputs())
    {
        if (in != nullptr)
        {
            inputs.push_back(in->id());
        }
    }
    nlohmann::json outputs = nlohmann::json::array();
    for (TensorGraph::TensorNode const *out : op.outputs())
    {
        if (out != nullptr)
        {
            outputs.push_back(out->id());
        }
    }
    return {
        {"op_name", wire},
        {"inputs", std::move(inputs)},
        {"outputs", std::move(outputs)},
        {"attrs", encode_op_attrs(op)},
    };
}

nlohmann::json encode_ops_range(
    TensorGraph const &graph,
    size_t op_begin,
    size_t op_end,
    std::vector<TensorGraph::TensorNode const *> const &extra_nodes)
{
    auto const &ops = graph.ops();
    size_t end = std::min(op_end, ops.size());
    size_t begin = std::min(op_begin, end);

    std::unordered_set<TensorGraph::NodeId> seen;
    std::vector<TensorGraph::TensorNode const *> nodes;
    auto add_node = [&](TensorGraph::TensorNode const *node)
    {
        if (node == nullptr || !seen.insert(node->id()).second)
        {
            return;
        }
        nodes.push_back(node);
    };
    for (TensorGraph::TensorNode const *node : extra_nodes)
    {
        add_node(node);
    }

    nlohmann::json encoded_ops = nlohmann::json::array();
    for (size_t i = begin; i < end; ++i)
    {
        std::shared_ptr<TensorGraph::OpNode> const &op = ops[i];
        if (op == nullptr)
        {
            continue;
        }
        for (TensorGraph::TensorNode const *in : op->inputs())
        {
            add_node(in);
        }
        for (TensorGraph::TensorNode const *out : op->outputs())
        {
            add_node(out);
        }
        encoded_ops.push_back(encode_op(*op));
    }

    std::sort(
        nodes.begin(),
        nodes.end(),
        [](TensorGraph::TensorNode const *a,
            TensorGraph::TensorNode const *b)
        {
            return a->id() < b->id();
        });

    nlohmann::json encoded_nodes = nlohmann::json::array();
    for (TensorGraph::TensorNode const *node : nodes)
    {
        encoded_nodes.push_back(encode_node(*node));
    }
    return encode_phase(encoded_nodes, encoded_ops);
}

} // namespace

bool is_v1_phase_op(std::string const &op_name)
{
    for (char const *name : kV1PhaseOps)
    {
        if (op_name == name)
        {
            return true;
        }
    }
    return false;
}

nlohmann::json encode_phase(
    nlohmann::json const &nodes,
    nlohmann::json const &ops)
{
    nlohmann::json ops_arr = ops.is_null() ? nlohmann::json::array() : ops;
    require_v1_ops(ops_arr);
    nlohmann::json nodes_arr =
        nodes.is_null() ? nlohmann::json::array() : nodes;
    return {
        {"nodes", std::move(nodes_arr)},
        {"ops", std::move(ops_arr)},
    };
}

nlohmann::json encode_phase(TensorGraph const &graph)
{
    TensorGraph::PhaseSnapshot pending;
    pending.op_begin = graph.phase_seal_cursor();
    pending.op_end = graph.num_ops();
    return encode_phase(graph, pending);
}

nlohmann::json encode_phase(
    TensorGraph const &graph,
    TensorGraph::PhaseSnapshot const &phase)
{
    return encode_ops_range(
        graph,
        phase.op_begin,
        phase.op_end,
        phase.carried_tensors);
}

std::pair<nlohmann::json, nlohmann::json> decode_phase(
    nlohmann::json const &blob)
{
    nlohmann::json nodes = nlohmann::json::array();
    nlohmann::json ops = nlohmann::json::array();
    if (blob.is_object())
    {
        if (blob.contains("nodes") && !blob["nodes"].is_null())
        {
            nodes = blob["nodes"];
        }
        if (blob.contains("ops") && !blob["ops"].is_null())
        {
            ops = blob["ops"];
        }
    }
    require_v1_ops(ops);
    return {std::move(nodes), std::move(ops)};
}

} // namespace nntile::tensor
