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

#include <nntile/defs.h>
#include <nntile/dtype.hh>
#include <nntile/tensor.hh>
#include <nntile/tensor/graph.hh>
#ifdef NNTILE_TORCH_NATIVE_OPS
#include <nntile/tensor/ops/torch_dispatch.hh>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
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

nlohmann::json encode_index_pair(std::array<Index, 2> const &v)
{
    return nlohmann::json::array({v[0], v[1]});
}

nlohmann::json encode_op_attrs(
    [[maybe_unused]] TensorGraph::OpNode const &op)
{
    nlohmann::json attrs = nlohmann::json::object();
#ifdef NNTILE_TORCH_NATIVE_OPS
    if (auto const *u = dynamic_cast<TensorTorchUnaryOp const *>(&op))
    {
        attrs["kind"] = static_cast<std::int32_t>(u->kind);
        return attrs;
    }
    if (auto const *b = dynamic_cast<TensorTorchBinaryOp const *>(&op))
    {
        attrs["kind"] = static_cast<std::int32_t>(b->kind);
        return attrs;
    }
    if (auto const *t = dynamic_cast<TensorTorchTernaryOp const *>(&op))
    {
        attrs["kind"] = static_cast<std::int32_t>(t->kind);
        return attrs;
    }
#endif
    if (auto const *o = dynamic_cast<TensorAddOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorAddInplaceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorAddFiberOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        attrs["axis"] = o->axis;
        attrs["batch_ndim"] = o->batch_ndim;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorAddFiberInplaceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        attrs["axis"] = o->axis;
        attrs["batch_ndim"] = o->batch_ndim;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorAddSliceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        attrs["axis"] = o->axis;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorAddSliceInplaceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        attrs["axis"] = o->axis;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorAdamStepOp const *>(&op))
    {
        attrs["num_iter"] = o->num_iter;
        attrs["beta_1"] = o->beta_1;
        attrs["beta_2"] = o->beta_2;
        attrs["eps"] = o->eps;
        attrs["lr"] = o->lr;
        attrs["weight_decay"] = o->weight_decay;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorAdamwStepOp const *>(&op))
    {
        attrs["num_iter"] = o->num_iter;
        attrs["beta_1"] = o->beta_1;
        attrs["beta_2"] = o->beta_2;
        attrs["eps"] = o->eps;
        attrs["lr"] = o->lr;
        attrs["weight_decay"] = o->weight_decay;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorSgdStepOp const *>(&op))
    {
        attrs["num_iter"] = o->num_iter;
        attrs["momentum"] = o->momentum;
        attrs["lr"] = o->lr;
        attrs["weight_decay"] = o->weight_decay;
        attrs["dampening"] = o->dampening;
        attrs["nesterov"] = o->nesterov;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorConcatOp const *>(&op))
    {
        attrs["axis"] = o->axis;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorConv2dInplaceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        attrs["padding"] = encode_index_pair(o->padding);
        attrs["stride"] = encode_index_pair(o->stride);
        attrs["dilation"] = encode_index_pair(o->dilation);
        return attrs;
    }
    if (auto const *o =
            dynamic_cast<TensorConv2dBwdInputInplaceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        attrs["padding"] = encode_index_pair(o->padding);
        attrs["stride"] = encode_index_pair(o->stride);
        attrs["dilation"] = encode_index_pair(o->dilation);
        return attrs;
    }
    if (auto const *o =
            dynamic_cast<TensorConv2dBwdWeightInplaceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        attrs["padding"] = encode_index_pair(o->padding);
        attrs["stride"] = encode_index_pair(o->stride);
        attrs["dilation"] = encode_index_pair(o->dilation);
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorCopyIntersectionOp const *>(&op))
    {
        attrs["src_offset"] = o->src_offset;
        attrs["dst_offset"] = o->dst_offset;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorEmbeddingOp const *>(&op))
    {
        attrs["axis"] = o->axis;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorEmbeddingBackwardOp const *>(&op))
    {
        attrs["axis"] = o->axis;
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        attrs["redux"] = o->redux;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorFillOp const *>(&op))
    {
        attrs["val"] = o->val;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorGeluBackwardOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorGelutanhBackwardOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorReluBackwardOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorSiluBackwardOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorGemmOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        attrs["trans_a"] = o->trans_a;
        attrs["trans_b"] = o->trans_b;
        attrs["ndim"] = o->ndim;
        attrs["batch_ndim"] = o->batch_ndim;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorHypotOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorHypotInplaceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorHypotScalarInverseOp const *>(&op))
    {
        attrs["eps"] = o->eps;
        attrs["alpha"] = o->alpha;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorLogScalarOp const *>(&op))
    {
        attrs["name"] = o->name;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorMaskScalarOp const *>(&op))
    {
        attrs["val"] = o->val;
        attrs["batch_ndim"] = o->batch_ndim;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorMaxsumexpOp const *>(&op))
    {
        attrs["axis"] = o->axis;
        attrs["beta"] = o->beta;
        attrs["redux"] = o->redux;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorMultiplyOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorMultiplyInplaceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorMultiplyFiberOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["axis"] = o->axis;
        return attrs;
    }
    if (auto const *o =
            dynamic_cast<TensorMultiplyFiberInplaceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["axis"] = o->axis;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorMultiplySliceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["axis"] = o->axis;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorNormOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorNormFiberOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        attrs["axis"] = o->axis;
        attrs["batch_ndim"] = o->batch_ndim;
        attrs["redux"] = o->redux;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorNormFiberInplaceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        attrs["axis"] = o->axis;
        attrs["batch_ndim"] = o->batch_ndim;
        attrs["redux"] = o->redux;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorNormSliceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        attrs["axis"] = o->axis;
        attrs["redux"] = o->redux;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorNormSliceInplaceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        attrs["axis"] = o->axis;
        attrs["redux"] = o->redux;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorPowOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["exp"] = o->exp;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorRandnOp const *>(&op))
    {
        attrs["start"] = o->start;
        attrs["underlying_shape"] = o->underlying_shape;
        attrs["seed"] = o->seed;
        attrs["mean"] = o->mean;
        attrs["stddev"] = o->stddev;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorScaleOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorScaleInplaceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorScaleFiberOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["axis"] = o->axis;
        attrs["batch_ndim"] = o->batch_ndim;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorScaleSliceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["axis"] = o->axis;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorSoftmaxOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["axis"] = o->axis;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorSoftmaxInplaceOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["axis"] = o->axis;
        return attrs;
    }
    if (auto const *o =
            dynamic_cast<TensorSubtractIndexedOutputsOp const *>(&op))
    {
        attrs["val"] = o->val;
        attrs["ignore_index"] = o->ignore_index;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorSumOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorSumFiberOp const *>(&op))
    {
        attrs["axis"] = o->axis;
        attrs["batch_ndim"] = o->batch_ndim;
        attrs["redux"] = o->redux;
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorSumSliceOp const *>(&op))
    {
        attrs["axis"] = o->axis;
        attrs["redux"] = o->redux;
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorSumprodFiberOp const *>(&op))
    {
        attrs["axis"] = o->axis;
        attrs["redux"] = o->redux;
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorSumprodSliceOp const *>(&op))
    {
        attrs["axis"] = o->axis;
        attrs["redux"] = o->redux;
        attrs["alpha"] = o->alpha;
        attrs["beta"] = o->beta;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorSwapTwoAxesOp const *>(&op))
    {
        attrs["dim0"] = o->dim0;
        attrs["dim1"] = o->dim1;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorTotalSumAccumOp const *>(&op))
    {
        attrs["alpha"] = o->alpha;
        attrs["ignore_index"] = o->ignore_index;
        return attrs;
    }
    if (auto const *o = dynamic_cast<TensorTransposeOp const *>(&op))
    {
        attrs["ndim"] = o->ndim;
        attrs["alpha"] = o->alpha;
        return attrs;
    }
    return attrs;
}

nlohmann::json encode_op(TensorGraph::OpNode const &op)
{
    std::string const wire = op.op_name();
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
