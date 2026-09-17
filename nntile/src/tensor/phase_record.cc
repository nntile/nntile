/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor/phase_record.cc
 * Reconstruct TensorGraph ops from Flush PhaseIR JSON.
 *
 * @version 1.1.0
 * */

#include <nntile/tensor/phase_codec.hh>

#include <nntile/defs.h>
#include <nntile/tensor.hh>
#ifdef NNTILE_TORCH_NATIVE_OPS
#include <nntile/tensor/ops/torch_dispatch.hh>
#endif

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nntile::tensor
{

namespace
{

void throw_unknown_op(std::string const &name)
{
    throw std::runtime_error("UnknownOp:" + name);
}

void throw_missing_node(std::int64_t id)
{
    throw std::runtime_error(
        "UnknownOp: missing node " + std::to_string(id));
}

DataType dtype_from_phase_name(std::string const &name)
{
    if (name == "float32" || name == "FP32" || name == "fp32")
    {
        return DataType::FP32;
    }
    if (name == "float64" || name == "FP64" || name == "fp64")
    {
        return DataType::FP64;
    }
    if (name == "float16" || name == "FP16" || name == "fp16")
    {
        return DataType::FP16;
    }
    if (name == "bfloat16" || name == "BF16" || name == "bf16")
    {
        return DataType::BF16;
    }
    if (name == "int64" || name == "INT64")
    {
        return DataType::INT64;
    }
    if (name == "bool" || name == "BOOL")
    {
        return DataType::BOOL;
    }
    throw std::runtime_error("UnknownOp: dtype " + name);
}

nlohmann::json const &attrs_of(nlohmann::json const &op)
{
    static nlohmann::json const empty = nlohmann::json::object();
    if (op.is_object() && op.contains("attrs") && op["attrs"].is_object())
    {
        return op["attrs"];
    }
    return empty;
}

std::vector<std::int64_t> id_list(
    nlohmann::json const &op,
    char const *key)
{
    std::vector<std::int64_t> out;
    if (!op.contains(key) || !op[key].is_array())
    {
        return out;
    }
    for (auto const &item : op[key])
    {
        out.push_back(item.get<std::int64_t>());
    }
    return out;
}

Scalar attr_scalar(
    nlohmann::json const &attrs,
    char const *key,
    Scalar def)
{
    if (!attrs.contains(key))
    {
        return def;
    }
    return attrs[key].get<Scalar>();
}

Index attr_index(
    nlohmann::json const &attrs,
    char const *key,
    Index def)
{
    if (!attrs.contains(key))
    {
        return def;
    }
    return attrs[key].get<Index>();
}

int attr_int(
    nlohmann::json const &attrs,
    char const *key,
    int def)
{
    if (!attrs.contains(key))
    {
        return def;
    }
    return attrs[key].get<int>();
}

bool attr_bool(
    nlohmann::json const &attrs,
    char const *key,
    bool def)
{
    if (!attrs.contains(key))
    {
        return def;
    }
    return attrs[key].get<bool>();
}

unsigned long long attr_ull(
    nlohmann::json const &attrs,
    char const *key,
    unsigned long long def)
{
    if (!attrs.contains(key))
    {
        return def;
    }
    return attrs[key].get<unsigned long long>();
}

std::string attr_string(
    nlohmann::json const &attrs,
    char const *key,
    std::string const &def)
{
    if (!attrs.contains(key))
    {
        return def;
    }
    return attrs[key].get<std::string>();
}

std::vector<Index> attr_index_vec(
    nlohmann::json const &attrs,
    char const *key)
{
    std::vector<Index> out;
    if (!attrs.contains(key) || !attrs[key].is_array())
    {
        return out;
    }
    for (auto const &item : attrs[key])
    {
        out.push_back(item.get<Index>());
    }
    return out;
}

std::array<Index, 2> attr_index2(
    nlohmann::json const &attrs,
    char const *key,
    std::array<Index, 2> def)
{
    if (!attrs.contains(key) || !attrs[key].is_array() ||
        attrs[key].size() < 2)
    {
        return def;
    }
    return {
        attrs[key].at(0).get<Index>(),
        attrs[key].at(1).get<Index>(),
    };
}

struct Recorder
{
    TensorGraph &graph;
    PhaseNodeMap &refs;
    PhaseNodeSpecs const &specs;
    std::vector<std::int64_t> inputs;
    std::vector<std::int64_t> outputs;
    nlohmann::json const &attrs;

    TensorRef need(std::int64_t id) const
    {
        auto it = refs.find(id);
        if (it == refs.end() || !it->second)
        {
            throw_missing_node(id);
        }
        return it->second;
    }

    TensorRef in_at(size_t i) const
    {
        if (i >= inputs.size())
        {
            throw std::runtime_error("UnknownOp: missing input");
        }
        return need(inputs[i]);
    }

    bool has_out(size_t i) const
    {
        return i < outputs.size() && refs.count(outputs[i]) != 0;
    }

    std::int64_t out_id(size_t i) const
    {
        if (i >= outputs.size())
        {
            throw std::runtime_error("UnknownOp: missing output");
        }
        return outputs[i];
    }

    void set_out(size_t i, TensorGraph::TensorNode *node)
    {
        refs[out_id(i)] = TensorRef::adopt(node);
    }

    TensorRef ensure(std::int64_t id)
    {
        return ensure_phase_node(graph, refs, specs, id);
    }

    TensorRef ensure_out(size_t i)
    {
        return ensure(out_id(i));
    }

    std::vector<Index> out_shape(size_t i) const
    {
        auto spec = specs.find(out_id(i));
        if (spec != specs.end() && !spec->second.shape.empty())
        {
            return spec->second.shape;
        }
        throw_missing_node(out_id(i));
        return {};
    }
};

} // namespace

void parse_phase_nodes(
    nlohmann::json const &nodes,
    PhaseNodeSpecs &specs)
{
    if (!nodes.is_array())
    {
        return;
    }
    for (auto const &node : nodes)
    {
        if (!node.is_object() || !node.contains("id"))
        {
            continue;
        }
        std::int64_t const id = node.at("id").get<std::int64_t>();
        PhaseNodeSpec spec;
        if (node.contains("shape") && node["shape"].is_array())
        {
            for (auto const &dim : node["shape"])
            {
                spec.shape.push_back(dim.get<Index>());
            }
        }
        spec.dtype = dtype_from_phase_name(
            node.value("dtype", std::string("float32")));
        spec.name = node.value("name", std::string());
        specs[id] = std::move(spec);
    }
}

TensorRef ensure_phase_node(
    TensorGraph &graph,
    PhaseNodeMap &refs,
    PhaseNodeSpecs const &specs,
    std::int64_t id)
{
    auto it = refs.find(id);
    if (it != refs.end() && it->second)
    {
        return it->second;
    }
    auto spec_it = specs.find(id);
    if (spec_it == specs.end())
    {
        throw_missing_node(id);
    }
    PhaseNodeSpec const &spec = spec_it->second;
    TensorRef t = graph.data(spec.shape, spec.dtype);
    if (!spec.name.empty())
    {
        t->set_name(spec.name);
    }
    refs.emplace(id, t);
    return t;
}

void record_phase_op(
    TensorGraph &graph,
    nlohmann::json const &op,
    PhaseNodeMap &refs,
    PhaseNodeSpecs const &specs)
{
    if (!op.is_object())
    {
        throw_unknown_op("");
    }
    std::string const name = op.value("op_name", std::string());
    if (!is_v1_phase_op(name))
    {
        throw_unknown_op(name);
    }
    Recorder rec{
        graph,
        refs,
        specs,
        id_list(op, "inputs"),
        id_list(op, "outputs"),
        attrs_of(op),
    };
    nlohmann::json const &a = rec.attrs;

    if (name == "UNREGISTER")
    {
        unregister(rec.in_at(0));
        return;
    }
    if (name == "INVALIDATE")
    {
        invalidate(rec.in_at(0));
        return;
    }
    if (name == "GATHER")
    {
        if (rec.has_out(0))
        {
            gather(rec.in_at(0), rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(0, gather(rec.in_at(0)));
        }
        return;
    }
    if (name == "SCATTER")
    {
        scatter(rec.in_at(0), rec.ensure_out(0));
        return;
    }
    if (name == "COPY")
    {
        if (rec.has_out(0))
        {
            copy(rec.in_at(0), rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(0, copy(rec.in_at(0)));
        }
        return;
    }
    if (name == "COPY_INTERSECTION")
    {
        copy_intersection(
            rec.in_at(0),
            attr_index_vec(a, "src_offset"),
            rec.ensure_out(0),
            attr_index_vec(a, "dst_offset"));
        return;
    }
    if (name == "CONTIGUOUS_VIEW")
    {
        contiguous_view(rec.in_at(0), rec.ensure_out(0));
        return;
    }
    if (name == "CLEAR")
    {
        clear(rec.ensure_out(0));
        return;
    }
    if (name == "FILL")
    {
        fill(attr_scalar(a, "val", 0), rec.ensure_out(0));
        return;
    }
    if (name == "RELU")
    {
        if (rec.has_out(0))
        {
            relu(rec.in_at(0), rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(0, relu(rec.in_at(0)));
        }
        return;
    }
    if (name == "RELU_INPLACE")
    {
        relu_inplace(rec.ensure(rec.inputs.empty()
            ? rec.out_id(0)
            : rec.inputs[0]));
        return;
    }
    if (name == "GELU")
    {
        if (rec.has_out(0))
        {
            gelu(rec.in_at(0), rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(0, gelu(rec.in_at(0)));
        }
        return;
    }
    if (name == "GELU_INPLACE")
    {
        gelu_inplace(rec.ensure(
            rec.inputs.empty() ? rec.out_id(0) : rec.inputs[0]));
        return;
    }
    if (name == "GELUTANH")
    {
        if (rec.has_out(0))
        {
            gelutanh(rec.in_at(0), rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(0, gelutanh(rec.in_at(0)));
        }
        return;
    }
    if (name == "GELUTANH_INPLACE")
    {
        gelutanh_inplace(rec.ensure(
            rec.inputs.empty() ? rec.out_id(0) : rec.inputs[0]));
        return;
    }
    if (name == "SILU")
    {
        if (rec.has_out(0))
        {
            silu(rec.in_at(0), rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(0, silu(rec.in_at(0)));
        }
        return;
    }
    if (name == "SILU_INPLACE")
    {
        silu_inplace(rec.ensure(
            rec.inputs.empty() ? rec.out_id(0) : rec.inputs[0]));
        return;
    }
    if (name == "SQRT")
    {
        if (rec.has_out(0))
        {
            sqrt(rec.in_at(0), rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(0, sqrt(rec.in_at(0)));
        }
        return;
    }
    if (name == "SQRT_INPLACE")
    {
        sqrt_inplace(rec.ensure(
            rec.inputs.empty() ? rec.out_id(0) : rec.inputs[0]));
        return;
    }
    if (name == "ADD")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Scalar const beta = attr_scalar(a, "beta", 1);
        if (rec.has_out(0))
        {
            add(alpha, rec.in_at(0), beta, rec.in_at(1),
                rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(0, add(alpha, rec.in_at(0), beta, rec.in_at(1)));
        }
        return;
    }
    if (name == "ADD_INPLACE")
    {
        add_inplace(
            attr_scalar(a, "alpha", 1),
            rec.in_at(0),
            attr_scalar(a, "beta", 1),
            rec.in_at(1));
        return;
    }
    if (name == "MULTIPLY")
    {
        rec.set_out(
            0,
            multiply(
                rec.in_at(0),
                rec.in_at(1),
                attr_scalar(a, "alpha", 1)));
        return;
    }
    if (name == "MULTIPLY_INPLACE")
    {
        multiply_inplace(
            attr_scalar(a, "alpha", 1),
            rec.in_at(0),
            rec.in_at(1));
        return;
    }
    if (name == "GEMM")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Scalar const beta = attr_scalar(a, "beta", 0);
        bool const trans_a = attr_bool(a, "trans_a", false);
        bool const trans_b = attr_bool(a, "trans_b", false);
        Index const ndim = attr_index(a, "ndim", 1);
        Index const batch_ndim = attr_index(a, "batch_ndim", 0);
        if (rec.has_out(0))
        {
            gemm(
                rec.in_at(0),
                rec.in_at(1),
                rec.need(rec.outputs[0]),
                alpha,
                beta,
                trans_a,
                trans_b,
                ndim,
                batch_ndim);
        }
        else
        {
            rec.set_out(
                0,
                gemm(
                    rec.in_at(0),
                    rec.in_at(1),
                    alpha,
                    trans_a,
                    trans_b,
                    ndim,
                    batch_ndim));
        }
        return;
    }
    if (name == "ADD_SLICE")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Scalar const beta = attr_scalar(a, "beta", 1);
        Index const axis = attr_index(a, "axis", 0);
        if (rec.has_out(0))
        {
            add_slice(
                alpha,
                rec.in_at(0),
                beta,
                rec.in_at(1),
                rec.need(rec.outputs[0]),
                axis);
        }
        else
        {
            rec.set_out(
                0,
                add_slice(
                    alpha, rec.in_at(0), beta, rec.in_at(1), axis));
        }
        return;
    }
    if (name == "ADD_SLICE_INPLACE")
    {
        add_slice_inplace(
            attr_scalar(a, "alpha", 1),
            rec.in_at(0),
            attr_scalar(a, "beta", 1),
            rec.in_at(1),
            attr_index(a, "axis", 0));
        return;
    }
    if (name == "ADD_FIBER")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Scalar const beta = attr_scalar(a, "beta", 1);
        Index const axis = attr_index(a, "axis", 0);
        Index const batch_ndim = attr_index(a, "batch_ndim", 0);
        if (rec.has_out(0))
        {
            add_fiber(
                alpha,
                rec.in_at(0),
                beta,
                rec.in_at(1),
                rec.need(rec.outputs[0]),
                axis,
                batch_ndim);
        }
        else
        {
            rec.set_out(
                0,
                add_fiber(
                    alpha,
                    rec.in_at(0),
                    beta,
                    rec.in_at(1),
                    axis,
                    batch_ndim));
        }
        return;
    }
    if (name == "ADD_FIBER_INPLACE")
    {
        add_fiber_inplace(
            attr_scalar(a, "alpha", 1),
            rec.in_at(0),
            attr_scalar(a, "beta", 1),
            rec.in_at(1),
            attr_index(a, "axis", 0),
            attr_index(a, "batch_ndim", 0));
        return;
    }
    if (name == "SCALE")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        if (rec.has_out(0))
        {
            scale(alpha, rec.in_at(0), rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(0, scale(alpha, rec.in_at(0)));
        }
        return;
    }
    if (name == "SCALE_INPLACE")
    {
        scale_inplace(
            attr_scalar(a, "alpha", 1),
            rec.ensure(
                rec.inputs.empty() ? rec.out_id(0) : rec.inputs[0]));
        return;
    }
    if (name == "SCALE_SLICE")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Index const axis = attr_index(a, "axis", 0);
        if (rec.has_out(0))
        {
            scale_slice(
                alpha, rec.in_at(0), rec.need(rec.outputs[0]), axis);
        }
        else
        {
            TensorRef const src = rec.in_at(0);
            Index axis_size = 1;
            if (axis >= 0 &&
                static_cast<size_t>(axis) < src->shape().size())
            {
                axis_size = src->shape()[static_cast<size_t>(axis)];
            }
            rec.set_out(0, scale_slice(alpha, src, axis, axis_size));
        }
        return;
    }
    if (name == "SCALE_FIBER")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Index const axis = attr_index(a, "axis", 0);
        Index const batch_ndim = attr_index(a, "batch_ndim", 0);
        if (rec.has_out(0))
        {
            scale_fiber(
                alpha,
                rec.in_at(0),
                rec.need(rec.outputs[0]),
                axis,
                batch_ndim);
        }
        else
        {
            rec.set_out(
                0,
                scale_fiber(
                    alpha,
                    rec.in_at(0),
                    rec.out_shape(0),
                    axis,
                    batch_ndim));
        }
        return;
    }
    if (name == "MULTIPLY_SLICE")
    {
        multiply_slice(
            attr_scalar(a, "alpha", 1),
            rec.in_at(0),
            rec.in_at(1),
            attr_index(a, "axis", 0));
        return;
    }
    if (name == "MULTIPLY_FIBER")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Index const axis = attr_index(a, "axis", 0);
        if (rec.has_out(0))
        {
            multiply_fiber(
                alpha,
                rec.in_at(0),
                rec.in_at(1),
                rec.need(rec.outputs[0]),
                axis);
        }
        else
        {
            rec.set_out(
                0,
                multiply_fiber(
                    alpha, rec.in_at(0), rec.in_at(1), axis));
        }
        return;
    }
    if (name == "MULTIPLY_FIBER_INPLACE")
    {
        multiply_fiber_inplace(
            attr_scalar(a, "alpha", 1),
            rec.in_at(0),
            rec.in_at(1),
            attr_index(a, "axis", 0));
        return;
    }
    if (name == "POW")
    {
        pow(
            attr_scalar(a, "alpha", 1),
            attr_scalar(a, "exp", 1),
            rec.ensure(
                rec.inputs.empty() ? rec.out_id(0) : rec.inputs[0]));
        return;
    }
    if (name == "HYPOT")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Scalar const beta = attr_scalar(a, "beta", 1);
        if (rec.has_out(0))
        {
            hypot(
                alpha,
                rec.in_at(0),
                beta,
                rec.in_at(1),
                rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(
                0,
                hypot(alpha, rec.in_at(0), beta, rec.in_at(1)));
        }
        return;
    }
    if (name == "HYPOT_INPLACE")
    {
        hypot_inplace(
            attr_scalar(a, "alpha", 1),
            rec.in_at(0),
            attr_scalar(a, "beta", 1),
            rec.in_at(1));
        return;
    }
    if (name == "HYPOT_SCALAR_INVERSE")
    {
        hypot_scalar_inverse(
            attr_scalar(a, "eps", 0),
            attr_scalar(a, "alpha", 1),
            rec.ensure(
                rec.inputs.empty() ? rec.out_id(0) : rec.inputs[0]));
        return;
    }
    if (name == "RELU_BACKWARD")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Scalar const beta = attr_scalar(a, "beta", 0);
        if (rec.has_out(0))
        {
            relu_backward(
                alpha,
                rec.in_at(0),
                rec.in_at(1),
                beta,
                rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(
                0,
                relu_backward(alpha, rec.in_at(0), rec.in_at(1)));
        }
        return;
    }
    if (name == "GELU_BACKWARD")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Scalar const beta = attr_scalar(a, "beta", 0);
        if (rec.has_out(0))
        {
            gelu_backward(
                alpha,
                rec.in_at(0),
                rec.in_at(1),
                beta,
                rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(
                0,
                gelu_backward(alpha, rec.in_at(0), rec.in_at(1)));
        }
        return;
    }
    if (name == "GELUTANH_BACKWARD")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Scalar const beta = attr_scalar(a, "beta", 0);
        if (rec.has_out(0))
        {
            gelutanh_backward(
                alpha,
                rec.in_at(0),
                rec.in_at(1),
                beta,
                rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(
                0,
                gelutanh_backward(alpha, rec.in_at(0), rec.in_at(1)));
        }
        return;
    }
    if (name == "SILU_BACKWARD")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Scalar const beta = attr_scalar(a, "beta", 0);
        if (rec.has_out(0))
        {
            silu_backward(
                alpha,
                rec.in_at(0),
                rec.in_at(1),
                beta,
                rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(
                0,
                silu_backward(alpha, rec.in_at(0), rec.in_at(1)));
        }
        return;
    }
    if (name == "SUM")
    {
        sum(
            rec.in_at(0),
            rec.ensure_out(0),
            attr_scalar(a, "alpha", 1),
            attr_scalar(a, "beta", 0));
        return;
    }
    if (name == "SUM_SLICE")
    {
        Index const axis = attr_index(a, "axis", 0);
        int const redux = attr_int(a, "redux", 0);
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Scalar const beta = attr_scalar(a, "beta", 0);
        if (rec.has_out(0))
        {
            sum_slice(
                rec.in_at(0),
                rec.need(rec.outputs[0]),
                axis,
                redux,
                alpha,
                beta);
        }
        else
        {
            rec.set_out(
                0,
                sum_slice(rec.in_at(0), axis, redux, alpha, beta));
        }
        return;
    }
    if (name == "SUM_FIBER")
    {
        Index const axis = attr_index(a, "axis", 0);
        Index const batch_ndim = attr_index(a, "batch_ndim", 0);
        int const redux = attr_int(a, "redux", 0);
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Scalar const beta = attr_scalar(a, "beta", 0);
        if (rec.has_out(0))
        {
            sum_fiber(
                rec.in_at(0),
                rec.need(rec.outputs[0]),
                axis,
                batch_ndim,
                redux,
                alpha,
                beta);
        }
        else
        {
            rec.set_out(
                0,
                sum_fiber(
                    rec.in_at(0),
                    axis,
                    batch_ndim,
                    redux,
                    alpha,
                    beta));
        }
        return;
    }
    if (name == "SUMPROD_FIBER")
    {
        sumprod_fiber(
            rec.in_at(0),
            rec.in_at(1),
            rec.ensure_out(0),
            attr_index(a, "axis", 0),
            attr_int(a, "redux", 0),
            attr_scalar(a, "alpha", 1),
            attr_scalar(a, "beta", 0));
        return;
    }
    if (name == "SUMPROD_SLICE")
    {
        sumprod_slice(
            rec.in_at(0),
            rec.in_at(1),
            rec.ensure_out(0),
            attr_index(a, "axis", 0),
            attr_int(a, "redux", 0),
            attr_scalar(a, "alpha", 1),
            attr_scalar(a, "beta", 0));
        return;
    }
    if (name == "TOTAL_SUM_ACCUM")
    {
        total_sum_accum(
            attr_scalar(a, "alpha", 1),
            rec.in_at(0),
            rec.in_at(1),
            rec.in_at(2),
            rec.ensure(rec.inputs.size() > 3
                ? rec.inputs[3]
                : rec.out_id(0)),
            attr_index(a, "ignore_index", -1));
        return;
    }
    if (name == "NORM")
    {
        norm(
            rec.in_at(0),
            rec.ensure_out(0),
            attr_scalar(a, "alpha", 1),
            attr_scalar(a, "beta", 0));
        return;
    }
    if (name == "NORM_FIBER")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Scalar const beta = attr_scalar(a, "beta", 0);
        Index const axis = attr_index(a, "axis", 0);
        Index const batch_ndim = attr_index(a, "batch_ndim", 0);
        int const redux = attr_int(a, "redux", 0);
        if (rec.has_out(0))
        {
            norm_fiber(
                alpha,
                rec.in_at(0),
                beta,
                rec.in_at(1),
                rec.need(rec.outputs[0]),
                axis,
                batch_ndim,
                redux);
        }
        else
        {
            rec.set_out(
                0,
                norm_fiber(
                    alpha,
                    rec.in_at(0),
                    beta,
                    rec.in_at(1),
                    axis,
                    batch_ndim,
                    redux));
        }
        return;
    }
    if (name == "NORM_FIBER_INPLACE")
    {
        norm_fiber_inplace(
            attr_scalar(a, "alpha", 1),
            rec.in_at(0),
            attr_scalar(a, "beta", 0),
            rec.in_at(1),
            attr_index(a, "axis", 0),
            attr_index(a, "batch_ndim", 0),
            attr_int(a, "redux", 0));
        return;
    }
    if (name == "NORM_SLICE")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Scalar const beta = attr_scalar(a, "beta", 0);
        Index const axis = attr_index(a, "axis", 0);
        int const redux = attr_int(a, "redux", 0);
        if (rec.has_out(0))
        {
            norm_slice(
                alpha,
                rec.in_at(0),
                beta,
                rec.in_at(1),
                rec.need(rec.outputs[0]),
                axis,
                redux);
        }
        else
        {
            rec.set_out(
                0,
                norm_slice(
                    alpha,
                    rec.in_at(0),
                    beta,
                    rec.in_at(1),
                    axis,
                    redux));
        }
        return;
    }
    if (name == "NORM_SLICE_INPLACE")
    {
        norm_slice_inplace(
            attr_scalar(a, "alpha", 1),
            rec.in_at(0),
            attr_scalar(a, "beta", 0),
            rec.in_at(1),
            attr_index(a, "axis", 0),
            attr_int(a, "redux", 0));
        return;
    }
    if (name == "MAXSUMEXP")
    {
        Index const axis = attr_index(a, "axis", 0);
        Scalar const beta = attr_scalar(a, "beta", 0);
        int const redux = attr_int(a, "redux", 0);
        if (rec.has_out(0))
        {
            maxsumexp(
                rec.in_at(0),
                rec.need(rec.outputs[0]),
                axis,
                beta,
                redux);
        }
        else
        {
            rec.set_out(0, maxsumexp(rec.in_at(0), axis, redux));
        }
        return;
    }
    if (name == "LOGSUMEXP")
    {
        if (rec.has_out(0))
        {
            logsumexp(rec.in_at(0), rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(0, logsumexp(rec.in_at(0)));
        }
        return;
    }
    if (name == "SOFTMAX")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Index const axis = attr_index(a, "axis", 0);
        if (rec.has_out(0))
        {
            softmax(
                rec.in_at(0),
                rec.in_at(1),
                rec.need(rec.outputs[0]),
                alpha,
                axis);
        }
        else
        {
            rec.set_out(
                0,
                softmax(rec.in_at(0), rec.in_at(1), alpha, axis));
        }
        return;
    }
    if (name == "SOFTMAX_INPLACE")
    {
        softmax_inplace(
            rec.in_at(0),
            rec.in_at(1),
            attr_scalar(a, "alpha", 1),
            attr_index(a, "axis", 0));
        return;
    }
    if (name == "TRANSPOSE")
    {
        Scalar const alpha = attr_scalar(a, "alpha", 1);
        Index const ndim = attr_index(a, "ndim", 1);
        if (rec.has_out(0))
        {
            transpose(
                alpha, rec.in_at(0), rec.need(rec.outputs[0]), ndim);
        }
        else
        {
            rec.set_out(0, transpose(alpha, rec.in_at(0), ndim));
        }
        return;
    }
    if (name == "SWAP_TWO_AXES")
    {
        swap_two_axes(
            rec.in_at(0),
            rec.ensure_out(0),
            attr_index(a, "dim0", 0),
            attr_index(a, "dim1", 1));
        return;
    }
    if (name == "CONCAT")
    {
        rec.set_out(
            0,
            concat(
                rec.in_at(0),
                rec.in_at(1),
                attr_index(a, "axis", 0)));
        return;
    }
    if (name == "EMBEDDING")
    {
        Index const axis = attr_index(a, "axis", 0);
        if (rec.has_out(0))
        {
            embedding(
                rec.in_at(0),
                rec.in_at(1),
                rec.need(rec.outputs[0]),
                axis);
        }
        else
        {
            rec.set_out(0, embedding(rec.in_at(0), rec.in_at(1), axis));
        }
        return;
    }
    if (name == "EMBEDDING_BACKWARD")
    {
        embedding_backward(
            rec.in_at(0),
            rec.in_at(1),
            rec.ensure(
                rec.inputs.size() > 2
                    ? rec.inputs[2]
                    : rec.out_id(0)),
            attr_index(a, "axis", 0),
            attr_scalar(a, "alpha", 1),
            attr_scalar(a, "beta", 0),
            attr_int(a, "redux", 0));
        return;
    }
    if (name == "CONV2D_INPLACE")
    {
        conv2d_inplace(
            attr_scalar(a, "alpha", 1),
            rec.in_at(0),
            rec.in_at(1),
            attr_scalar(a, "beta", 0),
            rec.ensure(
                rec.inputs.size() > 2
                    ? rec.inputs[2]
                    : rec.out_id(0)),
            attr_index2(a, "padding", {0, 0}),
            attr_index2(a, "stride", {1, 1}),
            attr_index2(a, "dilation", {1, 1}));
        return;
    }
    if (name == "CONV2D_BWD_INPUT_INPLACE")
    {
        conv2d_bwd_input_inplace(
            attr_scalar(a, "alpha", 1),
            rec.in_at(0),
            rec.in_at(1),
            attr_scalar(a, "beta", 0),
            rec.ensure(
                rec.inputs.size() > 2
                    ? rec.inputs[2]
                    : rec.out_id(0)),
            attr_index2(a, "padding", {0, 0}),
            attr_index2(a, "stride", {1, 1}),
            attr_index2(a, "dilation", {1, 1}));
        return;
    }
    if (name == "CONV2D_BWD_WEIGHT_INPLACE")
    {
        conv2d_bwd_weight_inplace(
            attr_scalar(a, "alpha", 1),
            rec.in_at(0),
            rec.in_at(1),
            attr_scalar(a, "beta", 0),
            rec.ensure(
                rec.inputs.size() > 2
                    ? rec.inputs[2]
                    : rec.out_id(0)),
            attr_index2(a, "padding", {0, 0}),
            attr_index2(a, "stride", {1, 1}),
            attr_index2(a, "dilation", {1, 1}));
        return;
    }
    if (name == "ROPE")
    {
        if (rec.has_out(0))
        {
            rope(
                rec.in_at(0),
                rec.in_at(1),
                rec.in_at(2),
                rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(
                0,
                rope(rec.in_at(0), rec.in_at(1), rec.in_at(2)));
        }
        return;
    }
    if (name == "ROPE_BACKWARD")
    {
        if (rec.has_out(0))
        {
            rope_backward(
                rec.in_at(0),
                rec.in_at(1),
                rec.in_at(2),
                rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(
                0,
                rope_backward(
                    rec.in_at(0), rec.in_at(1), rec.in_at(2)));
        }
        return;
    }
    if (name == "MASK_SCALAR")
    {
        mask_scalar(
            rec.in_at(0),
            attr_scalar(a, "val", 0),
            rec.ensure(
                rec.inputs.size() > 1
                    ? rec.inputs[1]
                    : rec.out_id(0)),
            attr_index(a, "batch_ndim", 0));
        return;
    }
    if (name == "SUBTRACT_INDEXED_OUTPUTS")
    {
        subtract_indexed_outputs(
            attr_scalar(a, "val", 1),
            rec.in_at(0),
            rec.ensure(
                rec.inputs.size() > 1
                    ? rec.inputs[1]
                    : rec.out_id(0)),
            attr_index(a, "ignore_index", -1));
        return;
    }
    if (name == "ADAM_STEP")
    {
        adam_step(
            attr_index(a, "num_iter", 1),
            attr_scalar(a, "beta_1", 0.9f),
            attr_scalar(a, "beta_2", 0.999f),
            attr_scalar(a, "eps", 1.0e-8f),
            attr_scalar(a, "lr", 1.0e-3f),
            attr_scalar(a, "weight_decay", 0),
            rec.in_at(0),
            rec.in_at(1),
            rec.in_at(2),
            rec.in_at(3));
        return;
    }
    if (name == "ADAMW_STEP")
    {
        adamw_step(
            attr_index(a, "num_iter", 1),
            attr_scalar(a, "beta_1", 0.9f),
            attr_scalar(a, "beta_2", 0.999f),
            attr_scalar(a, "eps", 1.0e-8f),
            attr_scalar(a, "lr", 1.0e-3f),
            attr_scalar(a, "weight_decay", 0),
            rec.in_at(0),
            rec.in_at(1),
            rec.in_at(2),
            rec.in_at(3));
        return;
    }
    if (name == "SGD_STEP")
    {
        sgd_step(
            attr_index(a, "num_iter", 1),
            attr_scalar(a, "momentum", 0),
            attr_scalar(a, "lr", 1.0e-3f),
            attr_scalar(a, "weight_decay", 0),
            attr_scalar(a, "dampening", 0),
            attr_bool(a, "nesterov", false),
            rec.in_at(0),
            rec.in_at(1),
            rec.in_at(2));
        return;
    }
    if (name == "RANDN")
    {
        randn(
            rec.ensure(
                rec.inputs.empty() ? rec.out_id(0) : rec.inputs[0]),
            attr_index_vec(a, "start"),
            attr_index_vec(a, "underlying_shape"),
            attr_ull(a, "seed", 0),
            attr_scalar(a, "mean", 0),
            attr_scalar(a, "stddev", 1));
        return;
    }
    if (name == "LOG_SCALAR")
    {
        log_scalar(attr_string(a, "name", ""), rec.in_at(0));
        return;
    }
#ifdef NNTILE_USE_FLASH_SDPA
    if (name == "FLASH_SDPA_FWD_CUDNN")
    {
        flash_sdpa_fwd_cudnn(
            rec.in_at(0),
            rec.in_at(1),
            rec.in_at(2),
            rec.ensure(rec.inputs.size() > 3
                ? rec.inputs[3]
                : rec.out_id(0)),
            rec.in_at(rec.inputs.size() > 4 ? 4 : 3),
            rec.ensure_out(rec.outputs.size() > 1 ? 1 : 0));
        return;
    }
    if (name == "FLASH_SDPA_BWD_CUDNN")
    {
        flash_sdpa_bwd_cudnn(
            rec.in_at(0),
            rec.in_at(1),
            rec.in_at(2),
            rec.in_at(3),
            rec.in_at(4),
            rec.in_at(5),
            rec.in_at(6),
            rec.ensure(rec.inputs.size() > 7
                ? rec.inputs[7]
                : rec.out_id(0)),
            rec.ensure(rec.inputs.size() > 8
                ? rec.inputs[8]
                : rec.out_id(1)),
            rec.ensure(rec.inputs.size() > 9
                ? rec.inputs[9]
                : rec.out_id(2)));
        return;
    }
#else
    if (name == "FLASH_SDPA_FWD_CUDNN" ||
        name == "FLASH_SDPA_BWD_CUDNN")
    {
        throw_unknown_op(name);
    }
#endif
#ifdef NNTILE_TORCH_NATIVE_OPS
    if (name == "TORCH_UNARY" || name == "TORCH_BINARY" ||
        name == "TORCH_TERNARY")
    {
        using nntile::starpu::TorchKind;
        auto const kind = static_cast<TorchKind>(attr_int(a, "kind", 0));
        if (name == "TORCH_UNARY")
        {
            std::vector<Index> shape = rec.in_at(0)->shape();
            auto spec = specs.find(rec.out_id(0));
            if (spec != specs.end() && !spec->second.shape.empty())
            {
                shape = spec->second.shape;
            }
            if (rec.has_out(0))
            {
                torch_unary(kind, rec.in_at(0), rec.need(rec.outputs[0]));
            }
            else
            {
                rec.set_out(0, torch_unary(kind, rec.in_at(0), shape));
            }
            return;
        }
        if (name == "TORCH_BINARY")
        {
            std::vector<Index> shape = rec.in_at(0)->shape();
            auto spec = specs.find(rec.out_id(0));
            if (spec != specs.end() && !spec->second.shape.empty())
            {
                shape = spec->second.shape;
            }
            if (rec.has_out(0))
            {
                torch_binary(
                    kind,
                    rec.in_at(0),
                    rec.in_at(1),
                    rec.need(rec.outputs[0]));
            }
            else
            {
                rec.set_out(
                    0,
                    torch_binary(
                        kind, rec.in_at(0), rec.in_at(1), shape));
            }
            return;
        }
        std::vector<Index> shape = rec.in_at(0)->shape();
        auto spec = specs.find(rec.out_id(0));
        if (spec != specs.end() && !spec->second.shape.empty())
        {
            shape = spec->second.shape;
        }
        if (rec.has_out(0))
        {
            torch_ternary(
                kind,
                rec.in_at(0),
                rec.in_at(1),
                rec.in_at(2),
                rec.need(rec.outputs[0]));
        }
        else
        {
            rec.set_out(
                0,
                torch_ternary(
                    kind,
                    rec.in_at(0),
                    rec.in_at(1),
                    rec.in_at(2),
                    shape));
        }
        return;
    }
#else
    if (name == "TORCH_UNARY" || name == "TORCH_BINARY" ||
        name == "TORCH_TERNARY")
    {
        throw_unknown_op(name);
    }
#endif
    throw_unknown_op(name);
}

void apply_phase(
    TensorGraph &graph,
    nlohmann::json const &blob,
    PhaseNodeMap &refs)
{
    auto decoded = decode_phase(blob);
    PhaseNodeSpecs specs;
    parse_phase_nodes(decoded.first, specs);
    for (auto const &op : decoded.second)
    {
        auto const inputs = id_list(op, "inputs");
        for (std::int64_t id : inputs)
        {
            if (!refs.count(id))
            {
                ensure_phase_node(graph, refs, specs, id);
            }
        }
        record_phase_op(graph, op, refs, specs);
    }
}

} // namespace nntile::tensor
