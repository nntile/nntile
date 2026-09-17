/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/remote_tile_codec.cc
 * Encode / apply classic TILE_* ops on the Unix-socket driver.
 *
 * @version 1.1.0
 * */

#include <nntile/remote_tile_codec.hh>

#include <nntile/defs.h>
#include <nntile/tile/graph_ops.hh>
#include <nntile/tile/ops/swap_two_axes.hh>
#ifdef NNTILE_TORCH_NATIVE_OPS
#include <nntile/tile/ops/torch_dispatch.hh>
#endif

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nntile
{

namespace
{

template <typename Op>
Op const &cast_op(TileGraph::OpNode const &op, char const *name)
{
    auto const *p = dynamic_cast<Op const *>(&op);
    if (p == nullptr)
    {
        throw std::runtime_error(std::string("UnknownOp: ") + name);
    }
    return *p;
}

void require_min(
    nlohmann::json const &arr, size_t n, char const *what)
{
    if (arr.size() < n)
    {
        throw std::runtime_error(std::string("UnknownOp: ") + what);
    }
}

Scalar attr_s(
    nlohmann::json const &attrs, char const *key, Scalar def)
{
    return attrs.value(key, def);
}

Index attr_i(
    nlohmann::json const &attrs, char const *key, Index def)
{
    return static_cast<Index>(attrs.value(key, def));
}

int attr_n(
    nlohmann::json const &attrs, char const *key, int def)
{
    return attrs.value(key, def);
}

bool attr_b(
    nlohmann::json const &attrs, char const *key, bool def)
{
    return attrs.value(key, def);
}

TileGraph::TileNode *arg(
    TileGraph &graph,
    nlohmann::json const &ids,
    size_t i,
    char const *what)
{
    require_min(ids, i + 1, what);
    return tile_node_by_id(
        graph, ids.at(i).get<TileGraph::NodeId>());
}

#ifdef NNTILE_TORCH_NATIVE_OPS
nlohmann::json encode_torch_extra(
    starpu::TorchKind kind,
    starpu::TorchDispatchArgs const &extra)
{
    nlohmann::json attrs = nlohmann::json::object();
    attrs["kind"] = static_cast<std::int32_t>(kind);
    attrs["n_in"] = extra.n_in;
    attrs["n_out"] = extra.n_out;
    nlohmann::json scalars = nlohmann::json::array();
    for (int i = 0; i < 4; ++i)
    {
        scalars.push_back(extra.scalars[i]);
    }
    attrs["scalars"] = std::move(scalars);
    nlohmann::json iargs = nlohmann::json::array();
    for (int i = 0; i < 16; ++i)
    {
        iargs.push_back(extra.iargs[i]);
    }
    attrs["iargs"] = std::move(iargs);
    return attrs;
}

starpu::TorchDispatchArgs decode_torch_extra(
    nlohmann::json const &attrs)
{
    starpu::TorchDispatchArgs extra{};
    extra.kind = static_cast<starpu::TorchKind>(
        attrs.value("kind", 0));
    extra.n_in = static_cast<Index>(attrs.value("n_in", 0));
    extra.n_out = static_cast<Index>(attrs.value("n_out", 1));
    if (attrs.contains("scalars") && attrs["scalars"].is_array())
    {
        auto const &s = attrs["scalars"];
        for (size_t i = 0; i < s.size() && i < 4; ++i)
        {
            extra.scalars[i] = s.at(i).get<Scalar>();
        }
    }
    if (attrs.contains("iargs") && attrs["iargs"].is_array())
    {
        auto const &a = attrs["iargs"];
        for (size_t i = 0; i < a.size() && i < 16; ++i)
        {
            extra.iargs[i] = a.at(i).get<Index>();
        }
    }
    return extra;
}
#endif

} // namespace


TileGraph::TileNode *tile_node_by_id(
    TileGraph &graph, TileGraph::NodeId id)
{
    for (auto const &node : graph.tile_nodes())
    {
        if (node && node->id() == id)
        {
            return node.get();
        }
    }
    throw std::runtime_error("unknown tile node id");
}

bool tile_graph_has_node(
    TileGraph const &graph, TileGraph::NodeId id)
{
    for (auto const &node : graph.tile_nodes())
    {
        if (node && node->id() == id)
        {
            return true;
        }
    }
    return false;
}

nlohmann::json encode_tile_op_attrs(
    TileGraph::OpNode const &op)
{
    nlohmann::json attrs = nlohmann::json::object();
    std::string const name = op.op_name();
    static char const *const k_no_attr[] = {
        "TILE_RELU",
        "TILE_RELU_INPLACE",
        "TILE_GELU",
        "TILE_GELU_INPLACE",
        "TILE_GELUTANH",
        "TILE_GELUTANH_INPLACE",
        "TILE_SILU",
        "TILE_SILU_INPLACE",
        "TILE_SQRT",
        "TILE_SQRT_INPLACE",
        "TILE_COPY",
        "TILE_COPY_SAME_NUMEL",
        "TILE_CLEAR",
        "TILE_UNREGISTER",
        "TILE_INVALIDATE",
        "TILE_LOGSUMEXP",
    };
    for (char const *n : k_no_attr)
    {
        if (name == n)
        {
            return attrs;
        }
    }
    if (name == "TILE_ADD_INPLACE")
    {
        auto const &node = cast_op<tile::TileAddInplaceOp>(
            op, "TILE_ADD_INPLACE");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        return attrs;
    }
    if (name == "TILE_ADD")
    {
        auto const &node = cast_op<tile::TileAddOp>(
            op, "TILE_ADD");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        return attrs;
    }
    if (name == "TILE_MULTIPLY")
    {
        auto const &node = cast_op<tile::TileMultiplyOp>(
            op, "TILE_MULTIPLY");
        attrs["alpha"] = node.alpha;
        return attrs;
    }
    if (name == "TILE_MULTIPLY_INPLACE")
    {
        auto const &node = cast_op<tile::TileMultiplyInplaceOp>(
            op, "TILE_MULTIPLY_INPLACE");
        attrs["alpha"] = node.alpha;
        return attrs;
    }
    if (name == "TILE_GEMM")
    {
        auto const &node = cast_op<tile::TileGemmOp>(
            op, "TILE_GEMM");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        attrs["trans_a"] = node.trans_a;
        attrs["trans_b"] = node.trans_b;
        attrs["ndim"] = node.ndim;
        attrs["batch_ndim"] = node.batch_ndim;
        return attrs;
    }
    if (name == "TILE_ADD_SLICE")
    {
        auto const &node = cast_op<tile::TileAddSliceOp>(
            op, "TILE_ADD_SLICE");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        attrs["axis"] = node.axis;
        return attrs;
    }
    if (name == "TILE_ADD_SLICE_INPLACE")
    {
        auto const &node = cast_op<tile::TileAddSliceInplaceOp>(
            op, "TILE_ADD_SLICE_INPLACE");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        attrs["axis"] = node.axis;
        return attrs;
    }
    if (name == "TILE_ADD_FIBER")
    {
        auto const &node = cast_op<tile::TileAddFiberOp>(
            op, "TILE_ADD_FIBER");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        attrs["axis"] = node.axis;
        attrs["batch_ndim"] = node.batch_ndim;
        return attrs;
    }
    if (name == "TILE_ADD_FIBER_INPLACE")
    {
        auto const &node = cast_op<tile::TileAddFiberInplaceOp>(
            op, "TILE_ADD_FIBER_INPLACE");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        attrs["axis"] = node.axis;
        attrs["batch_ndim"] = node.batch_ndim;
        return attrs;
    }
    if (name == "TILE_FILL")
    {
        auto const &node = cast_op<tile::TileFillOp>(
            op, "TILE_FILL");
        attrs["value"] = node.val;
        return attrs;
    }
    if (name == "TILE_COPY_INTERSECTION")
    {
        auto const &node = cast_op<tile::TileCopyIntersectionOp>(
            op, "TILE_COPY_INTERSECTION");
        attrs["src_offset"] = node.src_offset;
        attrs["dst_offset"] = node.dst_offset;
        return attrs;
    }
    if (name == "TILE_SCALE")
    {
        auto const &node = cast_op<tile::TileScaleOp>(
            op, "TILE_SCALE");
        attrs["alpha"] = node.alpha;
        return attrs;
    }
    if (name == "TILE_SCALE_INPLACE")
    {
        auto const &node = cast_op<tile::TileScaleInplaceOp>(
            op, "TILE_SCALE_INPLACE");
        attrs["alpha"] = node.alpha;
        return attrs;
    }
    if (name == "TILE_SCALE_SLICE")
    {
        auto const &node = cast_op<tile::TileScaleSliceOp>(
            op, "TILE_SCALE_SLICE");
        attrs["alpha"] = node.alpha;
        attrs["axis"] = node.axis;
        return attrs;
    }
    if (name == "TILE_SCALE_FIBER")
    {
        auto const &node = cast_op<tile::TileScaleFiberOp>(
            op, "TILE_SCALE_FIBER");
        attrs["alpha"] = node.alpha;
        attrs["axis"] = node.axis;
        attrs["batch_ndim"] = node.batch_ndim;
        return attrs;
    }
    if (name == "TILE_MULTIPLY_SLICE")
    {
        auto const &node = cast_op<tile::TileMultiplySliceOp>(
            op, "TILE_MULTIPLY_SLICE");
        attrs["alpha"] = node.alpha;
        attrs["axis"] = node.axis;
        return attrs;
    }
    if (name == "TILE_MULTIPLY_FIBER")
    {
        auto const &node = cast_op<tile::TileMultiplyFiberOp>(
            op, "TILE_MULTIPLY_FIBER");
        attrs["alpha"] = node.alpha;
        attrs["axis"] = node.axis;
        return attrs;
    }
    if (name == "TILE_MULTIPLY_FIBER_INPLACE")
    {
        auto const &node = cast_op<tile::TileMultiplyFiberInplaceOp>(
            op, "TILE_MULTIPLY_FIBER_INPLACE");
        attrs["alpha"] = node.alpha;
        attrs["axis"] = node.axis;
        return attrs;
    }
    if (name == "TILE_POW")
    {
        auto const &node = cast_op<tile::TilePowOp>(
            op, "TILE_POW");
        attrs["alpha"] = node.alpha;
        attrs["exp"] = node.exp;
        return attrs;
    }
    if (name == "TILE_HYPOT")
    {
        auto const &node = cast_op<tile::TileHypotOp>(
            op, "TILE_HYPOT");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        return attrs;
    }
    if (name == "TILE_HYPOT_INPLACE")
    {
        auto const &node = cast_op<tile::TileHypotInplaceOp>(
            op, "TILE_HYPOT_INPLACE");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        return attrs;
    }
    if (name == "TILE_HYPOT_SCALAR_INVERSE")
    {
        auto const &node = cast_op<tile::TileHypotScalarInverseOp>(
            op, "TILE_HYPOT_SCALAR_INVERSE");
        attrs["eps"] = node.eps;
        attrs["alpha"] = node.alpha;
        return attrs;
    }
    if (name == "TILE_RELU_BACKWARD")
    {
        auto const &node = cast_op<tile::TileReluBackwardOp>(
            op, "TILE_RELU_BACKWARD");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        return attrs;
    }
    if (name == "TILE_GELU_BACKWARD")
    {
        auto const &node = cast_op<tile::TileGeluBackwardOp>(
            op, "TILE_GELU_BACKWARD");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        return attrs;
    }
    if (name == "TILE_GELUTANH_BACKWARD")
    {
        auto const &node = cast_op<tile::TileGelutanhBackwardOp>(
            op, "TILE_GELUTANH_BACKWARD");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        return attrs;
    }
    if (name == "TILE_SILU_BACKWARD")
    {
        auto const &node = cast_op<tile::TileSiluBackwardOp>(
            op, "TILE_SILU_BACKWARD");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        return attrs;
    }
    if (name == "TILE_SUM")
    {
        auto const &node = cast_op<tile::TileSumOp>(
            op, "TILE_SUM");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        return attrs;
    }
    if (name == "TILE_SUM_SLICE")
    {
        auto const &node = cast_op<tile::TileSumSliceOp>(
            op, "TILE_SUM_SLICE");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        attrs["axis"] = node.axis;
        attrs["redux"] = node.redux;
        return attrs;
    }
    if (name == "TILE_SUM_FIBER")
    {
        auto const &node = cast_op<tile::TileSumFiberOp>(
            op, "TILE_SUM_FIBER");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        attrs["axis"] = node.axis;
        attrs["batch_ndim"] = node.batch_ndim;
        attrs["redux"] = node.redux;
        return attrs;
    }
    if (name == "TILE_SUMPROD_FIBER")
    {
        auto const &node = cast_op<tile::TileSumprodFiberOp>(
            op, "TILE_SUMPROD_FIBER");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        attrs["axis"] = node.axis;
        attrs["redux"] = node.redux;
        return attrs;
    }
    if (name == "TILE_SUMPROD_SLICE")
    {
        auto const &node = cast_op<tile::TileSumprodSliceOp>(
            op, "TILE_SUMPROD_SLICE");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        attrs["axis"] = node.axis;
        attrs["redux"] = node.redux;
        return attrs;
    }
    if (name == "TILE_NORM")
    {
        auto const &node = cast_op<tile::TileNormOp>(
            op, "TILE_NORM");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        return attrs;
    }
    if (name == "TILE_NORM_FIBER")
    {
        auto const &node = cast_op<tile::TileNormFiberOp>(
            op, "TILE_NORM_FIBER");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        attrs["axis"] = node.axis;
        attrs["batch_ndim"] = node.batch_ndim;
        attrs["redux"] = node.redux;
        return attrs;
    }
    if (name == "TILE_NORM_FIBER_INPLACE")
    {
        auto const &node = cast_op<tile::TileNormFiberInplaceOp>(
            op, "TILE_NORM_FIBER_INPLACE");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        attrs["axis"] = node.axis;
        attrs["batch_ndim"] = node.batch_ndim;
        attrs["redux"] = node.redux;
        return attrs;
    }
    if (name == "TILE_NORM_SLICE")
    {
        auto const &node = cast_op<tile::TileNormSliceOp>(
            op, "TILE_NORM_SLICE");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        attrs["axis"] = node.axis;
        attrs["redux"] = node.redux;
        return attrs;
    }
    if (name == "TILE_NORM_SLICE_INPLACE")
    {
        auto const &node = cast_op<tile::TileNormSliceInplaceOp>(
            op, "TILE_NORM_SLICE_INPLACE");
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        attrs["axis"] = node.axis;
        attrs["redux"] = node.redux;
        return attrs;
    }
    if (name == "TILE_MAXSUMEXP")
    {
        auto const &node = cast_op<tile::TileMaxsumexpOp>(
            op, "TILE_MAXSUMEXP");
        attrs["axis"] = node.axis;
        attrs["beta"] = node.beta;
        attrs["redux"] = node.redux;
        return attrs;
    }
    if (name == "TILE_SOFTMAX")
    {
        auto const &node = cast_op<tile::TileSoftmaxOp>(
            op, "TILE_SOFTMAX");
        attrs["alpha"] = node.alpha;
        attrs["axis"] = node.axis;
        return attrs;
    }
    if (name == "TILE_SOFTMAX_INPLACE")
    {
        auto const &node = cast_op<tile::TileSoftmaxInplaceOp>(
            op, "TILE_SOFTMAX_INPLACE");
        attrs["alpha"] = node.alpha;
        attrs["axis"] = node.axis;
        return attrs;
    }
    if (name == "TILE_TRANSPOSE")
    {
        auto const &node = cast_op<tile::TileTransposeOp>(
            op, "TILE_TRANSPOSE");
        attrs["alpha"] = node.alpha;
        attrs["ndim"] = node.ndim;
        return attrs;
    }
    if (name == "TILE_SWAP_TWO_AXES")
    {
        auto const &node = cast_op<tile::TileSwapTwoAxesOp>(
            op, "TILE_SWAP_TWO_AXES");
        attrs["dim0"] = node.dim0;
        attrs["dim1"] = node.dim1;
        return attrs;
    }
    if (name == "TILE_EMBEDDING")
    {
        auto const &node = cast_op<tile::TileEmbeddingOp>(
            op, "TILE_EMBEDDING");
        attrs["m"] = node.m;
        attrs["n"] = node.n;
        attrs["k"] = node.k;
        attrs["k_start"] = node.k_start;
        attrs["k_size"] = node.k_size;
        return attrs;
    }
    if (name == "TILE_EMBEDDING_BACKWARD")
    {
        auto const &node = cast_op<tile::TileEmbeddingBackwardOp>(
            op, "TILE_EMBEDDING_BACKWARD");
        attrs["m"] = node.m;
        attrs["n"] = node.n;
        attrs["k"] = node.k;
        attrs["k_start"] = node.k_start;
        attrs["k_size"] = node.k_size;
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        attrs["redux"] = node.redux;
        return attrs;
    }
    if (name == "TILE_ROPE")
    {
        auto const &node = cast_op<tile::TileRopeOp>(
            op, "TILE_ROPE");
        attrs["sin_pair0"] = node.sin_pair0;
        return attrs;
    }
    if (name == "TILE_ROPE_BACKWARD")
    {
        auto const &node = cast_op<tile::TileRopeBackwardOp>(
            op, "TILE_ROPE_BACKWARD");
        attrs["sin_pair0"] = node.sin_pair0;
        return attrs;
    }
    if (name == "TILE_MASK_SCALAR")
    {
        auto const &node = cast_op<tile::TileMaskScalarOp>(
            op, "TILE_MASK_SCALAR");
        attrs["value"] = node.val;
        attrs["batch_ndim"] = node.batch_ndim;
        return attrs;
    }
    if (name == "TILE_SUBTRACT_INDEXED_OUTPUTS")
    {
        auto const &node = cast_op<tile::TileSubtractIndexedOutputsOp>(
            op, "TILE_SUBTRACT_INDEXED_OUTPUTS");
        attrs["value"] = node.v;
        attrs["ignore_index"] = node.ignore_index;
        return attrs;
    }
    if (name == "TILE_TOTAL_SUM_ACCUM")
    {
        auto const &node = cast_op<tile::TileTotalSumAccumOp>(
            op, "TILE_TOTAL_SUM_ACCUM");
        attrs["alpha"] = node.alpha;
        attrs["ignore_index"] = node.ignore_index;
        return attrs;
    }
    if (name == "TILE_ADAM_STEP")
    {
        auto const &node = cast_op<tile::TileAdamStepOp>(
            op, "TILE_ADAM_STEP");
        attrs["num_iter"] = node.num_iter;
        attrs["beta_1"] = node.beta_1;
        attrs["beta_2"] = node.beta_2;
        attrs["eps"] = node.eps;
        attrs["lr"] = node.lr;
        attrs["weight_decay"] = node.weight_decay;
        return attrs;
    }
    if (name == "TILE_ADAMW_STEP")
    {
        auto const &node = cast_op<tile::TileAdamwStepOp>(
            op, "TILE_ADAMW_STEP");
        attrs["num_iter"] = node.num_iter;
        attrs["beta_1"] = node.beta_1;
        attrs["beta_2"] = node.beta_2;
        attrs["eps"] = node.eps;
        attrs["lr"] = node.lr;
        attrs["weight_decay"] = node.weight_decay;
        return attrs;
    }
    if (name == "TILE_SGD_STEP")
    {
        auto const &node = cast_op<tile::TileSgdStepOp>(
            op, "TILE_SGD_STEP");
        attrs["num_iter"] = node.num_iter;
        attrs["momentum"] = node.momentum;
        attrs["lr"] = node.lr;
        attrs["weight_decay"] = node.weight_decay;
        attrs["dampening"] = node.dampening;
        attrs["nesterov"] = node.nesterov;
        return attrs;
    }
    if (name == "TILE_RANDN")
    {
        auto const &node = cast_op<tile::TileRandnOp>(
            op, "TILE_RANDN");
        attrs["start"] = node.start;
        attrs["underlying_shape"] = node.underlying_shape;
        attrs["seed"] = node.seed;
        attrs["mean"] = node.mean;
        attrs["stddev"] = node.stddev;
        return attrs;
    }
    if (name == "TILE_LOG_SCALAR")
    {
        auto const &node = cast_op<tile::TileLogScalarOp>(
            op, "TILE_LOG_SCALAR");
        attrs["name"] = node.name;
        return attrs;
    }
    if (name == "TILE_CONV2D_INPLACE")
    {
        auto const &node = cast_op<tile::TileConv2dInplaceOp>(
            op, "TILE_CONV2D_INPLACE");
        attrs["src1_m"] = node.src1_m;
        attrs["src1_n"] = node.src1_n;
        attrs["src1_channels"] = node.src1_channels;
        attrs["batch"] = node.batch;
        attrs["src2_m"] = node.src2_m;
        attrs["src2_n"] = node.src2_n;
        attrs["dilation_m"] = node.dilation_m;
        attrs["dilation_n"] = node.dilation_n;
        attrs["dst_channels"] = node.dst_channels;
        attrs["offset_m"] = node.offset_m;
        attrs["offset_n"] = node.offset_n;
        attrs["dst_m"] = node.dst_m;
        attrs["dst_n"] = node.dst_n;
        attrs["stride_m"] = node.stride_m;
        attrs["stride_n"] = node.stride_n;
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        return attrs;
    }
    if (name == "TILE_CONV2D_BWD_INPUT_INPLACE")
    {
        auto const &node = cast_op<tile::TileConv2dBwdInputInplaceOp>(
            op, "TILE_CONV2D_BWD_INPUT_INPLACE");
        attrs["src1_m"] = node.src1_m;
        attrs["src1_n"] = node.src1_n;
        attrs["stride_m"] = node.stride_m;
        attrs["stride_n"] = node.stride_n;
        attrs["src1_channels"] = node.src1_channels;
        attrs["batch"] = node.batch;
        attrs["src2_m"] = node.src2_m;
        attrs["src2_n"] = node.src2_n;
        attrs["dilation_m"] = node.dilation_m;
        attrs["dilation_n"] = node.dilation_n;
        attrs["dst_channels"] = node.dst_channels;
        attrs["offset_m"] = node.offset_m;
        attrs["offset_n"] = node.offset_n;
        attrs["dst_m"] = node.dst_m;
        attrs["dst_n"] = node.dst_n;
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        return attrs;
    }
    if (name == "TILE_CONV2D_BWD_WEIGHT_INPLACE")
    {
        auto const &node = cast_op<tile::TileConv2dBwdWeightInplaceOp>(
            op, "TILE_CONV2D_BWD_WEIGHT_INPLACE");
        attrs["src1_m"] = node.src1_m;
        attrs["src1_n"] = node.src1_n;
        attrs["src1_channels"] = node.src1_channels;
        attrs["batch"] = node.batch;
        attrs["src2_m"] = node.src2_m;
        attrs["src2_n"] = node.src2_n;
        attrs["stride_m"] = node.stride_m;
        attrs["stride_n"] = node.stride_n;
        attrs["src2_channels"] = node.src2_channels;
        attrs["offset_m"] = node.offset_m;
        attrs["offset_n"] = node.offset_n;
        attrs["dst_m"] = node.dst_m;
        attrs["dst_n"] = node.dst_n;
        attrs["dilation_m"] = node.dilation_m;
        attrs["dilation_n"] = node.dilation_n;
        attrs["alpha"] = node.alpha;
        attrs["beta"] = node.beta;
        return attrs;
    }
#ifdef NNTILE_USE_FLASH_SDPA
    if (name == "TILE_FLASH_SDPA_FWD_CUDNN" ||
        name == "TILE_FLASH_SDPA_BWD_CUDNN")
    {
        return attrs;
    }
#endif
#ifdef NNTILE_TORCH_NATIVE_OPS
    if (name == "TILE_TORCH_UNARY")
    {
        auto const &u = cast_op<tile::TileTorchUnaryOp>(
            op, "TILE_TORCH_UNARY");
        return encode_torch_extra(u.kind, u.extra);
    }
    if (name == "TILE_TORCH_BINARY")
    {
        auto const &b = cast_op<tile::TileTorchBinaryOp>(
            op, "TILE_TORCH_BINARY");
        return encode_torch_extra(b.kind, b.extra);
    }
    if (name == "TILE_TORCH_TERNARY")
    {
        auto const &t =
            cast_op<tile::TileTorchTernaryOp>(
                op, "TILE_TORCH_TERNARY");
        return encode_torch_extra(t.kind, t.extra);
    }
#endif
    throw std::runtime_error("UnknownOp: " + name);
}

void apply_tile_op(TileGraph &graph, nlohmann::json const &op)
{
    std::string const name =
        op.at("op_name").get<std::string>();
    auto inputs = op.value("inputs", nlohmann::json::array());
    auto outputs = op.value("outputs", nlohmann::json::array());
    auto attrs = op.value("attrs", nlohmann::json::object());
    if (name == "TILE_ADD_INPLACE")
    {
        auto *x = arg(graph, inputs, 0, "TILE_ADD_INPLACE inputs");
        auto *y = arg(graph, inputs, 1, "TILE_ADD_INPLACE inputs");
        tile::add_inplace(
            attr_s(attrs, "alpha", 1.0f),
            x,
            attr_s(attrs, "beta", 1.0f),
            y
        );
        return;
    }
    if (name == "TILE_ADD")
    {
        auto *x = arg(graph, inputs, 0, "TILE_ADD inputs");
        auto *y = arg(graph, inputs, 1, "TILE_ADD inputs");
        auto *z = arg(graph, outputs, 0, "TILE_ADD outputs");
        tile::add(
            attr_s(attrs, "alpha", 1.0f),
            x,
            attr_s(attrs, "beta", 1.0f),
            y,
            z
        );
        return;
    }
    if (name == "TILE_MULTIPLY")
    {
        auto *x = arg(graph, inputs, 0, "TILE_MULTIPLY inputs");
        auto *y = arg(graph, inputs, 1, "TILE_MULTIPLY inputs");
        auto *z = arg(graph, outputs, 0, "TILE_MULTIPLY outputs");
        tile::multiply(
            attr_s(attrs, "alpha", 1.0f),
            x,
            y,
            z
        );
        return;
    }
    if (name == "TILE_MULTIPLY_INPLACE")
    {
        auto *s = arg(graph, inputs, 0, "TILE_MULTIPLY_INPLACE inputs");
        auto *d = arg(graph, inputs, 1, "TILE_MULTIPLY_INPLACE inputs");
        tile::multiply_inplace(
            attr_s(attrs, "alpha", 1.0f),
            s,
            d
        );
        return;
    }
    if (name == "TILE_GEMM")
    {
        auto *a = arg(graph, inputs, 0, "TILE_GEMM inputs");
        auto *b = arg(graph, inputs, 1, "TILE_GEMM inputs");
        TileGraph::NodeId c_id;
        if (inputs.size() >= 3)
        {
            c_id = inputs.at(2).get<TileGraph::NodeId>();
        }
        else
        {
            auto *c_out = arg(graph, outputs, 0, "TILE_GEMM outputs");
            c_id = c_out->id();
        }
        auto *c = tile_node_by_id(graph, c_id);
        tile::gemm(
            a,
            b,
            c,
            attr_s(attrs, "alpha", 1.0f),
            attr_s(attrs, "beta", 0.0f),
            attr_b(attrs, "trans_a", false),
            attr_b(attrs, "trans_b", false),
            attr_i(attrs, "ndim", 1),
            attr_i(attrs, "batch_ndim", 0)
        );
        return;
    }
    if (name == "TILE_ADD_SLICE")
    {
        auto *s1 = arg(graph, inputs, 0, "TILE_ADD_SLICE inputs");
        auto *s2 = arg(graph, inputs, 1, "TILE_ADD_SLICE inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_ADD_SLICE outputs");
        tile::add_slice(
            attr_s(attrs, "alpha", 1.0f),
            s1,
            attr_s(attrs, "beta", 1.0f),
            s2,
            dst,
            attr_i(attrs, "axis", 0)
        );
        return;
    }
    if (name == "TILE_ADD_SLICE_INPLACE")
    {
        auto *s = arg(graph, inputs, 0, "TILE_ADD_SLICE_INPLACE inputs");
        auto *d = arg(graph, inputs, 1, "TILE_ADD_SLICE_INPLACE inputs");
        tile::add_slice_inplace(
            attr_s(attrs, "alpha", 1.0f),
            s,
            attr_s(attrs, "beta", 1.0f),
            d,
            attr_i(attrs, "axis", 0)
        );
        return;
    }
    if (name == "TILE_ADD_FIBER")
    {
        auto *s1 = arg(graph, inputs, 0, "TILE_ADD_FIBER inputs");
        auto *s2 = arg(graph, inputs, 1, "TILE_ADD_FIBER inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_ADD_FIBER outputs");
        tile::add_fiber(
            attr_s(attrs, "alpha", 1.0f),
            s1,
            attr_s(attrs, "beta", 1.0f),
            s2,
            dst,
            attr_i(attrs, "axis", 0),
            attr_i(attrs, "batch_ndim", 0)
        );
        return;
    }
    if (name == "TILE_ADD_FIBER_INPLACE")
    {
        auto *s = arg(graph, inputs, 0, "TILE_ADD_FIBER_INPLACE inputs");
        auto *d = arg(graph, inputs, 1, "TILE_ADD_FIBER_INPLACE inputs");
        tile::add_fiber_inplace(
            attr_s(attrs, "alpha", 1.0f),
            s,
            attr_s(attrs, "beta", 1.0f),
            d,
            attr_i(attrs, "axis", 0),
            attr_i(attrs, "batch_ndim", 0)
        );
        return;
    }
    if (name == "TILE_FILL")
    {
        auto *x = arg(graph, outputs, 0, "TILE_FILL outputs");
        tile::fill(
            attr_s(attrs, "value", 0.0f),
            x
        );
        return;
    }
    if (name == "TILE_RELU")
    {
        auto *src = arg(graph, inputs, 0, "TILE_RELU inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_RELU outputs");
        tile::relu(
            src,
            dst
        );
        return;
    }
    if (name == "TILE_GELU")
    {
        auto *src = arg(graph, inputs, 0, "TILE_GELU inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_GELU outputs");
        tile::gelu(
            src,
            dst
        );
        return;
    }
    if (name == "TILE_GELUTANH")
    {
        auto *src = arg(graph, inputs, 0, "TILE_GELUTANH inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_GELUTANH outputs");
        tile::gelutanh(
            src,
            dst
        );
        return;
    }
    if (name == "TILE_SILU")
    {
        auto *src = arg(graph, inputs, 0, "TILE_SILU inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_SILU outputs");
        tile::silu(
            src,
            dst
        );
        return;
    }
    if (name == "TILE_SQRT")
    {
        auto *src = arg(graph, inputs, 0, "TILE_SQRT inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_SQRT outputs");
        tile::sqrt(
            src,
            dst
        );
        return;
    }
    if (name == "TILE_COPY")
    {
        auto *src = arg(graph, inputs, 0, "TILE_COPY inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_COPY outputs");
        tile::copy(
            src,
            dst
        );
        return;
    }
    if (name == "TILE_COPY_SAME_NUMEL")
    {
        auto *src = arg(graph, inputs, 0, "TILE_COPY_SAME_NUMEL inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_COPY_SAME_NUMEL outputs");
        tile::copy_same_numel(
            src,
            dst
        );
        return;
    }
    if (name == "TILE_LOGSUMEXP")
    {
        auto *src = arg(graph, inputs, 0, "TILE_LOGSUMEXP inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_LOGSUMEXP outputs");
        tile::logsumexp(
            src,
            dst
        );
        return;
    }
    if (name == "TILE_RELU_INPLACE")
    {
        auto *dst = arg(graph, inputs, 0, "TILE_RELU_INPLACE inputs");
        tile::relu_inplace(
            dst
        );
        return;
    }
    if (name == "TILE_GELU_INPLACE")
    {
        auto *dst = arg(graph, inputs, 0, "TILE_GELU_INPLACE inputs");
        tile::gelu_inplace(
            dst
        );
        return;
    }
    if (name == "TILE_GELUTANH_INPLACE")
    {
        auto *dst = arg(graph, inputs, 0, "TILE_GELUTANH_INPLACE inputs");
        tile::gelutanh_inplace(
            dst
        );
        return;
    }
    if (name == "TILE_SILU_INPLACE")
    {
        auto *dst = arg(graph, inputs, 0, "TILE_SILU_INPLACE inputs");
        tile::silu_inplace(
            dst
        );
        return;
    }
    if (name == "TILE_SQRT_INPLACE")
    {
        auto *dst = arg(graph, inputs, 0, "TILE_SQRT_INPLACE inputs");
        tile::sqrt_inplace(
            dst
        );
        return;
    }
    if (name == "TILE_CLEAR")
    {
        auto *x = arg(graph, outputs, 0, "TILE_CLEAR outputs");
        tile::clear(
            x
        );
        return;
    }
    if (name == "TILE_UNREGISTER")
    {
        auto *x = arg(graph, inputs, 0, "TILE_UNREGISTER inputs");
        tile::unregister(
            x
        );
        return;
    }
    if (name == "TILE_INVALIDATE")
    {
        auto *x = arg(graph, inputs, 0, "TILE_INVALIDATE inputs");
        tile::invalidate(
            x
        );
        return;
    }
    if (name == "TILE_COPY_INTERSECTION")
    {
        auto *src = arg(graph, inputs, 0, "TILE_COPY_INTERSECTION inputs");
        auto *dst = arg(graph, inputs, 1, "TILE_COPY_INTERSECTION inputs");
        auto *scratch = arg(graph, inputs, 2, "TILE_COPY_INTERSECTION inputs");
        auto src_off = attrs.value(
            "src_offset", std::vector<Index>{});
        auto dst_off = attrs.value(
            "dst_offset", std::vector<Index>{});
        tile::copy_intersection(
            src,
            src_off,
            dst,
            dst_off,
            scratch
        );
        return;
    }
    if (name == "TILE_SCALE")
    {
        auto *src = arg(graph, inputs, 0, "TILE_SCALE inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_SCALE outputs");
        tile::scale(
            attr_s(attrs, "alpha", 1.0f),
            src,
            dst
        );
        return;
    }
    if (name == "TILE_SCALE_INPLACE")
    {
        auto *dst = arg(graph, inputs, 0, "TILE_SCALE_INPLACE inputs");
        tile::scale_inplace(
            attr_s(attrs, "alpha", 1.0f),
            dst
        );
        return;
    }
    if (name == "TILE_SCALE_SLICE")
    {
        auto *s = arg(graph, inputs, 0, "TILE_SCALE_SLICE inputs");
        auto *d = arg(graph, inputs, 1, "TILE_SCALE_SLICE inputs");
        tile::scale_slice(
            attr_s(attrs, "alpha", 1.0f),
            s,
            d,
            attr_i(attrs, "axis", 0)
        );
        return;
    }
    if (name == "TILE_SCALE_FIBER")
    {
        auto *s = arg(graph, inputs, 0, "TILE_SCALE_FIBER inputs");
        auto *d = arg(graph, inputs, 1, "TILE_SCALE_FIBER inputs");
        tile::scale_fiber(
            attr_s(attrs, "alpha", 1.0f),
            s,
            d,
            attr_i(attrs, "axis", 0),
            attr_i(attrs, "batch_ndim", 0)
        );
        return;
    }
    if (name == "TILE_MULTIPLY_SLICE")
    {
        auto *s = arg(graph, inputs, 0, "TILE_MULTIPLY_SLICE inputs");
        auto *d = arg(graph, inputs, 1, "TILE_MULTIPLY_SLICE inputs");
        tile::multiply_slice(
            attr_s(attrs, "alpha", 1.0f),
            s,
            d,
            attr_i(attrs, "axis", 0)
        );
        return;
    }
    if (name == "TILE_MULTIPLY_FIBER")
    {
        auto *t1 = arg(graph, inputs, 0, "TILE_MULTIPLY_FIBER inputs");
        auto *t2 = arg(graph, inputs, 1, "TILE_MULTIPLY_FIBER inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_MULTIPLY_FIBER outputs");
        tile::multiply_fiber(
            attr_s(attrs, "alpha", 1.0f),
            t1,
            t2,
            dst,
            attr_i(attrs, "axis", 0)
        );
        return;
    }
    if (name == "TILE_MULTIPLY_FIBER_INPLACE")
    {
        auto *s = arg(graph, inputs, 0, "TILE_MULTIPLY_FIBER_INPLACE inputs");
        auto *d = arg(graph, inputs, 1, "TILE_MULTIPLY_FIBER_INPLACE inputs");
        tile::multiply_fiber_inplace(
            attr_s(attrs, "alpha", 1.0f),
            s,
            d,
            attr_i(attrs, "axis", 0)
        );
        return;
    }
    if (name == "TILE_POW")
    {
        auto *a = arg(graph, inputs, 0, "TILE_POW inputs");
        tile::pow(
            attr_s(attrs, "alpha", 1.0f),
            attr_s(attrs, "exp", 1.0f),
            a
        );
        return;
    }
    if (name == "TILE_HYPOT")
    {
        auto *src1 = arg(graph, inputs, 0, "TILE_HYPOT inputs");
        auto *src2 = arg(graph, inputs, 1, "TILE_HYPOT inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_HYPOT outputs");
        tile::hypot(
            attr_s(attrs, "alpha", 1.0f),
            src1,
            attr_s(attrs, "beta", 1.0f),
            src2,
            dst
        );
        return;
    }
    if (name == "TILE_HYPOT_INPLACE")
    {
        auto *src = arg(graph, inputs, 0, "TILE_HYPOT_INPLACE inputs");
        auto *dst = arg(graph, inputs, 1, "TILE_HYPOT_INPLACE inputs");
        tile::hypot_inplace(
            attr_s(attrs, "alpha", 1.0f),
            src,
            attr_s(attrs, "beta", 1.0f),
            dst
        );
        return;
    }
    if (name == "TILE_HYPOT_SCALAR_INVERSE")
    {
        auto *dst = arg(graph, inputs, 0, "TILE_HYPOT_SCALAR_INVERSE inputs");
        tile::hypot_scalar_inverse(
            attr_s(attrs, "eps", 0.0f),
            attr_s(attrs, "alpha", 1.0f),
            dst
        );
        return;
    }
    if (name == "TILE_RELU_BACKWARD")
    {
        auto *x = arg(graph, inputs, 0, "TILE_RELU_BACKWARD inputs");
        auto *dy = arg(graph, inputs, 1, "TILE_RELU_BACKWARD inputs");
        auto *dx = arg(graph, outputs, 0, "TILE_RELU_BACKWARD outputs");
        tile::relu_backward(
            attr_s(attrs, "alpha", 1.0f),
            x,
            dy,
            attr_s(attrs, "beta", 0.0f),
            dx
        );
        return;
    }
    if (name == "TILE_GELU_BACKWARD")
    {
        auto *x = arg(graph, inputs, 0, "TILE_GELU_BACKWARD inputs");
        auto *dy = arg(graph, inputs, 1, "TILE_GELU_BACKWARD inputs");
        auto *dx = arg(graph, outputs, 0, "TILE_GELU_BACKWARD outputs");
        tile::gelu_backward(
            attr_s(attrs, "alpha", 1.0f),
            x,
            dy,
            attr_s(attrs, "beta", 0.0f),
            dx
        );
        return;
    }
    if (name == "TILE_GELUTANH_BACKWARD")
    {
        auto *x = arg(graph, inputs, 0, "TILE_GELUTANH_BACKWARD inputs");
        auto *dy = arg(graph, inputs, 1, "TILE_GELUTANH_BACKWARD inputs");
        auto *dx = arg(graph, outputs, 0, "TILE_GELUTANH_BACKWARD outputs");
        tile::gelutanh_backward(
            attr_s(attrs, "alpha", 1.0f),
            x,
            dy,
            attr_s(attrs, "beta", 0.0f),
            dx
        );
        return;
    }
    if (name == "TILE_SILU_BACKWARD")
    {
        auto *x = arg(graph, inputs, 0, "TILE_SILU_BACKWARD inputs");
        auto *dy = arg(graph, inputs, 1, "TILE_SILU_BACKWARD inputs");
        auto *dx = arg(graph, outputs, 0, "TILE_SILU_BACKWARD outputs");
        tile::silu_backward(
            attr_s(attrs, "alpha", 1.0f),
            x,
            dy,
            attr_s(attrs, "beta", 0.0f),
            dx
        );
        return;
    }
    if (name == "TILE_SUM")
    {
        auto *src = arg(graph, inputs, 0, "TILE_SUM inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_SUM outputs");
        tile::sum(
            attr_s(attrs, "alpha", 1.0f),
            src,
            attr_s(attrs, "beta", 0.0f),
            dst
        );
        return;
    }
    if (name == "TILE_SUM_SLICE")
    {
        auto *src = arg(graph, inputs, 0, "TILE_SUM_SLICE inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_SUM_SLICE outputs");
        tile::sum_slice(
            attr_s(attrs, "alpha", 1.0f),
            src,
            attr_s(attrs, "beta", 0.0f),
            dst,
            attr_i(attrs, "axis", 0),
            attr_n(attrs, "redux", 0)
        );
        return;
    }
    if (name == "TILE_SUM_FIBER")
    {
        auto *src = arg(graph, inputs, 0, "TILE_SUM_FIBER inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_SUM_FIBER outputs");
        tile::sum_fiber(
            attr_s(attrs, "alpha", 1.0f),
            src,
            attr_s(attrs, "beta", 0.0f),
            dst,
            attr_i(attrs, "axis", 0),
            attr_i(attrs, "batch_ndim", 0),
            attr_n(attrs, "redux", 0)
        );
        return;
    }
    if (name == "TILE_SUMPROD_FIBER")
    {
        auto *a = arg(graph, inputs, 0, "TILE_SUMPROD_FIBER inputs");
        auto *b = arg(graph, inputs, 1, "TILE_SUMPROD_FIBER inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_SUMPROD_FIBER outputs");
        tile::sumprod_fiber(
            attr_s(attrs, "alpha", 1.0f),
            a,
            b,
            attr_s(attrs, "beta", 0.0f),
            dst,
            attr_i(attrs, "axis", 0),
            attr_n(attrs, "redux", 0)
        );
        return;
    }
    if (name == "TILE_SUMPROD_SLICE")
    {
        auto *t1 = arg(graph, inputs, 0, "TILE_SUMPROD_SLICE inputs");
        auto *t2 = arg(graph, inputs, 1, "TILE_SUMPROD_SLICE inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_SUMPROD_SLICE outputs");
        tile::sumprod_slice(
            attr_s(attrs, "alpha", 1.0f),
            t1,
            t2,
            attr_s(attrs, "beta", 0.0f),
            dst,
            attr_i(attrs, "axis", 0),
            attr_n(attrs, "redux", 0)
        );
        return;
    }
    if (name == "TILE_NORM")
    {
        auto *src = arg(graph, inputs, 0, "TILE_NORM inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_NORM outputs");
        tile::norm(
            attr_s(attrs, "alpha", 1.0f),
            src,
            attr_s(attrs, "beta", 0.0f),
            dst
        );
        return;
    }
    if (name == "TILE_NORM_FIBER")
    {
        auto *t1 = arg(graph, inputs, 0, "TILE_NORM_FIBER inputs");
        auto *t2 = arg(graph, inputs, 1, "TILE_NORM_FIBER inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_NORM_FIBER outputs");
        tile::norm_fiber(
            attr_s(attrs, "alpha", 1.0f),
            t1,
            attr_s(attrs, "beta", 0.0f),
            t2,
            dst,
            attr_i(attrs, "axis", 0),
            attr_i(attrs, "batch_ndim", 0),
            attr_n(attrs, "redux", 0)
        );
        return;
    }
    if (name == "TILE_NORM_FIBER_INPLACE")
    {
        auto *s = arg(graph, inputs, 0, "TILE_NORM_FIBER_INPLACE inputs");
        auto *d = arg(graph, inputs, 1, "TILE_NORM_FIBER_INPLACE inputs");
        tile::norm_fiber_inplace(
            attr_s(attrs, "alpha", 1.0f),
            s,
            attr_s(attrs, "beta", 0.0f),
            d,
            attr_i(attrs, "axis", 0),
            attr_i(attrs, "batch_ndim", 0),
            attr_n(attrs, "redux", 0)
        );
        return;
    }
    if (name == "TILE_NORM_SLICE")
    {
        auto *t1 = arg(graph, inputs, 0, "TILE_NORM_SLICE inputs");
        auto *t2 = arg(graph, inputs, 1, "TILE_NORM_SLICE inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_NORM_SLICE outputs");
        tile::norm_slice(
            attr_s(attrs, "alpha", 1.0f),
            t1,
            attr_s(attrs, "beta", 0.0f),
            t2,
            dst,
            attr_i(attrs, "axis", 0),
            attr_n(attrs, "redux", 0)
        );
        return;
    }
    if (name == "TILE_NORM_SLICE_INPLACE")
    {
        auto *s = arg(graph, inputs, 0, "TILE_NORM_SLICE_INPLACE inputs");
        auto *d = arg(graph, inputs, 1, "TILE_NORM_SLICE_INPLACE inputs");
        tile::norm_slice_inplace(
            attr_s(attrs, "alpha", 1.0f),
            s,
            attr_s(attrs, "beta", 0.0f),
            d,
            attr_i(attrs, "axis", 0),
            attr_n(attrs, "redux", 0)
        );
        return;
    }
    if (name == "TILE_MAXSUMEXP")
    {
        auto *src = arg(graph, inputs, 0, "TILE_MAXSUMEXP inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_MAXSUMEXP outputs");
        tile::maxsumexp(
            src,
            dst,
            attr_i(attrs, "axis", 0),
            attr_s(attrs, "beta", 0.0f),
            attr_n(attrs, "redux", 0)
        );
        return;
    }
    if (name == "TILE_SOFTMAX")
    {
        auto *mse = arg(graph, inputs, 0, "TILE_SOFTMAX inputs");
        auto *src = arg(graph, inputs, 1, "TILE_SOFTMAX inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_SOFTMAX outputs");
        tile::softmax(
            mse,
            src,
            attr_s(attrs, "alpha", 1.0f),
            dst,
            attr_i(attrs, "axis", 0)
        );
        return;
    }
    if (name == "TILE_SOFTMAX_INPLACE")
    {
        auto *mse = arg(graph, inputs, 0, "TILE_SOFTMAX_INPLACE inputs");
        auto *dst = arg(graph, inputs, 1, "TILE_SOFTMAX_INPLACE inputs");
        tile::softmax_inplace(
            mse,
            attr_s(attrs, "alpha", 1.0f),
            dst,
            attr_i(attrs, "axis", 0)
        );
        return;
    }
    if (name == "TILE_TRANSPOSE")
    {
        auto *src = arg(graph, inputs, 0, "TILE_TRANSPOSE inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_TRANSPOSE outputs");
        tile::transpose(
            attr_s(attrs, "alpha", 1.0f),
            src,
            dst,
            attr_i(attrs, "ndim", 0)
        );
        return;
    }
    if (name == "TILE_SWAP_TWO_AXES")
    {
        auto *src = arg(graph, inputs, 0, "TILE_SWAP_TWO_AXES inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_SWAP_TWO_AXES outputs");
        tile::swap_two_axes(
            src,
            dst,
            attr_i(attrs, "dim0", 0),
            attr_i(attrs, "dim1", 0)
        );
        return;
    }
    if (name == "TILE_EMBEDDING")
    {
        auto *index = arg(graph, inputs, 0, "TILE_EMBEDDING inputs");
        auto *vocab = arg(graph, inputs, 1, "TILE_EMBEDDING inputs");
        auto *embed = arg(graph, outputs, 0, "TILE_EMBEDDING outputs");
        tile::embedding(
            attr_i(attrs, "m", 0),
            attr_i(attrs, "n", 0),
            attr_i(attrs, "k", 0),
            attr_i(attrs, "k_start", 0),
            attr_i(attrs, "k_size", 0),
            index,
            vocab,
            embed
        );
        return;
    }
    if (name == "TILE_EMBEDDING_BACKWARD")
    {
        auto *index = arg(graph, inputs, 0, "TILE_EMBEDDING_BACKWARD inputs");
        auto *embed = arg(graph, inputs, 1, "TILE_EMBEDDING_BACKWARD inputs");
        auto *vocab = arg(graph, inputs, 2, "TILE_EMBEDDING_BACKWARD inputs");
        tile::embedding_backward(
            attr_i(attrs, "m", 0),
            attr_i(attrs, "n", 0),
            attr_i(attrs, "k", 0),
            attr_i(attrs, "k_start", 0),
            attr_i(attrs, "k_size", 0),
            attr_s(attrs, "alpha", 1.0f),
            attr_s(attrs, "beta", 1.0f),
            index,
            embed,
            vocab,
            attr_n(attrs, "redux", 0)
        );
        return;
    }
    if (name == "TILE_ROPE")
    {
        auto *sin = arg(graph, inputs, 0, "TILE_ROPE inputs");
        auto *cos = arg(graph, inputs, 1, "TILE_ROPE inputs");
        auto *src = arg(graph, inputs, 2, "TILE_ROPE inputs");
        auto *dst = arg(graph, outputs, 0, "TILE_ROPE outputs");
        tile::rope(
            sin,
            cos,
            src,
            dst,
            attr_i(attrs, "sin_pair0", 0)
        );
        return;
    }
    if (name == "TILE_ROPE_BACKWARD")
    {
        auto *sin = arg(graph, inputs, 0, "TILE_ROPE_BACKWARD inputs");
        auto *cos = arg(graph, inputs, 1, "TILE_ROPE_BACKWARD inputs");
        auto *dy = arg(graph, inputs, 2, "TILE_ROPE_BACKWARD inputs");
        auto *dx = arg(graph, outputs, 0, "TILE_ROPE_BACKWARD outputs");
        tile::rope_backward(
            sin,
            cos,
            dy,
            dx,
            attr_i(attrs, "sin_pair0", 0)
        );
        return;
    }
    if (name == "TILE_MASK_SCALAR")
    {
        auto *mask = arg(graph, inputs, 0, "TILE_MASK_SCALAR inputs");
        auto *a = arg(graph, inputs, 1, "TILE_MASK_SCALAR inputs");
        tile::mask_scalar(
            mask,
            attr_s(attrs, "value", 0.0f),
            a,
            attr_i(attrs, "batch_ndim", 0)
        );
        return;
    }
    if (name == "TILE_SUBTRACT_INDEXED_OUTPUTS")
    {
        auto *labels = arg(
            graph,
            inputs,
            0,
            "TILE_SUBTRACT_INDEXED_OUTPUTS inputs");
        auto *dst = arg(
            graph,
            inputs,
            1,
            "TILE_SUBTRACT_INDEXED_OUTPUTS inputs");
        tile::subtract_indexed_outputs(
            attr_s(attrs, "value", 1.0f),
            labels,
            dst,
            attr_i(attrs, "ignore_index", 0)
        );
        return;
    }
    if (name == "TILE_TOTAL_SUM_ACCUM")
    {
        auto *lse = arg(graph, inputs, 0, "TILE_TOTAL_SUM_ACCUM inputs");
        auto *src = arg(graph, inputs, 1, "TILE_TOTAL_SUM_ACCUM inputs");
        auto *labels = arg(graph, inputs, 2, "TILE_TOTAL_SUM_ACCUM inputs");
        auto *val = arg(graph, inputs, 3, "TILE_TOTAL_SUM_ACCUM inputs");
        tile::total_sum_accum(
            attr_s(attrs, "alpha", 1.0f),
            lse,
            src,
            labels,
            val,
            attr_i(attrs, "ignore_index", 0)
        );
        return;
    }
    if (name == "TILE_ADAM_STEP")
    {
        auto *grad = arg(graph, inputs, 0, "TILE_ADAM_STEP inputs");
        auto *m = arg(graph, inputs, 1, "TILE_ADAM_STEP inputs");
        auto *v = arg(graph, inputs, 2, "TILE_ADAM_STEP inputs");
        auto *p = arg(graph, inputs, 3, "TILE_ADAM_STEP inputs");
        tile::adam_step(
            attr_i(attrs, "num_iter", 0),
            attr_s(attrs, "beta_1", 0.9f),
            attr_s(attrs, "beta_2", 0.999f),
            attr_s(attrs, "eps", 1e-8f),
            attr_s(attrs, "lr", 0.001f),
            attr_s(attrs, "weight_decay", 0.0f),
            grad,
            m,
            v,
            p
        );
        return;
    }
    if (name == "TILE_ADAMW_STEP")
    {
        auto *grad = arg(graph, inputs, 0, "TILE_ADAMW_STEP inputs");
        auto *m = arg(graph, inputs, 1, "TILE_ADAMW_STEP inputs");
        auto *v = arg(graph, inputs, 2, "TILE_ADAMW_STEP inputs");
        auto *p = arg(graph, inputs, 3, "TILE_ADAMW_STEP inputs");
        tile::adamw_step(
            attr_i(attrs, "num_iter", 0),
            attr_s(attrs, "beta_1", 0.9f),
            attr_s(attrs, "beta_2", 0.999f),
            attr_s(attrs, "eps", 1e-8f),
            attr_s(attrs, "lr", 0.001f),
            attr_s(attrs, "weight_decay", 0.0f),
            grad,
            m,
            v,
            p
        );
        return;
    }
    if (name == "TILE_SGD_STEP")
    {
        auto *grad = arg(graph, inputs, 0, "TILE_SGD_STEP inputs");
        auto *vel = arg(graph, inputs, 1, "TILE_SGD_STEP inputs");
        auto *p = arg(graph, inputs, 2, "TILE_SGD_STEP inputs");
        tile::sgd_step(
            attr_i(attrs, "num_iter", 0),
            attr_s(attrs, "momentum", 0.0f),
            attr_s(attrs, "lr", 0.001f),
            attr_s(attrs, "weight_decay", 0.0f),
            attr_s(attrs, "dampening", 0.0f),
            attr_b(attrs, "nesterov", false),
            grad,
            vel,
            p
        );
        return;
    }
    if (name == "TILE_RANDN")
    {
        auto *dst = arg(graph, outputs, 0, "TILE_RANDN outputs");
        auto start = attrs.value(
            "start", std::vector<Index>{});
        auto ushape = attrs.value(
            "underlying_shape", std::vector<Index>{});
        unsigned long long seed = 0;
        if (attrs.contains("seed"))
        {
            seed = attrs.at("seed").get<unsigned long long>();
        }
        tile::randn(
            dst,
            start,
            ushape,
            seed,
            attr_s(attrs, "mean", 0.0f),
            attr_s(attrs, "stddev", 1.0f)
        );
        return;
    }
    if (name == "TILE_LOG_SCALAR")
    {
        auto *value = arg(graph, inputs, 0, "TILE_LOG_SCALAR inputs");
        tile::log_scalar(
            attrs.value("name", std::string()), value);
        return;
    }
    if (name == "TILE_CONV2D_INPLACE")
    {
        auto *src1 = arg(graph, inputs, 0, "TILE_CONV2D_INPLACE inputs");
        auto *src2 = arg(graph, inputs, 1, "TILE_CONV2D_INPLACE inputs");
        auto *dst = arg(graph, inputs, 2, "TILE_CONV2D_INPLACE inputs");
        tile::conv2d_inplace(
            attr_i(attrs, "src1_m", 0),
            attr_i(attrs, "src1_n", 0),
            attr_i(attrs, "src1_channels", 0),
            attr_i(attrs, "batch", 0),
            attr_i(attrs, "src2_m", 0),
            attr_i(attrs, "src2_n", 0),
            attr_i(attrs, "dilation_m", 0),
            attr_i(attrs, "dilation_n", 0),
            attr_i(attrs, "dst_channels", 0),
            attr_i(attrs, "offset_m", 0),
            attr_i(attrs, "offset_n", 0),
            attr_s(attrs, "alpha", 1.0f),
            src1,
            src2,
            attr_i(attrs, "dst_m", 0),
            attr_i(attrs, "dst_n", 0),
            attr_i(attrs, "stride_m", 0),
            attr_i(attrs, "stride_n", 0),
            attr_s(attrs, "beta", 0.0f),
            dst
        );
        return;
    }
    if (name == "TILE_CONV2D_BWD_INPUT_INPLACE")
    {
        auto *src1 = arg(
            graph,
            inputs,
            0,
            "TILE_CONV2D_BWD_INPUT_INPLACE inputs");
        auto *src2 = arg(
            graph,
            inputs,
            1,
            "TILE_CONV2D_BWD_INPUT_INPLACE inputs");
        auto *dst = arg(
            graph,
            inputs,
            2,
            "TILE_CONV2D_BWD_INPUT_INPLACE inputs");
        tile::conv2d_bwd_input_inplace(
            attr_i(attrs, "src1_m", 0),
            attr_i(attrs, "src1_n", 0),
            attr_i(attrs, "stride_m", 0),
            attr_i(attrs, "stride_n", 0),
            attr_i(attrs, "src1_channels", 0),
            attr_i(attrs, "batch", 0),
            attr_i(attrs, "src2_m", 0),
            attr_i(attrs, "src2_n", 0),
            attr_i(attrs, "dilation_m", 0),
            attr_i(attrs, "dilation_n", 0),
            attr_i(attrs, "dst_channels", 0),
            attr_i(attrs, "offset_m", 0),
            attr_i(attrs, "offset_n", 0),
            attr_s(attrs, "alpha", 1.0f),
            src1,
            src2,
            attr_i(attrs, "dst_m", 0),
            attr_i(attrs, "dst_n", 0),
            attr_s(attrs, "beta", 0.0f),
            dst
        );
        return;
    }
    if (name == "TILE_CONV2D_BWD_WEIGHT_INPLACE")
    {
        auto *src1 = arg(
            graph,
            inputs,
            0,
            "TILE_CONV2D_BWD_WEIGHT_INPLACE inputs");
        auto *src2 = arg(
            graph,
            inputs,
            1,
            "TILE_CONV2D_BWD_WEIGHT_INPLACE inputs");
        auto *dst = arg(
            graph,
            inputs,
            2,
            "TILE_CONV2D_BWD_WEIGHT_INPLACE inputs");
        tile::conv2d_bwd_weight_inplace(
            attr_i(attrs, "src1_m", 0),
            attr_i(attrs, "src1_n", 0),
            attr_i(attrs, "src1_channels", 0),
            attr_i(attrs, "batch", 0),
            attr_i(attrs, "src2_m", 0),
            attr_i(attrs, "src2_n", 0),
            attr_i(attrs, "stride_m", 0),
            attr_i(attrs, "stride_n", 0),
            attr_i(attrs, "src2_channels", 0),
            attr_i(attrs, "offset_m", 0),
            attr_i(attrs, "offset_n", 0),
            attr_s(attrs, "alpha", 1.0f),
            src1,
            src2,
            attr_i(attrs, "dst_m", 0),
            attr_i(attrs, "dst_n", 0),
            attr_i(attrs, "dilation_m", 0),
            attr_i(attrs, "dilation_n", 0),
            attr_s(attrs, "beta", 0.0f),
            dst
        );
        return;
    }
#ifdef NNTILE_USE_FLASH_SDPA
    if (name == "TILE_FLASH_SDPA_FWD_CUDNN")
    {
        auto *K = arg(graph, inputs, 0, "TILE_FLASH_SDPA_FWD_CUDNN inputs");
        auto *Q = arg(graph, inputs, 1, "TILE_FLASH_SDPA_FWD_CUDNN inputs");
        auto *mask = arg(graph, inputs, 2, "TILE_FLASH_SDPA_FWD_CUDNN inputs");
        auto *lse = arg(graph, inputs, 3, "TILE_FLASH_SDPA_FWD_CUDNN inputs");
        auto *V = arg(graph, inputs, 4, "TILE_FLASH_SDPA_FWD_CUDNN inputs");
        auto *A = arg(graph, inputs, 5, "TILE_FLASH_SDPA_FWD_CUDNN inputs");
        tile::flash_sdpa_fwd_cudnn(
            K,
            Q,
            mask,
            lse,
            V,
            A
        );
        return;
    }
    if (name == "TILE_FLASH_SDPA_BWD_CUDNN")
    {
        auto *K = arg(graph, inputs, 0, "TILE_FLASH_SDPA_BWD_CUDNN inputs");
        auto *Q = arg(graph, inputs, 1, "TILE_FLASH_SDPA_BWD_CUDNN inputs");
        auto *V = arg(graph, inputs, 2, "TILE_FLASH_SDPA_BWD_CUDNN inputs");
        auto *A = arg(graph, inputs, 3, "TILE_FLASH_SDPA_BWD_CUDNN inputs");
        auto *dA = arg(graph, inputs, 4, "TILE_FLASH_SDPA_BWD_CUDNN inputs");
        auto *mask = arg(graph, inputs, 5, "TILE_FLASH_SDPA_BWD_CUDNN inputs");
        auto *lse = arg(graph, inputs, 6, "TILE_FLASH_SDPA_BWD_CUDNN inputs");
        auto *dK = arg(graph, inputs, 7, "TILE_FLASH_SDPA_BWD_CUDNN inputs");
        auto *dQ = arg(graph, inputs, 8, "TILE_FLASH_SDPA_BWD_CUDNN inputs");
        auto *dV = arg(graph, inputs, 9, "TILE_FLASH_SDPA_BWD_CUDNN inputs");
        tile::flash_sdpa_bwd_cudnn(
            K,
            Q,
            V,
            A,
            dA,
            mask,
            lse,
            dK,
            dQ,
            dV
        );
        return;
    }
#endif
#ifdef NNTILE_TORCH_NATIVE_OPS
    if (name == "TILE_TORCH_UNARY")
    {
        auto extra = decode_torch_extra(attrs);
        extra.kind = static_cast<starpu::TorchKind>(
            attrs.value("kind", 0));
        tile::torch_unary(
            extra.kind,
            arg(graph, inputs, 0, "TILE_TORCH_UNARY inputs"),
            arg(graph, outputs, 0, "TILE_TORCH_UNARY outputs"),
            extra
        );
        return;
    }
    if (name == "TILE_TORCH_BINARY")
    {
        auto extra = decode_torch_extra(attrs);
        extra.kind = static_cast<starpu::TorchKind>(
            attrs.value("kind", 0));
        tile::torch_binary(
            extra.kind,
            arg(graph, inputs, 0, "TILE_TORCH_BINARY inputs"),
            arg(graph, inputs, 1, "TILE_TORCH_BINARY inputs"),
            arg(graph, outputs, 0, "TILE_TORCH_BINARY outputs"),
            extra
        );
        return;
    }
    if (name == "TILE_TORCH_TERNARY")
    {
        auto extra = decode_torch_extra(attrs);
        extra.kind = static_cast<starpu::TorchKind>(
            attrs.value("kind", 0));
        tile::torch_ternary(
            extra.kind,
            arg(graph, inputs, 0, "TILE_TORCH_TERNARY inputs"),
            arg(graph, inputs, 1, "TILE_TORCH_TERNARY inputs"),
            arg(graph, inputs, 2, "TILE_TORCH_TERNARY inputs"),
            arg(graph, outputs, 0, "TILE_TORCH_TERNARY outputs"),
            extra
        );
        return;
    }
#endif
    throw std::runtime_error("UnknownOp: " + name);
}

} // namespace nntile
