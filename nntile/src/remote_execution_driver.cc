/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/remote_execution_driver.cc
 * Unix-socket TileGraph driver (NNTILE_DRIVER_SOCKET).
 *
 * @version 1.1.0
 * */

#include <nntile/remote_execution_driver.hh>

#include <nntile/context.hh>
#include <nntile/dtype.hh>
#include <nntile/tile/ops/add.hh>
#include <nntile/tile/ops/add_inplace.hh>
#include <nntile/tile/ops/add_slice.hh>
#include <nntile/tile/ops/copy.hh>
#include <nntile/tile/ops/copy_intersection.hh>
#include <nntile/tile/ops/fill.hh>
#include <nntile/tile/ops/gemm.hh>
#include <nntile/tile/ops/multiply.hh>
#include <nntile/tile/ops/relu.hh>
#include <nntile/tile/ops/unregister.hh>
#ifdef NNTILE_TORCH_NATIVE_OPS
#include <nntile/tile/ops/torch_dispatch.hh>
#endif

#include <arpa/inet.h>
#include <grp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace nntile
{

namespace
{

constexpr uint32_t kMaxMsg = 64u * 1024u * 1024u;
constexpr int kDriverProtocol = 1;

void throw_errno(char const *what)
{
    throw std::runtime_error(
        std::string(what) + ": " + std::strerror(errno));
}

void write_all(int fd, void const *buf, size_t n)
{
    auto const *p = static_cast<char const *>(buf);
    size_t off = 0;
    while (off < n)
    {
        ssize_t const w = ::write(fd, p + off, n - off);
        if (w < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            throw_errno("write");
        }
        if (w == 0)
        {
            throw std::runtime_error("write: socket closed");
        }
        off += static_cast<size_t>(w);
    }
}

void read_all(int fd, void *buf, size_t n)
{
    auto *p = static_cast<char *>(buf);
    size_t off = 0;
    while (off < n)
    {
        ssize_t const r = ::read(fd, p + off, n - off);
        if (r < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            throw_errno("read");
        }
        if (r == 0)
        {
            throw std::runtime_error("read: socket closed");
        }
        off += static_cast<size_t>(r);
    }
}

void send_json(int fd, nlohmann::json const &msg)
{
    std::string const body = msg.dump();
    if (body.size() > kMaxMsg)
    {
        throw std::runtime_error("driver message too large");
    }
    uint32_t n = htonl(static_cast<uint32_t>(body.size()));
    write_all(fd, &n, sizeof(n));
    write_all(fd, body.data(), body.size());
}

nlohmann::json recv_json(int fd)
{
    uint32_t n = 0;
    read_all(fd, &n, sizeof(n));
    n = ntohl(n);
    if (n == 0 || n > kMaxMsg)
    {
        throw std::runtime_error("driver message too large");
    }
    std::string body(n, '\0');
    read_all(fd, body.data(), body.size());
    return nlohmann::json::parse(body);
}

nlohmann::json require_ok(nlohmann::json const &reply, char const *op)
{
    if (reply.contains("type") && reply["type"] == "Error")
    {
        throw std::runtime_error(
            std::string("RemoteExecutionDriver ") + op + ": " +
            reply.value("code", "Error") + " " +
            reply.value("message", ""));
    }
    return reply;
}

DataType dtype_from_string(std::string const &name)
{
    if (name == "FP32" || name == "fp32" || name == "float32")
    {
        return DataType::FP32;
    }
    if (name == "INT64" || name == "int64")
    {
        return DataType::INT64;
    }
    throw std::runtime_error("UnknownOp: dtype " + name);
}

void require_min(
    nlohmann::json const &arr, size_t n, char const *what)
{
    if (arr.size() < n)
    {
        throw std::runtime_error(std::string("UnknownOp: ") + what);
    }
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

nlohmann::json encode_attrs(TileGraph::OpNode const &op)
{
    nlohmann::json attrs = nlohmann::json::object();
    std::string const name = op.op_name();
    if (name == "TILE_ADD_INPLACE")
    {
        auto const *add =
            dynamic_cast<tile::TileAddInplaceOp const *>(&op);
        if (add == nullptr)
        {
            throw std::runtime_error("UnknownOp: TILE_ADD_INPLACE");
        }
        attrs["alpha"] = add->alpha;
        attrs["beta"] = add->beta;
        return attrs;
    }
    if (name == "TILE_ADD")
    {
        auto const *add = dynamic_cast<tile::TileAddOp const *>(&op);
        if (add == nullptr)
        {
            throw std::runtime_error("UnknownOp: TILE_ADD");
        }
        attrs["alpha"] = add->alpha;
        attrs["beta"] = add->beta;
        return attrs;
    }
    if (name == "TILE_MULTIPLY")
    {
        auto const *mul =
            dynamic_cast<tile::TileMultiplyOp const *>(&op);
        if (mul == nullptr)
        {
            throw std::runtime_error("UnknownOp: TILE_MULTIPLY");
        }
        attrs["alpha"] = mul->alpha;
        return attrs;
    }
    if (name == "TILE_GEMM")
    {
        auto const *gemm = dynamic_cast<tile::TileGemmOp const *>(&op);
        if (gemm == nullptr)
        {
            throw std::runtime_error("UnknownOp: TILE_GEMM");
        }
        attrs["alpha"] = gemm->alpha;
        attrs["beta"] = gemm->beta;
        attrs["trans_a"] = gemm->trans_a;
        attrs["trans_b"] = gemm->trans_b;
        attrs["ndim"] = gemm->ndim;
        attrs["batch_ndim"] = gemm->batch_ndim;
        return attrs;
    }
    if (name == "TILE_ADD_SLICE")
    {
        auto const *slice =
            dynamic_cast<tile::TileAddSliceOp const *>(&op);
        if (slice == nullptr)
        {
            throw std::runtime_error("UnknownOp: TILE_ADD_SLICE");
        }
        attrs["alpha"] = slice->alpha;
        attrs["beta"] = slice->beta;
        attrs["axis"] = slice->axis;
        return attrs;
    }
    if (name == "TILE_FILL")
    {
        auto const *fill = dynamic_cast<tile::TileFillOp const *>(&op);
        if (fill == nullptr)
        {
            throw std::runtime_error("UnknownOp: TILE_FILL");
        }
        attrs["value"] = fill->val;
        return attrs;
    }
    if (name == "TILE_COPY_INTERSECTION")
    {
        auto const *copy =
            dynamic_cast<tile::TileCopyIntersectionOp const *>(&op);
        if (copy == nullptr)
        {
            throw std::runtime_error(
                "UnknownOp: TILE_COPY_INTERSECTION");
        }
        attrs["src_offset"] = copy->src_offset;
        attrs["dst_offset"] = copy->dst_offset;
        return attrs;
    }
    if (name == "TILE_RELU" || name == "TILE_COPY" ||
        name == "TILE_UNREGISTER")
    {
        return attrs;
    }
#ifdef NNTILE_TORCH_NATIVE_OPS
    if (name == "TILE_TORCH_UNARY")
    {
        auto const *u =
            dynamic_cast<tile::TileTorchUnaryOp const *>(&op);
        if (u == nullptr)
        {
            throw std::runtime_error("UnknownOp: TILE_TORCH_UNARY");
        }
        return encode_torch_extra(u->kind, u->extra);
    }
    if (name == "TILE_TORCH_BINARY")
    {
        auto const *b =
            dynamic_cast<tile::TileTorchBinaryOp const *>(&op);
        if (b == nullptr)
        {
            throw std::runtime_error("UnknownOp: TILE_TORCH_BINARY");
        }
        return encode_torch_extra(b->kind, b->extra);
    }
    if (name == "TILE_TORCH_TERNARY")
    {
        auto const *t =
            dynamic_cast<tile::TileTorchTernaryOp const *>(&op);
        if (t == nullptr)
        {
            throw std::runtime_error("UnknownOp: TILE_TORCH_TERNARY");
        }
        return encode_torch_extra(t->kind, t->extra);
    }
#endif
    throw std::runtime_error("UnknownOp: " + name);
}

nlohmann::json encode_graph(
    TileGraph const &graph,
    std::unordered_map<TileGraph::NodeId, std::vector<float>> const &binds)
{
    nlohmann::json nodes = nlohmann::json::array();
    for (auto const &node : graph.tile_nodes())
    {
        if (!node)
        {
            continue;
        }
        nlohmann::json j = {
            {"id", node->id()},
            {"name", node->name()},
            {"dtype", dtype_to_string(node->dtype())},
            {"shape", node->shape()},
        };
        auto it = binds.find(node->id());
        if (it != binds.end() && node->dtype() == DataType::FP32)
        {
            j["data"] = it->second;
        }
        nodes.push_back(std::move(j));
    }
    nlohmann::json ops = nlohmann::json::array();
    for (auto const &op : graph.ops())
    {
        if (!op)
        {
            continue;
        }
        nlohmann::json ins = nlohmann::json::array();
        for (auto const *in : op->inputs())
        {
            ins.push_back(in->id());
        }
        nlohmann::json outs = nlohmann::json::array();
        for (auto const *out : op->outputs())
        {
            outs.push_back(out->id());
        }
        ops.push_back(
            {
                {"op_name", op->op_name()},
                {"inputs", std::move(ins)},
                {"outputs", std::move(outs)},
                {"attrs", encode_attrs(*op)},
            });
    }
    return {
        {"name", graph.name()},
        {"nodes", std::move(nodes)},
        {"ops", std::move(ops)},
    };
}

TileGraph::TileNode *node_by_id(
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

bool has_node(TileGraph const &graph, TileGraph::NodeId id)
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

void apply_op(TileGraph &graph, nlohmann::json const &op)
{
    std::string const name = op.at("op_name").get<std::string>();
    auto inputs = op.value("inputs", nlohmann::json::array());
    auto outputs = op.value("outputs", nlohmann::json::array());
    auto attrs = op.value("attrs", nlohmann::json::object());
    if (name == "TILE_ADD_INPLACE")
    {
        require_min(inputs, 2, "TILE_ADD_INPLACE inputs");
        auto *x = node_by_id(
            graph, inputs.at(0).get<TileGraph::NodeId>());
        auto *y = node_by_id(
            graph, inputs.at(1).get<TileGraph::NodeId>());
        tile::add_inplace(
            attrs.value("alpha", 1.0f),
            x,
            attrs.value("beta", 1.0f),
            y);
        return;
    }
    if (name == "TILE_ADD")
    {
        require_min(inputs, 2, "TILE_ADD inputs");
        require_min(outputs, 1, "TILE_ADD outputs");
        auto *x = node_by_id(
            graph, inputs.at(0).get<TileGraph::NodeId>());
        auto *y = node_by_id(
            graph, inputs.at(1).get<TileGraph::NodeId>());
        auto *z = node_by_id(
            graph, outputs.at(0).get<TileGraph::NodeId>());
        tile::add(
            attrs.value("alpha", 1.0f),
            x,
            attrs.value("beta", 1.0f),
            y,
            z);
        return;
    }
    if (name == "TILE_MULTIPLY")
    {
        require_min(inputs, 2, "TILE_MULTIPLY inputs");
        require_min(outputs, 1, "TILE_MULTIPLY outputs");
        auto *x = node_by_id(
            graph, inputs.at(0).get<TileGraph::NodeId>());
        auto *y = node_by_id(
            graph, inputs.at(1).get<TileGraph::NodeId>());
        auto *z = node_by_id(
            graph, outputs.at(0).get<TileGraph::NodeId>());
        tile::multiply(attrs.value("alpha", 1.0f), x, y, z);
        return;
    }
    if (name == "TILE_GEMM")
    {
        require_min(inputs, 2, "TILE_GEMM inputs");
        TileGraph::NodeId c_id;
        if (inputs.size() >= 3)
        {
            c_id = inputs.at(2).get<TileGraph::NodeId>();
        }
        else
        {
            require_min(outputs, 1, "TILE_GEMM outputs");
            c_id = outputs.at(0).get<TileGraph::NodeId>();
        }
        auto *a = node_by_id(
            graph, inputs.at(0).get<TileGraph::NodeId>());
        auto *b = node_by_id(
            graph, inputs.at(1).get<TileGraph::NodeId>());
        auto *c = node_by_id(graph, c_id);
        tile::gemm(
            a,
            b,
            c,
            attrs.value("alpha", 1.0f),
            attrs.value("beta", 0.0f),
            attrs.value("trans_a", false),
            attrs.value("trans_b", false),
            static_cast<Index>(attrs.value("ndim", 1)),
            static_cast<Index>(attrs.value("batch_ndim", 0)));
        return;
    }
    if (name == "TILE_ADD_SLICE")
    {
        require_min(inputs, 2, "TILE_ADD_SLICE inputs");
        require_min(outputs, 1, "TILE_ADD_SLICE outputs");
        auto *s1 = node_by_id(
            graph, inputs.at(0).get<TileGraph::NodeId>());
        auto *s2 = node_by_id(
            graph, inputs.at(1).get<TileGraph::NodeId>());
        auto *dst = node_by_id(
            graph, outputs.at(0).get<TileGraph::NodeId>());
        tile::add_slice(
            attrs.value("alpha", 1.0f),
            s1,
            attrs.value("beta", 1.0f),
            s2,
            dst,
            static_cast<Index>(attrs.value("axis", 0)));
        return;
    }
    if (name == "TILE_FILL")
    {
        require_min(outputs, 1, "TILE_FILL outputs");
        auto *x = node_by_id(
            graph, outputs.at(0).get<TileGraph::NodeId>());
        tile::fill(attrs.value("value", 0.0f), x);
        return;
    }
    if (name == "TILE_RELU")
    {
        require_min(inputs, 1, "TILE_RELU arity");
        require_min(outputs, 1, "TILE_RELU arity");
        auto *src = node_by_id(
            graph, inputs.at(0).get<TileGraph::NodeId>());
        auto *dst = node_by_id(
            graph, outputs.at(0).get<TileGraph::NodeId>());
        tile::relu(src, dst);
        return;
    }
    if (name == "TILE_COPY")
    {
        require_min(inputs, 1, "TILE_COPY inputs");
        require_min(outputs, 1, "TILE_COPY outputs");
        auto *src = node_by_id(
            graph, inputs.at(0).get<TileGraph::NodeId>());
        auto *dst = node_by_id(
            graph, outputs.at(0).get<TileGraph::NodeId>());
        tile::copy(src, dst);
        return;
    }
    if (name == "TILE_COPY_INTERSECTION")
    {
        require_min(inputs, 3, "TILE_COPY_INTERSECTION inputs");
        auto *src = node_by_id(
            graph, inputs.at(0).get<TileGraph::NodeId>());
        auto *dst = node_by_id(
            graph, inputs.at(1).get<TileGraph::NodeId>());
        auto *scratch = node_by_id(
            graph, inputs.at(2).get<TileGraph::NodeId>());
        auto src_off =
            attrs.value("src_offset", std::vector<Index>{});
        auto dst_off =
            attrs.value("dst_offset", std::vector<Index>{});
        tile::copy_intersection(
            src, src_off, dst, dst_off, scratch);
        return;
    }
    if (name == "TILE_UNREGISTER")
    {
        require_min(inputs, 1, "TILE_UNREGISTER inputs");
        tile::unregister(
            node_by_id(
                graph, inputs.at(0).get<TileGraph::NodeId>()));
        return;
    }
#ifdef NNTILE_TORCH_NATIVE_OPS
    if (name == "TILE_TORCH_UNARY")
    {
        require_min(inputs, 1, "TILE_TORCH_UNARY inputs");
        require_min(outputs, 1, "TILE_TORCH_UNARY outputs");
        auto extra = decode_torch_extra(attrs);
        extra.kind = static_cast<starpu::TorchKind>(
            attrs.value("kind", 0));
        tile::torch_unary(
            extra.kind,
            node_by_id(
                graph, inputs.at(0).get<TileGraph::NodeId>()),
            node_by_id(
                graph, outputs.at(0).get<TileGraph::NodeId>()),
            extra);
        return;
    }
    if (name == "TILE_TORCH_BINARY")
    {
        require_min(inputs, 2, "TILE_TORCH_BINARY inputs");
        require_min(outputs, 1, "TILE_TORCH_BINARY outputs");
        auto extra = decode_torch_extra(attrs);
        extra.kind = static_cast<starpu::TorchKind>(
            attrs.value("kind", 0));
        tile::torch_binary(
            extra.kind,
            node_by_id(
                graph, inputs.at(0).get<TileGraph::NodeId>()),
            node_by_id(
                graph, inputs.at(1).get<TileGraph::NodeId>()),
            node_by_id(
                graph, outputs.at(0).get<TileGraph::NodeId>()),
            extra);
        return;
    }
    if (name == "TILE_TORCH_TERNARY")
    {
        require_min(inputs, 3, "TILE_TORCH_TERNARY inputs");
        require_min(outputs, 1, "TILE_TORCH_TERNARY outputs");
        auto extra = decode_torch_extra(attrs);
        extra.kind = static_cast<starpu::TorchKind>(
            attrs.value("kind", 0));
        tile::torch_ternary(
            extra.kind,
            node_by_id(
                graph, inputs.at(0).get<TileGraph::NodeId>()),
            node_by_id(
                graph, inputs.at(1).get<TileGraph::NodeId>()),
            node_by_id(
                graph, inputs.at(2).get<TileGraph::NodeId>()),
            node_by_id(
                graph, outputs.at(0).get<TileGraph::NodeId>()),
            extra);
        return;
    }
#endif
    throw std::runtime_error("UnknownOp: " + name);
}

void ingest_nodes(TileGraph &graph, nlohmann::json const &nodes)
{
    for (auto const &jn : nodes)
    {
        auto const want = jn.at("id").get<TileGraph::NodeId>();
        if (has_node(graph, want))
        {
            continue;
        }
        std::vector<Index> shape =
            jn.at("shape").get<std::vector<Index>>();
        auto *node = graph.data(
            std::move(shape),
            jn.value("name", std::string()),
            dtype_from_string(
                jn.value("dtype", std::string("FP32"))));
        if (node->id() != want)
        {
            throw std::runtime_error(
                "TileGraph node ids must be dense from 0");
        }
    }
}

void ingest_ops(TileGraph &graph, nlohmann::json const &ops)
{
    size_t const have = graph.num_ops();
    if (ops.size() < have)
    {
        throw std::runtime_error(
            "Submit graph shrank; remote session is append-only");
    }
    for (size_t i = have; i < ops.size(); ++i)
    {
        apply_op(graph, ops.at(i));
    }
}

void bind_node_data(
    RuntimeExecutionDriver &driver,
    TileGraph &graph,
    nlohmann::json const &nodes)
{
    for (auto const &jn : nodes)
    {
        if (!jn.contains("data"))
        {
            continue;
        }
        auto *node = node_by_id(
            graph, jn.at("id").get<TileGraph::NodeId>());
        if (node->dtype() != DataType::FP32)
        {
            continue;
        }
        auto data = jn["data"].get<std::vector<float>>();
        driver.runtime().bind_data(node, data);
    }
}

int connect_unix(std::string const &path)
{
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
    {
        throw_errno("socket");
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path))
    {
        ::close(fd);
        throw std::runtime_error("NNTILE_DRIVER_SOCKET path too long");
    }
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (::connect(
            fd,
            reinterpret_cast<sockaddr *>(&addr),
            sizeof(addr)) != 0)
    {
        int const err = errno;
        ::close(fd);
        errno = err;
        throw_errno("connect");
    }
    return fd;
}

void apply_socket_acl(std::string const &path)
{
    if (::chmod(path.c_str(), S_IRUSR | S_IWUSR) != 0)
    {
        throw_errno("chmod");
    }
    struct group *gr = ::getgrnam(driver_socket_group_name());
    if (gr == nullptr)
    {
        return;
    }
    if (::chown(
            path.c_str(),
            static_cast<uid_t>(-1),
            gr->gr_gid) != 0)
    {
        if (errno == EPERM || errno == EACCES)
        {
            return;
        }
        throw_errno("chown");
    }
}

int listen_unix(std::string const &path)
{
    ::unlink(path.c_str());
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
    {
        throw_errno("socket");
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path))
    {
        ::close(fd);
        throw std::runtime_error("NNTILE_DRIVER_SOCKET path too long");
    }
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(
            fd,
            reinterpret_cast<sockaddr *>(&addr),
            sizeof(addr)) != 0)
    {
        int const err = errno;
        ::close(fd);
        errno = err;
        throw_errno("bind");
    }
    try
    {
        apply_socket_acl(path);
    }
    catch (...)
    {
        ::close(fd);
        ::unlink(path.c_str());
        throw;
    }
    if (::listen(fd, 1) != 0)
    {
        int const err = errno;
        ::close(fd);
        ::unlink(path.c_str());
        errno = err;
        throw_errno("listen");
    }
    return fd;
}

void handle_client(int fd)
{
    auto hello = recv_json(fd);
    if (hello.value("type", "") != "Handshake")
    {
        send_json(
            fd,
            {
                {"type", "Error"},
                {"code", "Protocol"},
                {"message", "expected Handshake"},
            });
        return;
    }
    if (hello.value("protocol_version", 0) != kDriverProtocol)
    {
        send_json(
            fd,
            {
                {"type", "Error"},
                {"code", "Protocol"},
                {"message", "protocol_version mismatch"},
            });
        return;
    }
    send_json(fd, {{"type", "HandshakeOk"}});

    std::unique_ptr<TileGraph> graph;
    std::unique_ptr<RuntimeExecutionDriver> driver;
    std::unordered_map<TileGraph::NodeId, std::vector<float>> queued_bind;
    auto flush_queued_binds = [&]()
    {
        if (!driver || !graph)
        {
            return;
        }
        for (auto it = queued_bind.begin(); it != queued_bind.end();)
        {
            if (!has_node(*graph, it->first))
            {
                ++it;
                continue;
            }
            driver->runtime().bind_data(
                node_by_id(*graph, it->first), it->second);
            it = queued_bind.erase(it);
        }
    };
    try
    {
        while (true)
        {
            auto msg = recv_json(fd);
            std::string const type = msg.value("type", "");
            if (type == "Submit")
            {
                auto const &wire = msg.at("graph");
                if (!graph)
                {
                    graph = std::make_unique<TileGraph>(
                        wire.value("name", std::string("remote")));
                }
                ingest_nodes(*graph, wire.at("nodes"));
                ingest_ops(*graph, wire.at("ops"));
                if (!driver)
                {
                    driver = std::make_unique<RuntimeExecutionDriver>(
                        *graph);
                }
                driver->runtime().compile();
                bind_node_data(*driver, *graph, wire.at("nodes"));
                flush_queued_binds();
                driver->submit(*graph);
                send_json(fd, {{"type", "SubmitOk"}});
            }
            else if (type == "Wait")
            {
                if (driver)
                {
                    driver->wait();
                }
                send_json(fd, {{"type", "WaitOk"}});
            }
            else if (type == "Bind")
            {
                auto const id =
                    msg.at("node_id").get<TileGraph::NodeId>();
                auto data = msg.at("data").get<std::vector<float>>();
                if (driver && graph && has_node(*graph, id))
                {
                    driver->runtime().bind_data(
                        node_by_id(*graph, id), data);
                }
                else
                {
                    queued_bind[id] = std::move(data);
                }
                send_json(fd, {{"type", "BindOk"}});
            }
            else if (type == "Gather")
            {
                if (!driver || !graph)
                {
                    throw std::runtime_error("gather before submit");
                }
                auto *node = node_by_id(
                    *graph,
                    msg.at("node_id").get<TileGraph::NodeId>());
                auto data = driver->runtime().get_output<float>(node);
                send_json(
                    fd,
                    {{"type", "GatherOk"}, {"data", data}});
            }
            else
            {
                send_json(
                    fd,
                    {
                        {"type", "Error"},
                        {"code", "Protocol"},
                        {"message", "unknown type " + type},
                    });
            }
        }
    }
    catch (std::exception const &ex)
    {
        std::string const what = ex.what();
        if (what.find("socket closed") != std::string::npos)
        {
            return;
        }
        try
        {
            std::string code = "Internal";
            if (what.rfind("UnknownOp", 0) == 0)
            {
                code = "UnknownOp";
            }
            send_json(
                fd,
                {
                    {"type", "Error"},
                    {"code", code},
                    {"message", what},
                });
        }
        catch (...)
        {
        }
    }
}

} // namespace

std::string default_driver_socket_path()
{
    char const *env = std::getenv("NNTILE_DRIVER_SOCKET");
    if (env != nullptr && env[0] != '\0')
    {
        return env;
    }
    return "/tmp/nntile-driver.sock";
}

char const *driver_socket_group_name()
{
    return "nntile-ops";
}

RemoteExecutionDriver::RemoteExecutionDriver(std::string socket_path)
    : path_(
          socket_path.empty() ? default_driver_socket_path()
                              : std::move(socket_path))
{
    std::exception_ptr last;
    for (int i = 0; i < 50; ++i)
    {
        try
        {
            fd_ = connect_unix(path_);
            last = nullptr;
            break;
        }
        catch (...)
        {
            last = std::current_exception();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    if (fd_ < 0)
    {
        if (last)
        {
            std::rethrow_exception(last);
        }
        throw std::runtime_error("RemoteExecutionDriver: connect failed");
    }
    send_json(
        fd_,
        {
            {"type", "Handshake"},
            {"protocol_version", kDriverProtocol},
        });
    auto const ack = require_ok(recv_json(fd_), "handshake");
    if (ack.value("type", "") != "HandshakeOk")
    {
        throw std::runtime_error(
            "RemoteExecutionDriver handshake: expected HandshakeOk");
    }
}

RemoteExecutionDriver::~RemoteExecutionDriver()
{
    if (fd_ >= 0)
    {
        ::close(fd_);
        fd_ = -1;
    }
}

void RemoteExecutionDriver::submit(TileGraph const &graph)
{
    nlohmann::json msg = {
        {"type", "Submit"},
        {"graph", encode_graph(graph, pending_bind_)},
    };
    send_json(fd_, msg);
    require_ok(recv_json(fd_), "submit");
    pending_bind_.clear();
    submitted_ = true;
}

void RemoteExecutionDriver::wait()
{
    send_json(fd_, {{"type", "Wait"}});
    require_ok(recv_json(fd_), "wait");
}

void RemoteExecutionDriver::bind(
    TileGraph::NodeId id, std::vector<float> const &data)
{
    if (!submitted_)
    {
        pending_bind_[id] = data;
        return;
    }
    send_json(
        fd_,
        {{"type", "Bind"}, {"node_id", id}, {"data", data}});
    require_ok(recv_json(fd_), "bind");
}

std::vector<float> RemoteExecutionDriver::gather(TileGraph::NodeId id)
{
    send_json(
        fd_,
        {{"type", "Gather"}, {"node_id", id}});
    auto reply = require_ok(recv_json(fd_), "gather");
    return reply.at("data").get<std::vector<float>>();
}

ExecutionDaemon::ExecutionDaemon(std::string socket_path)
    : path_(
          socket_path.empty() ? default_driver_socket_path()
                              : std::move(socket_path))
{
}

ExecutionDaemon::~ExecutionDaemon()
{
    stop();
}

void ExecutionDaemon::start()
{
    if (thread_.joinable())
    {
        throw std::runtime_error("ExecutionDaemon already started");
    }
    // Production ``executiond`` has no ContextFixture. Skip if this
    // process already inited StarPU (C++ tests).
    if (!starpu_is_initialized())
    {
        ctx_ = std::make_unique<Context>(-1, -1, 0);
    }
    stop_ = false;
    listen_fd_ = listen_unix(path_);
    thread_ = std::thread([this]() { run(); });
}

void ExecutionDaemon::stop()
{
    stop_ = true;
    if (listen_fd_ >= 0)
    {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    if (thread_.joinable())
    {
        thread_.join();
    }
    ::unlink(path_.c_str());
    ctx_.reset();
}

void ExecutionDaemon::run()
{
    while (!stop_)
    {
        pollfd pfd{};
        pfd.fd = listen_fd_;
        pfd.events = POLLIN;
        int const rc = ::poll(&pfd, 1, 100);
        if (rc < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            break;
        }
        if (rc == 0 || (pfd.revents & POLLIN) == 0)
        {
            continue;
        }
        int client = ::accept(listen_fd_, nullptr, nullptr);
        if (client < 0)
        {
            if (errno == EINTR || errno == EAGAIN || errno == EBADF)
            {
                continue;
            }
            break;
        }
        handle_client(client);
        ::close(client);
    }
}

} // namespace nntile
