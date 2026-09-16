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

#include <nntile/dtype.hh>
#include <nntile/tile/ops/add_inplace.hh>
#include <nntile/tile/ops/fill.hh>
#include <nntile/tile/ops/relu.hh>

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
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
    throw std::runtime_error("UnknownOp: dtype " + name);
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
        if (it != binds.end())
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
        nlohmann::json attrs = nlohmann::json::object();
        std::string const name = op->op_name();
        if (name == "TILE_ADD_INPLACE")
        {
            auto const *add =
                dynamic_cast<tile::TileAddInplaceOp const *>(
                    op.get());
            if (add == nullptr)
            {
                throw std::runtime_error("UnknownOp: TILE_ADD_INPLACE");
            }
            attrs["alpha"] = add->alpha;
            attrs["beta"] = add->beta;
        }
        else if (name == "TILE_FILL")
        {
            auto const *fill =
                dynamic_cast<tile::TileFillOp const *>(op.get());
            if (fill == nullptr)
            {
                throw std::runtime_error("UnknownOp: TILE_FILL");
            }
            attrs["value"] = fill->val;
        }
        else if (name == "TILE_RELU")
        {
            // no attrs
        }
        else
        {
            throw std::runtime_error("UnknownOp: " + name);
        }
        ops.push_back(
            {
                {"op_name", name},
                {"inputs", std::move(ins)},
                {"outputs", std::move(outs)},
                {"attrs", std::move(attrs)},
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

void apply_op(TileGraph &graph, nlohmann::json const &op)
{
    std::string const name = op.at("op_name").get<std::string>();
    auto inputs = op.value("inputs", nlohmann::json::array());
    auto outputs = op.value("outputs", nlohmann::json::array());
    auto attrs = op.value("attrs", nlohmann::json::object());
    if (name == "TILE_ADD_INPLACE")
    {
        if (inputs.size() < 2)
        {
            throw std::runtime_error("UnknownOp: TILE_ADD_INPLACE inputs");
        }
        auto *x = node_by_id(graph, inputs.at(0).get<TileGraph::NodeId>());
        auto *y = node_by_id(graph, inputs.at(1).get<TileGraph::NodeId>());
        tile::add_inplace(
            attrs.value("alpha", 1.0f),
            x,
            attrs.value("beta", 1.0f),
            y);
        return;
    }
    if (name == "TILE_FILL")
    {
        if (outputs.empty())
        {
            throw std::runtime_error("UnknownOp: TILE_FILL outputs");
        }
        auto *x = node_by_id(
            graph, outputs.at(0).get<TileGraph::NodeId>());
        tile::fill(attrs.value("value", 0.0f), x);
        return;
    }
    if (name == "TILE_RELU")
    {
        if (inputs.empty() || outputs.empty())
        {
            throw std::runtime_error("UnknownOp: TILE_RELU arity");
        }
        auto *src = node_by_id(
            graph, inputs.at(0).get<TileGraph::NodeId>());
        auto *dst = node_by_id(
            graph, outputs.at(0).get<TileGraph::NodeId>());
        tile::relu(src, dst);
        return;
    }
    throw std::runtime_error("UnknownOp: " + name);
}

std::unique_ptr<TileGraph> decode_graph(nlohmann::json const &wire)
{
    auto graph = std::make_unique<TileGraph>(
        wire.value("name", std::string("remote")));
    for (auto const &jn : wire.at("nodes"))
    {
        std::vector<Index> shape = jn.at("shape").get<std::vector<Index>>();
        auto *node = graph->data(
            std::move(shape),
            jn.value("name", std::string()),
            dtype_from_string(jn.value("dtype", std::string("FP32"))));
        if (jn.contains("id") &&
            node->id() != jn["id"].get<TileGraph::NodeId>())
        {
            throw std::runtime_error(
                "TileGraph node ids must be dense from 0");
        }
    }
    for (auto const &op : wire.at("ops"))
    {
        apply_op(*graph, op);
    }
    return graph;
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
    if (::chmod(path.c_str(), S_IRUSR | S_IWUSR) != 0)
    {
        int const err = errno;
        ::close(fd);
        ::unlink(path.c_str());
        errno = err;
        throw_errno("chmod");
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
    try
    {
        while (true)
        {
            auto msg = recv_json(fd);
            std::string const type = msg.value("type", "");
            if (type == "Submit")
            {
                graph = decode_graph(msg.at("graph"));
                driver =
                    std::make_unique<RuntimeExecutionDriver>(*graph);
                driver->runtime().compile();
                bind_node_data(*driver, *graph, msg["graph"]["nodes"]);
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
                if (!driver || !graph)
                {
                    throw std::runtime_error("bind before submit");
                }
                auto *node = node_by_id(
                    *graph,
                    msg.at("node_id").get<TileGraph::NodeId>());
                auto data = msg.at("data").get<std::vector<float>>();
                driver->runtime().bind_data(node, data);
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
