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
#include <nntile/remote_tile_codec.hh>

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
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

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
    if (name == "BOOL" || name == "bool")
    {
        return DataType::BOOL;
    }
    throw std::runtime_error("UnknownOp: dtype " + name);
}

nlohmann::json encode_graph(
    TileGraph const &graph,
    std::unordered_map<TileGraph::NodeId, std::vector<float>> const
        &binds,
    std::unordered_map<TileGraph::NodeId, std::vector<std::int64_t>> const
        &binds_i64,
    std::unordered_map<TileGraph::NodeId, std::vector<std::uint8_t>> const
        &binds_bool)
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
        if (node->dtype() == DataType::FP32)
        {
            auto it = binds.find(node->id());
            if (it != binds.end())
            {
                j["data"] = it->second;
            }
        }
        else if (node->dtype() == DataType::INT64)
        {
            auto it = binds_i64.find(node->id());
            if (it != binds_i64.end())
            {
                j["data"] = it->second;
            }
        }
        else if (node->dtype() == DataType::BOOL)
        {
            auto it = binds_bool.find(node->id());
            if (it != binds_bool.end())
            {
                j["data"] = it->second;
            }
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
                {"attrs", encode_tile_op_attrs(*op)},
            });
    }
    return {
        {"name", graph.name()},
        {"nodes", std::move(nodes)},
        {"ops", std::move(ops)},
    };
}

void bind_host_data(
    RuntimeExecutionDriver &driver,
    TileGraph::TileNode *node,
    DataType dtype,
    nlohmann::json const &data)
{
    if (dtype == DataType::FP32)
    {
        driver.runtime().bind_data(
            node, data.get<std::vector<float>>());
        return;
    }
    if (dtype == DataType::INT64)
    {
        driver.runtime().bind_data(
            node, data.get<std::vector<std::int64_t>>());
        return;
    }
    if (dtype == DataType::BOOL)
    {
        auto bytes = data.get<std::vector<std::uint8_t>>();
        std::unique_ptr<bool[]> buf(new bool[bytes.size()]);
        for (size_t i = 0; i < bytes.size(); ++i)
        {
            buf[i] = bytes[i] != 0;
        }
        driver.runtime().bind_data(
            node, buf.get(), bytes.size());
        return;
    }
    throw std::runtime_error(
        "UnknownOp: dtype " + dtype_to_string(dtype));
}

void ingest_nodes(TileGraph &graph, nlohmann::json const &nodes)
{
    for (auto const &jn : nodes)
    {
        auto const want = jn.at("id").get<TileGraph::NodeId>();
        if (tile_graph_has_node(graph, want))
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
        apply_tile_op(graph, ops.at(i));
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
        auto *node = tile_node_by_id(
            graph, jn.at("id").get<TileGraph::NodeId>());
        bind_host_data(driver, node, node->dtype(), jn["data"]);
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
    struct QueuedBind
    {
        DataType dtype = DataType::FP32;
        nlohmann::json data;
    };
    std::unordered_map<TileGraph::NodeId, QueuedBind> queued_bind;
    auto flush_queued_binds = [&]()
    {
        if (!driver || !graph)
        {
            return;
        }
        for (auto it = queued_bind.begin(); it != queued_bind.end();)
        {
            if (!tile_graph_has_node(*graph, it->first))
            {
                ++it;
                continue;
            }
            bind_host_data(
                *driver,
                tile_node_by_id(*graph, it->first),
                it->second.dtype,
                it->second.data);
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
                DataType const dtype = dtype_from_string(
                    msg.value("dtype", std::string("FP32")));
                if (driver && graph &&
                    tile_graph_has_node(*graph, id))
                {
                    bind_host_data(
                        *driver,
                        tile_node_by_id(*graph, id),
                        dtype,
                        msg.at("data"));
                }
                else
                {
                    queued_bind[id] = QueuedBind{
                        dtype, msg.at("data")};
                }
                send_json(fd, {{"type", "BindOk"}});
            }
            else if (type == "Gather")
            {
                if (!driver || !graph)
                {
                    throw std::runtime_error("gather before submit");
                }
                auto *node = tile_node_by_id(
                    *graph,
                    msg.at("node_id").get<TileGraph::NodeId>());
                nlohmann::json data;
                if (node->dtype() == DataType::INT64)
                {
                    data = driver->runtime().get_output<std::int64_t>(
                        node);
                }
                else if (node->dtype() == DataType::BOOL)
                {
                    auto raw =
                        driver->runtime().get_output<bool>(node);
                    std::vector<std::uint8_t> bytes(raw.size());
                    for (size_t i = 0; i < raw.size(); ++i)
                    {
                        bytes[i] = raw[i] ? 1 : 0;
                    }
                    data = bytes;
                }
                else
                {
                    data = driver->runtime().get_output<float>(node);
                }
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
        {"graph",
         encode_graph(
             graph,
             pending_bind_,
             pending_bind_i64_,
             pending_bind_bool_)},
    };
    send_json(fd_, msg);
    require_ok(recv_json(fd_), "submit");
    pending_bind_.clear();
    pending_bind_i64_.clear();
    pending_bind_bool_.clear();
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
        {
            {"type", "Bind"},
            {"node_id", id},
            {"dtype", "FP32"},
            {"data", data},
        });
    require_ok(recv_json(fd_), "bind");
}

void RemoteExecutionDriver::bind_int64(
    TileGraph::NodeId id, std::vector<std::int64_t> const &data)
{
    if (!submitted_)
    {
        pending_bind_i64_[id] = data;
        return;
    }
    send_json(
        fd_,
        {
            {"type", "Bind"},
            {"node_id", id},
            {"dtype", "INT64"},
            {"data", data},
        });
    require_ok(recv_json(fd_), "bind");
}

void RemoteExecutionDriver::bind_bool(
    TileGraph::NodeId id, std::vector<std::uint8_t> const &data)
{
    if (!submitted_)
    {
        pending_bind_bool_[id] = data;
        return;
    }
    send_json(
        fd_,
        {
            {"type", "Bind"},
            {"node_id", id},
            {"dtype", "BOOL"},
            {"data", data},
        });
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

std::vector<std::int64_t> RemoteExecutionDriver::gather_int64(
    TileGraph::NodeId id)
{
    send_json(
        fd_,
        {{"type", "Gather"}, {"node_id", id}});
    auto reply = require_ok(recv_json(fd_), "gather");
    return reply.at("data").get<std::vector<std::int64_t>>();
}

std::vector<std::uint8_t> RemoteExecutionDriver::gather_bool(
    TileGraph::NodeId id)
{
    send_json(
        fd_,
        {{"type", "Gather"}, {"node_id", id}});
    auto reply = require_ok(recv_json(fd_), "gather");
    return reply.at("data").get<std::vector<std::uint8_t>>();
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
