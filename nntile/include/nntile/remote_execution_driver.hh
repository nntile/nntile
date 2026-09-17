/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/remote_execution_driver.hh
 * RemoteExecutionDriver: submit an already-lowered TileGraph over
 * ``NNTILE_DRIVER_SOCKET``. Does not compile TensorGraph → TileGraph.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/execution_driver.hh>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace nntile
{

class Context;

//! Socket path: ``NNTILE_DRIVER_SOCKET`` or ``/tmp/nntile-driver.sock``.
std::string default_driver_socket_path();

//! Unix-socket group (``nntile-ops``). Mode stays 0600.
char const *driver_socket_group_name();

//! Client of ``nntile-executiond``. bind() before submit() is queued
//! into the Submit payload; bind() after submit() is a Bind message.
//! StarPU stays in the daemon process.
class RemoteExecutionDriver : public ExecutionDriver
{
  public:
    explicit RemoteExecutionDriver(std::string socket_path = {});

    ~RemoteExecutionDriver() override;

    RemoteExecutionDriver(RemoteExecutionDriver const &) = delete;
    RemoteExecutionDriver &operator=(
        RemoteExecutionDriver const &) = delete;

    void submit(TileGraph const &graph) override;

    void wait() override;

    void bind(
        TileGraph::NodeId id, std::vector<float> const &data);

    void bind_int64(
        TileGraph::NodeId id, std::vector<std::int64_t> const &data);

    void bind_bool(
        TileGraph::NodeId id, std::vector<std::uint8_t> const &data);

    std::vector<float> gather(TileGraph::NodeId id);

    std::vector<std::int64_t> gather_int64(TileGraph::NodeId id);

    std::vector<std::uint8_t> gather_bool(TileGraph::NodeId id);

  private:
    std::string path_;
    int fd_ = -1;
    bool submitted_ = false;
    std::unordered_map<TileGraph::NodeId, std::vector<float>>
        pending_bind_;
    std::unordered_map<TileGraph::NodeId, std::vector<std::int64_t>>
        pending_bind_i64_;
    std::unordered_map<TileGraph::NodeId, std::vector<std::uint8_t>>
        pending_bind_bool_;
};

//! Listen on a Unix socket (mode 0600, group ``nntile-ops`` when
//! present) and keep one RuntimeExecutionDriver for the connection.
class ExecutionDaemon
{
  public:
    explicit ExecutionDaemon(std::string socket_path = {});

    ~ExecutionDaemon();

    ExecutionDaemon(ExecutionDaemon const &) = delete;
    ExecutionDaemon &operator=(ExecutionDaemon const &) = delete;

    void start();

    void stop();

    std::string const &path() const
    {
        return path_;
    }

  private:
    void run();

    std::string path_;
    int listen_fd_ = -1;
    std::atomic<bool> stop_{false};
    std::thread thread_;
    //! Owned only when this process had no Context yet (production
    //! ``executiond``). C++ tests use ContextFixture and leave this null.
    std::unique_ptr<Context> ctx_;
};

} // namespace nntile
