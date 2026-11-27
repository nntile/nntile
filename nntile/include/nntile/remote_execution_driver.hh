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
#include <sys/types.h>
#include <thread>
#include <unordered_map>
#include <vector>

namespace nntile
{

class Context;

//! Socket path: ``NNTILE_DRIVER_SOCKET`` or ``/tmp/nntile-driver.sock``.
std::string default_driver_socket_path();

//! Unix-socket group (``nntile-ops``). The node is 0600 unless the
//! group exists and the daemon may chown it, in which case the mode is
//! widened to 0660 for that group.
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

//! CUDA worker restriction applied by an ExecutionDaemon at start.
//! ``Auto`` consults ``NNTILE_DAEMON_RESTRICT_CUDA`` ("1" restricts,
//! anything else does not); ``None`` / ``Cuda`` force the choice.
enum class DaemonCudaRestrict
{
    Auto,
    None,
    Cuda,
};

//! Listen on a Unix socket (same-user only by default; mode 0600, or
//! 0660 for group ``nntile-ops`` when present) and keep one
//! RuntimeExecutionDriver for the connection. Other local users must
//! present ``NNTILE_DRIVER_TOKEN`` in the handshake.
class ExecutionDaemon
{
  public:
    explicit ExecutionDaemon(
        std::string socket_path = {},
        DaemonCudaRestrict restrict_cuda = DaemonCudaRestrict::Auto);

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
    DaemonCudaRestrict restrict_cuda_ = DaemonCudaRestrict::Auto;
    int listen_fd_ = -1;
    //! True when the socket node was actually chowned to the group
    //! (and therefore widened to 0660) at start().
    bool socket_group_grant_ = false;
    //! Group the node was granted to; meaningful only when
    //! socket_group_grant_ is true.
    gid_t socket_group_gid_ = 0;
    //! Self-pipe written by stop() so run() wakes up promptly on
    //! every platform (shutdown() of a listening Unix socket is a
    //! no-op on macOS).
    int wakeup_fd_[2] = {-1, -1};
    //! Client fd currently serviced, so stop() can interrupt a
    //! blocking read on it.
    std::atomic<int> client_fd_{-1};
    std::atomic<bool> stop_{false};
    std::thread thread_;
    //! Owned only when this process had no Context yet (production
    //! ``executiond``). C++ tests use ContextFixture and leave this null.
    std::unique_ptr<Context> ctx_;
};

} // namespace nntile
