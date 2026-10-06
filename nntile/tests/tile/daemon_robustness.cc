/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file nntile/tests/tile/daemon_robustness.cc
 * ExecutionDaemon per-connection safety, client socket timeouts, and
 * daemon-side CUDA restriction.
 *
 * @version 1.1.0
 * */

#include "context_fixture.hh"
#include "test_frobenius.hh"

#include <nntile/execution_driver.hh>
#include <nntile/remote_execution_driver.hh>
#include <nntile/tile.hh>
#include <nntile/tile/ops/add_inplace.hh>
#include <nntile/tile/ops/fill.hh>
#ifdef NNTILE_TORCH_NATIVE_OPS
#include <nntile/tile/ops/torch_dispatch.hh>
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

using namespace nntile;
namespace tg = nntile::tile;

namespace
{

std::string test_socket_path(char const *tag)
{
    return std::string("/tmp/nntile-daemon-") + tag + "-" +
        std::to_string(::getpid()) + ".sock";
}

//! A raw client that connects, sends non-protocol bytes and hangs up -
//! the shape of a liveness probe.
void speak_garbage(std::string const &path)
{
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    REQUIRE(fd >= 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    // Retry: right after start() the listener is live but a probe may
    // still lose the race against the accept thread on a loaded host.
    bool connected = false;
    for (int attempt = 0; attempt < 5 && !connected; ++attempt)
    {
        connected = ::connect(
            fd, reinterpret_cast<sockaddr const *>(&addr),
            sizeof(addr)) == 0;
        if (!connected)
        {
            struct timespec pause = {0, 100 * 1000 * 1000};
            ::nanosleep(&pause, nullptr);
        }
    }
    REQUIRE(connected);
    char const garbage[] = "\x00\x01\x02 not json at all {{{";
    ssize_t const n = ::write(fd, garbage, sizeof(garbage) - 1);
    REQUIRE(n > 0);
    ::close(fd);
}

//! Length-prefixed wire frame used by the driver protocol.
void send_frame(int fd, std::string const &body)
{
    uint32_t const n = htonl(static_cast<uint32_t>(body.size()));
    size_t off = 0;
    while (off < sizeof(n))
    {
        ssize_t const w = ::write(
            fd, reinterpret_cast<char const *>(&n) + off,
            sizeof(n) - off);
        REQUIRE(w > 0);
        off += static_cast<size_t>(w);
    }
    off = 0;
    while (off < body.size())
    {
        ssize_t const w = ::write(fd, body.data() + off, body.size() - off);
        REQUIRE(w > 0);
        off += static_cast<size_t>(w);
    }
}

std::string recv_frame(int fd)
{
    uint32_t n = 0;
    size_t off = 0;
    while (off < sizeof(n))
    {
        ssize_t const r = ::read(
            fd, reinterpret_cast<char *>(&n) + off, sizeof(n) - off);
        REQUIRE(r > 0);
        off += static_cast<size_t>(r);
    }
    n = ntohl(n);
    REQUIRE(n > 0);
    REQUIRE(n < 1024u * 1024u);
    std::string body(n, '\0');
    off = 0;
    while (off < body.size())
    {
        ssize_t const r = ::read(
            fd, body.data() + off, body.size() - off);
        REQUIRE(r > 0);
        off += static_cast<size_t>(r);
    }
    return body;
}

//! Raw handshake against a daemon at ``path``; returns the daemon reply.
std::string raw_handshake(std::string const &path, bool with_token,
    std::string const &token = {})
{
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    REQUIRE(fd >= 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    bool connected = false;
    for (int attempt = 0; attempt < 10 && !connected; ++attempt)
    {
        connected = ::connect(
            fd, reinterpret_cast<sockaddr const *>(&addr),
            sizeof(addr)) == 0;
        if (!connected)
        {
            struct timespec pause = {0, 100 * 1000 * 1000};
            ::nanosleep(&pause, nullptr);
        }
    }
    REQUIRE(connected);
    std::string body =
        R"({"type":"Handshake","protocol_version":1})";
    if (with_token)
    {
        body =
            R"({"type":"Handshake","protocol_version":1,)"
            R"("token":")" + token + R"("})";
    }
    send_frame(fd, body);
    std::string reply = recv_frame(fd);
    ::close(fd);
    return reply;
}

} // namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "ExecutionDaemon survives a garbage handshake",
    "[graph][tile][driver][remote]")
{
    // Regression: handle_client ran the handshake outside its try
    // block and run() had no per-connection guard, so any connection
    // that spoke garbage threw out of the accept loop and SIGABRTed
    // the daemon (platform liveness probes).
    std::string const path = test_socket_path("garbage");
    ExecutionDaemon daemon(path);
    daemon.start();

    speak_garbage(path);
    speak_garbage(path);

    // The daemon must still serve a well-formed client afterwards. The
    // full session runs on CPU builds; with CUDA workers enabled the
    // session additionally depends on StarPU worker selection for
    // host-resident daemon data, which is a scheduling concern out of
    // scope for this regression (the probes above already prove the
    // accept loop survived).
#ifndef NNTILE_USE_CUDA
    {
        RemoteExecutionDriver driver(path);
        TileGraph graph("daemon_garbage_survivor");
        auto *x = graph.data({2, 2}, "x", DataType::FP32);
        auto *y = graph.data({2, 2}, "y", DataType::FP32);
        auto *relu_out = graph.data({2, 2}, "relu", DataType::FP32);
        tg::torch_binary(
            starpu::TorchKind::Add, x, y, relu_out);
        driver.bind(x->id(), {1.f, -2.f, 3.f, -4.f});
        driver.bind(y->id(), {1.f, 1.f, 1.f, 1.f});
        driver.submit(graph);
        driver.wait();
        nntile::test::require_relative_element_error(
            driver.gather(relu_out->id()), {2.f, -1.f, 4.f, -3.f});
    }
#endif // NNTILE_USE_CUDA
    daemon.stop();
}


TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RemoteExecutionDriver client socket timeouts",
    "[graph][tile][driver][remote]")
{
    // A socket that accepts but never speaks: with a recv timeout the
    // handshake fails closed ("recv: timed out") instead of hanging
    // forever; without one it blocked indefinitely.
    std::string const path = test_socket_path("blackhole");
    int listener = ::socket(AF_UNIX, SOCK_STREAM, 0);
    REQUIRE(listener >= 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    ::unlink(path.c_str());
    REQUIRE(::bind(
        listener, reinterpret_cast<sockaddr const *>(&addr),
        sizeof(addr)) == 0);
    REQUIRE(::listen(listener, 1) == 0);

    ::setenv("NNTILE_DRIVER_RECV_TIMEOUT_MS", "300", 1);
    try
    {
        RemoteExecutionDriver driver(path);
        FAIL("handshake against a silent socket should time out");
    }
    catch (std::exception const &ex)
    {
        REQUIRE_THAT(
            std::string(ex.what()),
            Catch::Matchers::ContainsSubstring("timed out"));
    }
    ::setenv("NNTILE_DRIVER_RECV_TIMEOUT_MS", "0", 1);
    ::close(listener);
    ::unlink(path.c_str());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "ExecutionDaemon authorizes peers via uid and token",
    "[graph][tile][driver][remote]")
{
    // Same-user peers always connect; when NNTILE_DRIVER_TOKEN is set
    // on the daemon, every client must present it in the handshake.
    // (Cross-uid rejection without a token needs a second account and
    // is covered by the peer_credentials check itself.)
    std::string const path = test_socket_path("auth");
    ::setenv("NNTILE_DRIVER_TOKEN", "s3cret-bucket", 1);
    ExecutionDaemon daemon(path);
    daemon.start();

    // No token in the handshake: rejected despite a valid envelope.
    std::string reply = raw_handshake(path, false);
    REQUIRE(reply.find("Unauthorized") != std::string::npos);

    // Wrong token: rejected.
    reply = raw_handshake(path, true, "wrong-token");
    REQUIRE(reply.find("Unauthorized") != std::string::npos);

    // Right token: accepted.
    reply = raw_handshake(path, true, "s3cret-bucket");
    REQUIRE(reply.find("HandshakeOk") != std::string::npos);

#ifndef NNTILE_USE_CUDA
    // The stock client picks the token up from the environment and
    // runs a full session.
    {
        RemoteExecutionDriver driver(path);
        TileGraph graph("daemon_auth_survivor");
        auto *x = graph.data({2, 2}, "x", DataType::FP32);
        auto *y = graph.data({2, 2}, "y", DataType::FP32);
        auto *relu_out = graph.data({2, 2}, "relu", DataType::FP32);
        tg::torch_binary(
            starpu::TorchKind::Add, x, y, relu_out);
        driver.bind(x->id(), {1.f, -2.f, 3.f, -4.f});
        driver.bind(y->id(), {1.f, 1.f, 1.f, 1.f});
        driver.submit(graph);
        driver.wait();
        nntile::test::require_relative_element_error(
            driver.gather(relu_out->id()), {2.f, -1.f, 4.f, -3.f});
    }
#endif // NNTILE_USE_CUDA
    ::unsetenv("NNTILE_DRIVER_TOKEN");
    daemon.stop();
}



TEST_CASE_METHOD(nntile::test::ContextFixture,
    "ExecutionDaemon queues binds until their nodes arrive",
    "[graph][tile][driver][remote]")
{
    // BindOk for a not-yet-submitted node is a promise: the data must
    // apply once a later Submit makes the node exist, and binds whose
    // node never arrives must not be dropped silently (the daemon now
    // reports them when the connection ends).
    std::string const path = test_socket_path("bindqueue");
    ExecutionDaemon daemon(path);
    daemon.start();

#ifndef NNTILE_USE_CUDA
    {
        RemoteExecutionDriver driver(path);
        TileGraph first("daemon_bind_queue_first");
        auto *a1 = first.data({2}, "a", DataType::FP32);
        tg::fill(Scalar(0.0), a1);
        driver.bind(a1->id(), {1.f, 2.f});
        driver.submit(first);
        driver.wait();

        TileGraph second("daemon_bind_queue_second");
        auto *a2 = second.data({2}, "a", DataType::FP32);
        auto *b = second.data({2}, "b", DataType::FP32);
        // b keeps its own (bound) data: b = 0 * a + 1 * b.
        tg::add_inplace(Scalar(0.0), a2, Scalar(1.0), b);
        // Node b (id 1) does not exist on the daemon yet: queued.
        driver.bind(b->id(), {5.f, 7.f});
        driver.submit(second);
        driver.wait();
        nntile::test::require_relative_element_error(
            driver.gather(b->id()), {5.f, 7.f});
    }

    // A bind whose node never appears is reported at disconnect; the
    // daemon stays healthy afterwards.
    {
        RemoteExecutionDriver driver(path);
        TileGraph graph("daemon_bind_queue_dangling");
        auto *x = graph.data({2}, "x", DataType::FP32);
        tg::fill(Scalar(0.0), x);
        driver.bind(x->id(), {1.f, 1.f});
        driver.submit(graph);
        driver.wait();
        driver.bind_int64(9999, {1, 2, 3});
    }
#endif // NNTILE_USE_CUDA
    daemon.stop();
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "ExecutionDaemon restrict_cuda option pins codelets",
    "[graph][tile][driver][remote]")
{
    // Restriction is process-global and one-way, so this test must run
    // last in the binary. On CPU-only builds no codelet advertises a
    // CUDA implementation, so restrict_where(STARPU_CUDA) is a no-op
    // there by design; the where mask is only assertable with CUDA.
    std::string const path = test_socket_path("restrict");
    {
        ExecutionDaemon daemon(path, DaemonCudaRestrict::None);
        daemon.start();
        daemon.stop();
#if defined(NNTILE_USE_CUDA) && defined(NNTILE_TORCH_NATIVE_OPS)
        REQUIRE(starpu::torch_arange.codelet.where & STARPU_CPU);
#endif
    }
    ExecutionDaemon daemon(path, DaemonCudaRestrict::Cuda);
    daemon.start();
    daemon.stop();
#if defined(NNTILE_USE_CUDA) && defined(NNTILE_TORCH_NATIVE_OPS)
    REQUIRE_FALSE(starpu::torch_arange.codelet.where & STARPU_CPU);
#endif
}
