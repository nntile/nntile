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
#include <nntile/tile/ops/fill.hh>
#include <nntile/starpu/fill.hh>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

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
    REQUIRE(::connect(
        fd, reinterpret_cast<sockaddr const *>(&addr), sizeof(addr)) == 0);
    char const garbage[] = "\x00\x01\x02 not json at all {{{";
    ssize_t const n = ::write(fd, garbage, sizeof(garbage) - 1);
    REQUIRE(n > 0);
    ::close(fd);
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

    // The daemon must still serve a well-formed client afterwards.
    {
        RemoteExecutionDriver driver(path);
        TileGraph graph("daemon_garbage_survivor");
        auto *x = graph.data({4}, "x", DataType::FP32);
        auto *y = graph.data({4}, "y", DataType::FP32);
        tg::fill(Scalar(3.0), y);
        tg::copy(x, y);
        driver.bind(x->id(), {1.f, 2.f, 3.f, 4.f});
        driver.submit(graph);
        driver.wait();
        nntile::test::require_relative_element_error(
            driver.gather(y->id()), {1.f, 2.f, 3.f, 4.f});
    }
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
