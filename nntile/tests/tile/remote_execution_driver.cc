/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile/remote_execution_driver.cc
 * RemoteExecutionDriver submits a TileGraph over a Unix socket.
 *
 * @version 1.1.0
 * */

#include "context_fixture.hh"
#include "test_frobenius.hh"

#include <nntile/remote_execution_driver.hh>
#include <nntile/tile.hh>
#include <nntile/tile/ops/add_inplace.hh>
#include <nntile/tile/ops/fill.hh>
#include <nntile/tile/ops/gemm.hh>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace nntile;
namespace tg = nntile::tile;

namespace
{

std::string test_socket_path(char const *tag)
{
    return std::string("/tmp/nntile-driver-") + tag + "-" +
        std::to_string(::getpid()) + ".sock";
}

} // namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RemoteExecutionDriver submit over NNTILE_DRIVER_SOCKET",
    "[graph][tile][driver][remote]")
{
    std::string const path = test_socket_path("add");
    ExecutionDaemon daemon(path);
    daemon.start();

    struct stat st{};
    REQUIRE(::stat(path.c_str(), &st) == 0);
    REQUIRE((st.st_mode & 0777) ==
        static_cast<mode_t>(S_IRUSR | S_IWUSR));

    std::vector<Index> shape = {4};
    TileGraph graph("driver_remote");
    auto *x = graph.data(shape, "x", DataType::FP32);
    auto *y = graph.data(shape, "y", DataType::FP32);
    tg::add_inplace(Scalar(2.0), x, Scalar(1.0), y);

    RemoteExecutionDriver driver(path);
    driver.bind(x->id(), {1, 2, 3, 4});
    driver.bind(y->id(), {10, 20, 30, 40});
    driver.submit(graph);
    driver.wait();
    auto const result = driver.gather(y->id());
    nntile::test::require_relative_element_error(
        result, {12.f, 24.f, 36.f, 48.f});
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RemoteExecutionDriver rejects unknown tile ops",
    "[graph][tile][driver][remote]")
{
    std::string const path = test_socket_path("unknown");
    ExecutionDaemon daemon(path);
    daemon.start();

    TileGraph graph("driver_remote_unknown");
    auto *a = graph.data({2, 2}, "a", DataType::FP32);
    auto *b = graph.data({2, 2}, "b", DataType::FP32);
    auto *c = graph.data({2, 2}, "c", DataType::FP32);
    tg::gemm(a, b, c, Scalar(1.0), Scalar(0.0), false, false, 1, 0);

    RemoteExecutionDriver driver(path);
    REQUIRE_THROWS_WITH(
        driver.submit(graph),
        Catch::Matchers::ContainsSubstring("UnknownOp"));
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RemoteExecutionDriver fill over NNTILE_DRIVER_SOCKET",
    "[graph][tile][driver][remote]")
{
    std::string const path = test_socket_path("fill");
    ExecutionDaemon daemon(path);
    daemon.start();

    TileGraph graph("driver_remote_fill");
    auto *x = graph.data({4}, "x", DataType::FP32);
    tg::fill(Scalar(3.0), x);

    RemoteExecutionDriver driver(path);
    driver.bind(x->id(), {0, 0, 0, 0});
    driver.submit(graph);
    driver.wait();
    auto const result = driver.gather(x->id());
    nntile::test::require_relative_element_error(
        result, {3.f, 3.f, 3.f, 3.f});
}
