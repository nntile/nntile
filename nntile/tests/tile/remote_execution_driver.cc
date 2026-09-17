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
#include <nntile/tile/ops/add.hh>
#include <nntile/tile/ops/add_inplace.hh>
#include <nntile/tile/ops/add_slice.hh>
#include <nntile/tile/ops/copy.hh>
#include <nntile/tile/ops/fill.hh>
#include <nntile/tile/ops/gelu.hh>
#include <nntile/tile/ops/gemm.hh>
#include <nntile/tile/ops/multiply.hh>
#include <nntile/tile/ops/relu.hh>
#ifdef NNTILE_TORCH_NATIVE_OPS
#include <nntile/tile/ops/torch_dispatch.hh>
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <grp.h>
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

void require_socket_acl(std::string const &path)
{
    struct stat st{};
    REQUIRE(::stat(path.c_str(), &st) == 0);
    REQUIRE((st.st_mode & 0777) ==
        static_cast<mode_t>(S_IRUSR | S_IWUSR));
    REQUIRE(std::string(driver_socket_group_name()) == "nntile-ops");
    struct group *gr = ::getgrnam(driver_socket_group_name());
    if (gr != nullptr)
    {
        REQUIRE(st.st_gid == gr->gr_gid);
    }
}

} // namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RemoteExecutionDriver submit over NNTILE_DRIVER_SOCKET",
    "[graph][tile][driver][remote]")
{
    std::string const path = test_socket_path("add");
    ExecutionDaemon daemon(path);
    daemon.start();
    require_socket_acl(path);

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
    auto *a = graph.data({2}, "a", DataType::FP32);
    auto *b = graph.data({2}, "b", DataType::FP32);
    tg::gelu(a, b);

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

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RemoteExecutionDriver v1 Flush tile allowlist",
    "[graph][tile][driver][remote]")
{
    std::string const path = test_socket_path("v1");
    ExecutionDaemon daemon(path);
    daemon.start();
    RemoteExecutionDriver driver(path);

    TileGraph graph("driver_remote_v1");
    auto *x = graph.data({2, 2}, "x", DataType::FP32);
    auto *y = graph.data({2, 2}, "y", DataType::FP32);
    auto *add_out = graph.data({2, 2}, "add", DataType::FP32);
    auto *mul_out = graph.data({2, 2}, "mul", DataType::FP32);
    auto *mm_out = graph.data({2, 2}, "mm", DataType::FP32);
    auto *relu_out = graph.data({2, 2}, "relu", DataType::FP32);
    auto *copy_out = graph.data({2, 2}, "copy", DataType::FP32);
    auto *bias = graph.data({2}, "bias", DataType::FP32);
    auto *slice_out = graph.data({2, 2}, "slice", DataType::FP32);

    tg::add(Scalar(1.0), x, Scalar(1.0), y, add_out);
    tg::multiply(Scalar(1.0), x, y, mul_out);
    tg::gemm(
        x, y, mm_out, Scalar(1.0), Scalar(0.0), false, false, 1, 0);
    tg::relu(x, relu_out);
    tg::copy(x, copy_out);
    tg::add_slice(
        Scalar(1.0), bias, Scalar(1.0), add_out, slice_out, 0);

    driver.bind(x->id(), {1, 2, 3, 4});
    driver.bind(y->id(), {1, 0, 0, 1});
    driver.bind(mm_out->id(), {0, 0, 0, 0});
    driver.bind(bias->id(), {0.5f, -0.25f});
    driver.bind(slice_out->id(), {0, 0, 0, 0});
    driver.submit(graph);
    driver.wait();

    nntile::test::require_relative_element_error(
        driver.gather(add_out->id()), {2.f, 2.f, 3.f, 5.f});
    nntile::test::require_relative_element_error(
        driver.gather(mul_out->id()), {1.f, 0.f, 0.f, 4.f});
    nntile::test::require_relative_element_error(
        driver.gather(copy_out->id()), {1.f, 2.f, 3.f, 4.f});
    nntile::test::require_relative_element_error(
        driver.gather(relu_out->id()), {1.f, 2.f, 3.f, 4.f});
    nntile::test::require_relative_element_error(
        driver.gather(mm_out->id()), {1.f, 2.f, 3.f, 4.f});
    auto const slice = driver.gather(slice_out->id());
    REQUIRE(slice.size() == 4);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RemoteExecutionDriver tiles survive two Flushes",
    "[graph][tile][driver][remote]")
{
    std::string const path = test_socket_path("resident");
    ExecutionDaemon daemon(path);
    daemon.start();

    TileGraph graph("driver_remote_resident");
    auto *x = graph.data({2}, "x", DataType::FP32);
    auto *y = graph.data({2}, "y", DataType::FP32);
    tg::add_inplace(Scalar(1.0), x, Scalar(1.0), y);

    RemoteExecutionDriver driver(path);
    driver.bind(x->id(), {1, 2});
    driver.bind(y->id(), {3, 4});
    driver.submit(graph);
    driver.wait();
    nntile::test::require_relative_element_error(
        driver.gather(y->id()), {4.f, 6.f});

    auto *z = graph.data({2}, "z", DataType::FP32);
    tg::add_inplace(Scalar(1.0), y, Scalar(1.0), z);
    driver.bind(z->id(), {0, 0});
    driver.submit(graph);
    driver.wait();
    nntile::test::require_relative_element_error(
        driver.gather(y->id()), {4.f, 6.f});
    nntile::test::require_relative_element_error(
        driver.gather(z->id()), {4.f, 6.f});
}

#ifdef NNTILE_TORCH_NATIVE_OPS
TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RemoteExecutionDriver TILE_TORCH v1 kinds",
    "[graph][tile][driver][remote]")
{
    std::string const path = test_socket_path("torch");
    ExecutionDaemon daemon(path);
    daemon.start();
    RemoteExecutionDriver driver(path);

    TileGraph graph("driver_remote_torch");
    auto *x = graph.data({2, 2}, "x", DataType::FP32);
    auto *y = graph.data({2, 2}, "y", DataType::FP32);
    auto *add_out = graph.data({2, 2}, "add", DataType::FP32);
    auto *relu_out = graph.data({2, 2}, "relu", DataType::FP32);
    auto *bias = graph.data({2}, "bias", DataType::FP32);
    auto *lin_out = graph.data({2, 2}, "lin", DataType::FP32);
    tg::torch_binary(
        starpu::TorchKind::Add, x, y, add_out);
    tg::torch_unary(
        starpu::TorchKind::Relu, x, relu_out);
    tg::torch_ternary(
        starpu::TorchKind::Linear, x, y, bias, lin_out);

    driver.bind(x->id(), {1, -2, 3, -4});
    driver.bind(y->id(), {1, 1, 1, 1});
    driver.bind(bias->id(), {0.5f, -0.5f});
    driver.submit(graph);
    driver.wait();
    nntile::test::require_relative_element_error(
        driver.gather(add_out->id()), {2.f, -1.f, 4.f, -3.f});
    nntile::test::require_relative_element_error(
        driver.gather(relu_out->id()), {1.f, 0.f, 3.f, 0.f});
    auto const lin = driver.gather(lin_out->id());
    REQUIRE(lin.size() == 4);
}
#endif
