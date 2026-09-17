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

#include <nntile/execution_driver.hh>
#include <nntile/remote_execution_driver.hh>
#include <nntile/tile.hh>
#include <nntile/tile/ops/add.hh>
#include <nntile/tile/ops/add_inplace.hh>
#include <nntile/tile/ops/add_slice.hh>
#include <nntile/tile/ops/clear.hh>
#include <nntile/tile/ops/conv2d_inplace.hh>
#include <nntile/tile/ops/copy.hh>
#include <nntile/tile/ops/embedding.hh>
#include <nntile/tile/ops/fill.hh>
#include <nntile/tile/ops/gelu.hh>
#include <nntile/tile/ops/gemm.hh>
#include <nntile/tile/ops/multiply.hh>
#include <nntile/tile/ops/relu.hh>
#include <nntile/tile/ops/scale.hh>
#include <nntile/tile/ops/sum_slice.hh>
#include <nntile/tile/ops/transpose.hh>
#include <nntile/tile/ops/unregister.hh>
#ifdef NNTILE_TORCH_NATIVE_OPS
#include <nntile/tile/ops/torch_dispatch.hh>
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <grp.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

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

    struct FakeFftOp : TileGraph::OpNode
    {
        explicit FakeFftOp(TileGraph::TileNode *x)
        {
            inputs_ = {x};
            outputs_ = {x};
        }

        std::string op_name() const override
        {
            return "TILE_FFT";
        }

        void execute(Runtime &) const override
        {
        }

        std::shared_ptr<OpNode> clone() const override
        {
            return std::make_shared<FakeFftOp>(*this);
        }
    };

    TileGraph graph("driver_remote_unknown");
    auto *a = graph.data({2}, "a", DataType::FP32);
    graph.add_op(std::make_shared<FakeFftOp>(a));

    RemoteExecutionDriver driver(path);
    REQUIRE_THROWS_WITH(
        driver.submit(graph),
        Catch::Matchers::ContainsSubstring("UnknownOp"));
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RemoteExecutionDriver TILE_GELU family",
    "[graph][tile][driver][remote]")
{
    std::string const path = test_socket_path("gelu");
    ExecutionDaemon daemon(path);
    daemon.start();

    TileGraph local_g("driver_local_gelu");
    auto *lx = local_g.data({4}, "x", DataType::FP32);
    auto *ly = local_g.data({4}, "y", DataType::FP32);
    tg::gelu(lx, ly);
    RuntimeExecutionDriver local(local_g);
    local.runtime().compile();
    local.runtime().bind_data(
        lx, std::vector<float>{-1.f, 0.f, 1.f, 2.f});
    local.runtime().bind_data(
        ly, std::vector<float>{0.f, 0.f, 0.f, 0.f});
    local.submit(local_g);
    local.wait();
    auto const expect = local.runtime().get_output<float>(ly);

    TileGraph graph("driver_remote_gelu");
    auto *x = graph.data({4}, "x", DataType::FP32);
    auto *y = graph.data({4}, "y", DataType::FP32);
    tg::gelu(x, y);
    RemoteExecutionDriver driver(path);
    driver.bind(x->id(), {-1.f, 0.f, 1.f, 2.f});
    driver.bind(y->id(), {0.f, 0.f, 0.f, 0.f});
    driver.submit(graph);
    driver.wait();
    nntile::test::require_relative_element_error(
        driver.gather(y->id()), expect);
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

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RemoteExecutionDriver TILE_UNREGISTER",
    "[graph][tile][driver][remote]")
{
    std::string const path = test_socket_path("unreg");
    ExecutionDaemon daemon(path);
    daemon.start();

    TileGraph graph("driver_remote_unreg");
    auto *x = graph.data({4}, "x", DataType::FP32);
    auto *y = graph.data({4}, "y", DataType::FP32);
    tg::fill(Scalar(1.0), x);
    tg::copy(x, y);
    tg::unregister(x);

    RemoteExecutionDriver driver(path);
    driver.bind(x->id(), {0, 0, 0, 0});
    driver.bind(y->id(), {0, 0, 0, 0});
    driver.submit(graph);
    driver.wait();
    nntile::test::require_relative_element_error(
        driver.gather(y->id()), {1.f, 1.f, 1.f, 1.f});
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RemoteExecutionDriver classic TILE families",
    "[graph][tile][driver][remote]")
{
    std::string const path = test_socket_path("families");
    ExecutionDaemon daemon(path);
    daemon.start();
    RemoteExecutionDriver driver(path);

    TileGraph graph("driver_remote_families");
    auto *src = graph.data({4}, "src", DataType::FP32);
    auto *scaled = graph.data({4}, "scaled", DataType::FP32);
    auto *cleared = graph.data({4}, "cleared", DataType::FP32);
    auto *view = graph.data({2, 2}, "view", DataType::FP32);
    auto *sum_src = graph.data({2, 3}, "sum_src", DataType::FP32);
    auto *sum_dst = graph.data({3}, "sum_dst", DataType::FP32);
    auto *tr_src = graph.data({2, 3}, "tr_src", DataType::FP32);
    auto *tr_dst = graph.data({3, 2}, "tr_dst", DataType::FP32);
    tg::scale(Scalar(2.0), src, scaled);
    tg::clear(cleared);
    tg::copy_same_numel(src, view);
    tg::sum_slice(Scalar(1.0), sum_src, Scalar(0.0), sum_dst, 0, 0);
    tg::transpose(Scalar(1.0), tr_src, tr_dst, 1);

    driver.bind(src->id(), {1, 2, 3, 4});
    driver.bind(scaled->id(), {0, 0, 0, 0});
    driver.bind(cleared->id(), {9, 9, 9, 9});
    driver.bind(view->id(), {0, 0, 0, 0});
    driver.bind(sum_src->id(), {1, 2, 3, 4, 5, 6});
    driver.bind(sum_dst->id(), {0, 0, 0});
    driver.bind(tr_src->id(), {1, 2, 3, 4, 5, 6});
    driver.bind(tr_dst->id(), {0, 0, 0, 0, 0, 0});
    driver.submit(graph);
    driver.wait();
    nntile::test::require_relative_element_error(
        driver.gather(scaled->id()), {2.f, 4.f, 6.f, 8.f});
    nntile::test::require_relative_element_error(
        driver.gather(cleared->id()), {0.f, 0.f, 0.f, 0.f});
    nntile::test::require_relative_element_error(
        driver.gather(view->id()), {1.f, 2.f, 3.f, 4.f});
    nntile::test::require_relative_element_error(
        driver.gather(sum_dst->id()), {5.f, 7.f, 9.f});
    auto const tr = driver.gather(tr_dst->id());
    REQUIRE(tr.size() == 6);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RemoteExecutionDriver TILE_EMBEDDING INT64",
    "[graph][tile][driver][remote]")
{
    std::string const path = test_socket_path("emb");
    ExecutionDaemon daemon(path);
    daemon.start();

    Index const m = 2, n = 2, k = 3, k0 = 0, ks = 3;
    TileGraph graph("driver_remote_emb");
    auto *index = graph.data({m, n}, "index", DataType::INT64);
    auto *vocab = graph.data({ks, 5}, "vocab", DataType::FP32);
    auto *embed = graph.data({m, k, n}, "embed", DataType::FP32);
    tg::embedding(m, n, k, k0, ks, index, vocab, embed);

    std::vector<float> voc(15);
    for (int i = 0; i < 15; ++i)
    {
        voc[static_cast<size_t>(i)] = static_cast<float>(i + 1);
    }
    RemoteExecutionDriver driver(path);
    driver.bind_int64(index->id(), {0, 2, 4, 1});
    driver.bind(vocab->id(), voc);
    driver.bind(embed->id(), std::vector<float>(12, 0.f));
    driver.submit(graph);
    driver.wait();
    auto const out = driver.gather(embed->id());
    REQUIRE(out.size() == 12);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RemoteExecutionDriver TILE_CONV2D_INPLACE",
    "[graph][tile][driver][remote]")
{
    std::string const path = test_socket_path("conv");
    ExecutionDaemon daemon(path);
    daemon.start();

    TileGraph graph("driver_remote_conv");
    auto *X = graph.data({3, 3, 1, 1}, "X", DataType::FP32);
    auto *C = graph.data({2, 2, 1, 1}, "C", DataType::FP32);
    auto *Y = graph.data({2, 2, 1, 1}, "Y", DataType::FP32);
    tg::conv2d_inplace(
        3, 3, 1, 1, 2, 2, 1, 1, 1, 0, 0, Scalar(1.0), X, C, 2, 2, 1, 1,
        Scalar(0.0), Y);

    std::vector<float> xv(9), cv(4), yv(4, 0.f);
    for (Index i = 0; i < 9; ++i)
    {
        xv[static_cast<size_t>(i)] = static_cast<float>(i + 1);
    }
    for (Index i = 0; i < 4; ++i)
    {
        cv[static_cast<size_t>(i)] = static_cast<float>(i + 1);
    }
    RemoteExecutionDriver driver(path);
    driver.bind(X->id(), xv);
    driver.bind(C->id(), cv);
    driver.bind(Y->id(), yv);
    driver.submit(graph);
    driver.wait();
    auto const out = driver.gather(Y->id());
    REQUIRE(out.size() == 4);
}

TEST_CASE(
    "ExecutionDaemon inits StarPU when none exists",
    "[graph][tile][driver][remote]")
{
    std::string const path = test_socket_path("nocontext");
    ExecutionDaemon daemon(path);
    daemon.start();
    require_socket_acl(path);

    TileGraph graph("driver_remote_nocontext");
    auto *x = graph.data({4}, "x", DataType::FP32);
    tg::fill(Scalar(2.0), x);

    RemoteExecutionDriver driver(path);
    driver.bind(x->id(), {0, 0, 0, 0});
    driver.submit(graph);
    driver.wait();
    nntile::test::require_relative_element_error(
        driver.gather(x->id()), {2.f, 2.f, 2.f, 2.f});
}
