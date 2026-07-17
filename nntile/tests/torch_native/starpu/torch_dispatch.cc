/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file nntile/tests/torch_native/starpu/torch_dispatch.cc
 * StarPU torch_dispatch codelets vs matching CPU aten *_out.
 */

#include "aten_ref.hh"
#include "context_fixture.hh"

#include <nntile/core/torch_meta.hh>
#include <nntile/starpu/torch_dispatch.hh>

#include <ATen/ops/mm.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/relu.h>

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace nntile;
using namespace nntile::starpu;
namespace tn = nntile::test::torch_native;

namespace
{

TorchDispatchArgs pack2d(
    TorchKind kind,
    Index rows,
    Index cols)
{
    TorchDispatchArgs args{};
    args.kind = kind;
    args.n_in = 1;
    args.n_out = 1;
    args.in_ndim[0] = 2;
    args.out_ndim[0] = 2;
    args.in_sizes[0][0] = rows;
    args.in_sizes[0][1] = cols;
    args.out_sizes[0][0] = rows;
    args.out_sizes[0][1] = cols;
    args.in_strides[0][0] = cols;
    args.in_strides[0][1] = 1;
    args.out_strides[0][0] = cols;
    args.out_strides[0][1] = 1;
    return args;
}

} // namespace

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "starpu torch_unary Relu matches aten::relu_out",
    "[torch_native][starpu]")
{
    const Index rows = 2;
    const Index cols = 3;
    const Index nelems = rows * cols;
    std::vector<float> in = {-1.f, 0.f, 2.f, -3.f, 4.f, 0.5f};
    std::vector<float> out(nelems, 0.f);
    std::vector<float> ref(nelems, 0.f);

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor t_in = tn::blob_cpu(in.data(), {rows, cols});
            at::Tensor t_ref = tn::blob_cpu(ref.data(), {rows, cols});
            at::relu_out(t_ref, t_in);
        });

    auto meta = pack2d(TorchKind::Relu, rows, cols);
    VariableHandle h_in(in.data(), sizeof(float) * nelems);
    VariableHandle h_out(out.data(), sizeof(float) * nelems);
    torch_unary.restrict_where(STARPU_CPU);
    torch_unary.submit<std::tuple<fp32_t>>(-1, meta, h_in, h_out);
    starpu_task_wait_for_all();
    h_in.unregister();
    h_out.unregister();
    tn::require_close(out, ref);
}

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "starpu torch_binary Mul matches aten::mul_out",
    "[torch_native][starpu]")
{
    const Index rows = 2;
    const Index cols = 2;
    const Index nelems = 4;
    std::vector<float> a = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> b = {2.f, 0.5f, -1.f, 3.f};
    std::vector<float> out(nelems, 0.f);
    std::vector<float> ref(nelems, 0.f);

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor ta = tn::blob_cpu(a.data(), {rows, cols});
            at::Tensor tb = tn::blob_cpu(b.data(), {rows, cols});
            at::Tensor tr = tn::blob_cpu(ref.data(), {rows, cols});
            at::mul_out(tr, ta, tb);
        });

    TorchDispatchArgs meta = pack2d(TorchKind::Mul, rows, cols);
    meta.n_in = 2;
    meta.in_ndim[1] = 2;
    meta.in_sizes[1][0] = rows;
    meta.in_sizes[1][1] = cols;
    meta.in_strides[1][0] = cols;
    meta.in_strides[1][1] = 1;

    VariableHandle ha(a.data(), sizeof(float) * nelems);
    VariableHandle hb(b.data(), sizeof(float) * nelems);
    VariableHandle hout(out.data(), sizeof(float) * nelems);
    torch_binary.restrict_where(STARPU_CPU);
    torch_binary.submit<std::tuple<fp32_t>>(-1, meta, ha, hb, hout);
    starpu_task_wait_for_all();
    ha.unregister();
    hb.unregister();
    hout.unregister();
    tn::require_close(out, ref);
}

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "starpu torch_binary Mm matches aten::mm_out",
    "[torch_native][starpu]")
{
    std::vector<float> a = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}; // 2x3
    std::vector<float> b = {1.f, 0.f, 0.f, 1.f, 1.f, 1.f}; // 3x2
    std::vector<float> out(4, 0.f);
    std::vector<float> ref(4, 0.f);

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor ta = tn::blob_cpu(a.data(), {2, 3});
            at::Tensor tb = tn::blob_cpu(b.data(), {3, 2});
            at::Tensor tr = tn::blob_cpu(ref.data(), {2, 2});
            at::mm_out(tr, ta, tb);
        });

    TorchDispatchArgs meta{};
    meta.kind = TorchKind::Mm;
    meta.n_in = 2;
    meta.n_out = 1;
    meta.in_ndim[0] = 2;
    meta.in_ndim[1] = 2;
    meta.out_ndim[0] = 2;
    meta.in_sizes[0][0] = 2;
    meta.in_sizes[0][1] = 3;
    meta.in_strides[0][0] = 3;
    meta.in_strides[0][1] = 1;
    meta.in_sizes[1][0] = 3;
    meta.in_sizes[1][1] = 2;
    meta.in_strides[1][0] = 2;
    meta.in_strides[1][1] = 1;
    meta.out_sizes[0][0] = 2;
    meta.out_sizes[0][1] = 2;
    meta.out_strides[0][0] = 2;
    meta.out_strides[0][1] = 1;

    VariableHandle ha(a.data(), sizeof(float) * 6);
    VariableHandle hb(b.data(), sizeof(float) * 6);
    VariableHandle hout(out.data(), sizeof(float) * 4);
    torch_binary.restrict_where(STARPU_CPU);
    torch_binary.submit<std::tuple<fp32_t>>(-1, meta, ha, hb, hout);
    starpu_task_wait_for_all();
    ha.unregister();
    hb.unregister();
    hout.unregister();
    tn::require_close(out, ref);
}
