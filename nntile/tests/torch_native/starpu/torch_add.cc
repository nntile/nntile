/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file nntile/tests/torch_native/starpu/torch_add.cc
 * StarPU torch_add codelet vs aten::add_out (CPU, no grad).
 */

#include "aten_ref.hh"
#include "context_fixture.hh"

#include <nntile/starpu/torch_add.hh>

#include <ATen/ops/add.h>

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace nntile;
using namespace nntile::starpu;
namespace tn = nntile::test::torch_native;

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "starpu torch_add matches aten::add_out",
    "[torch_native][starpu]")
{
    const std::vector<Index> shape = {3, 4};
    const Index nelems = 12;
    const Scalar alpha = 1.5f;

    std::vector<float> self(nelems), other(nelems), out(nelems, 0.f);
    std::vector<float> ref(nelems, 0.f);
    for (Index i = 0; i < nelems; ++i)
    {
        self[static_cast<size_t>(i)] = static_cast<float>(i + 1);
        other[static_cast<size_t>(i)] = static_cast<float>(-i);
    }

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor t_self = tn::blob_cpu(self.data(), shape);
            at::Tensor t_other = tn::blob_cpu(other.data(), shape);
            at::Tensor t_ref = tn::blob_cpu(ref.data(), shape);
            at::add_out(t_ref, t_self, t_other, alpha);
        });

    TorchAdd<std::tuple<fp32_t>>::args_t meta{};
    meta.ndim = 2;
    meta.alpha = alpha;
    meta.sizes[0] = 3;
    meta.sizes[1] = 4;
    meta.self_strides[0] = 4;
    meta.self_strides[1] = 1;
    meta.other_strides[0] = 4;
    meta.other_strides[1] = 1;
    meta.out_strides[0] = 4;
    meta.out_strides[1] = 1;

    VariableHandle h_self(self.data(), sizeof(float) * nelems);
    VariableHandle h_other(other.data(), sizeof(float) * nelems);
    VariableHandle h_out(out.data(), sizeof(float) * nelems);
    torch_add.restrict_where(STARPU_CPU);
    torch_add.submit<std::tuple<fp32_t>>(
        -1,
        meta,
        h_self,
        h_other,
        h_out);
    starpu_task_wait_for_all();
    h_self.unregister();
    h_other.unregister();
    h_out.unregister();

    tn::require_close(out, ref, 1e-5, 1e-5, "torch_add");
}
