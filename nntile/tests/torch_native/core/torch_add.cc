/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file nntile/tests/torch_native/core/torch_add.cc
 * core::torch_add_out vs aten::add_out.
 */

#include "aten_ref.hh"
#include "context_fixture.hh"

#include <nntile/core/torch_add.hh>
#include <nntile/core/torch_meta.hh>

#include <ATen/ops/add.h>

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace nntile;
using namespace nntile::core;
namespace tn = nntile::test::torch_native;

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "core torch_add_out matches aten::add_out",
    "[torch_native][core]")
{
    const std::vector<Index> shape = {4, 3};
    const Index nelems = 12;
    const Scalar alpha = 0.75f;

    Tile<fp32_t> self(shape), other(shape), out(shape);
    std::vector<float> ref(nelems, 0.f);

    {
        auto s = self.acquire(STARPU_W);
        auto o = other.acquire(STARPU_W);
        for (Index i = 0; i < nelems; ++i)
        {
            s[i] = static_cast<float>(i);
            o[i] = static_cast<float>(2 * i + 1);
        }
        s.release();
        o.release();
    }

    std::vector<float> self_host(nelems), other_host(nelems);
    {
        auto s = self.acquire(STARPU_R);
        auto o = other.acquire(STARPU_R);
        for (Index i = 0; i < nelems; ++i)
        {
            self_host[static_cast<size_t>(i)] = s[i];
            other_host[static_cast<size_t>(i)] = o[i];
        }
        s.release();
        o.release();
    }

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor t_self = tn::blob_cpu(self_host.data(), shape);
            at::Tensor t_other = tn::blob_cpu(other_host.data(), shape);
            at::Tensor t_ref = tn::blob_cpu(ref.data(), shape);
            at::add_out(t_ref, t_self, t_other, alpha);
        });

    const TorchTileMeta meta = make_contiguous_torch_meta(shape);
    torch_add_out<fp32_t>(
        -1,
        self,
        meta,
        other,
        meta,
        out,
        meta,
        alpha);
    starpu_task_wait_for_all();

    std::vector<float> got(nelems);
    {
        auto z = out.acquire(STARPU_R);
        for (Index i = 0; i < nelems; ++i)
        {
            got[static_cast<size_t>(i)] = z[i];
        }
        z.release();
    }
    tn::require_close(got, ref);
}
