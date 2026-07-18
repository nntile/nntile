/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file nntile/tests/torch_native/aten_ref.hh
 * Helpers: compare StarPU buffers to CPU aten reference results.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/grad_mode.h>

// ATen pulls in a CHECK macro that clashes with Catch2.
#ifdef CHECK
#   undef CHECK
#endif

#include <nntile/base_types.hh>

namespace nntile::test::torch_native
{

inline float as_float(nntile::fp32_t v)
{
    return static_cast<float>(
        static_cast<nntile::fp32_t::repr_t>(v));
}

inline void require_close(
    const std::vector<float> &got,
    const std::vector<float> &ref,
    double rtol = 1e-5,
    double atol = 1e-5,
    char const *what = "buffer")
{
    if (got.size() != ref.size())
    {
        throw std::runtime_error(
            std::string(what) + ": size mismatch");
    }
    for (std::size_t i = 0; i < got.size(); ++i)
    {
        const double a = static_cast<double>(got[i]);
        const double b = static_cast<double>(ref[i]);
        const double tol = atol + rtol * std::abs(b);
        if (std::abs(a - b) > tol)
        {
            throw std::runtime_error(
                std::string(what) + ": mismatch at " +
                std::to_string(i) + " got=" + std::to_string(a) +
                " ref=" + std::to_string(b));
        }
    }
}

inline std::vector<std::int64_t> to_i64(
    const std::vector<nntile::Index> &shape)
{
    return std::vector<std::int64_t>(shape.begin(), shape.end());
}

inline at::Tensor blob_cpu(
    float *ptr,
    const std::vector<nntile::Index> &shape)
{
    auto sizes = to_i64(shape);
    return at::from_blob(
        ptr,
        at::IntArrayRef(sizes),
        /*deleter=*/[](void *) {},
        at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

//! Run ``fn`` under NoGrad + below AD (same guards as StarPU codelets).
template <typename Fn>
inline void with_cpu_aten(Fn &&fn)
{
    at::AutoDispatchBelowADInplaceOrView guard;
    at::NoGradGuard no_grad;
    fn();
}

} // namespace nntile::test::torch_native
