/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/starpu/torch_add.hh
 * Torch-native add on StarPU buffers (CPU, no nntile::kernel).
 *
 * @version 1.1.0
 */

#pragma once

#include <nntile/defs.h>

#ifndef NNTILE_TORCH_NATIVE_OPS
#error "nntile/starpu/torch_add.hh requires NNTILE_TORCH_NATIVE_OPS"
#endif

#include <tuple>

#include <nntile/core/torch_meta.hh>
#include <nntile/starpu/codelet.hh>
#include <nntile/starpu/handle.hh>

namespace nntile::starpu
{

//! Max rank packed into torch_add args_t (contiguous FP32 path).
inline constexpr Index torch_add_max_ndim = core::torch_native_max_ndim;

//! Torch-native add: out = self + alpha * other (aten::add.out).
//! Access: self ``STARPU_R``, other ``STARPU_R``, out ``STARPU_W``.
template<typename T>
class TorchAdd;

template<typename T>
class TorchAdd<std::tuple<T>>
{
public:
    CodeletTyped<T> codelet;

    TorchAdd();

    //! All tensor meta compressed for starpu_task_insert.
    struct args_t
    {
        Index ndim;
        Scalar alpha;
        Index sizes[torch_add_max_ndim];
        Index self_strides[torch_add_max_ndim];
        Index other_strides[torch_add_max_ndim];
        Index out_strides[torch_add_max_ndim];
    };

    static uint32_t footprint(struct starpu_task *task);

    static void cpu(void *buffers[], void *cl_args) noexcept;

    static constexpr func_array cpu_funcs = {
        cpu
    };

    static constexpr func_array cuda_funcs = {};

    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle self,
        Handle other,
        Handle out
    );
};

using torch_add_pack_t = OperationPack<
    TorchAdd,
    std::tuple<nntile::fp32_t>
>;

extern torch_add_pack_t torch_add;

} // namespace nntile::starpu
