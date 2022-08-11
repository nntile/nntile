/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/randn.cc
 * Randn operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-11
 * */

#include "nntile/starpu/randn.hh"
#include "nntile/kernel/cpu/randn.hh"

namespace nntile
{
namespace starpu
{

//! Randn operation on StarPU buffers
template<typename T>
void randn_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    const Index *ndim_ptr, *nelems_ptr, *start, *shape, *stride,
          *underlying_shape;
    const unsigned long long *seed_ptr;
    const T *mean_ptr, *stddev_ptr;
    Starpu::unpack_args_ptr(cl_args, ndim_ptr, nelems_ptr, seed_ptr, mean_ptr,
            stddev_ptr, start, shape, stride, underlying_shape);
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    Index *tmp_index = interfaces[1]->get_ptr<Index>();
    // Launch kernel
    kernel::cpu::randn<T>(*ndim_ptr, *nelems_ptr, *seed_ptr, *mean_ptr,
            *stddev_ptr, start, shape, underlying_shape, data, stride,
            tmp_index);
}

//! Footprint for randn tasks that depend on shape
template<typename T>
static
uint32_t randn_footprint(struct starpu_task *task)
{
    // Get arguments
    const Index *ndim_ptr, *nelems_ptr, *start, *shape, *stride,
          *underlying_shape;
    const unsigned long long *seed_ptr;
    const T *mean_ptr, *stddev_ptr;
    Starpu::unpack_args_ptr(task->cl_arg, ndim_ptr, nelems_ptr, seed_ptr,
            mean_ptr, stddev_ptr, start, shape, stride, underlying_shape);
    std::size_t shape_size = *ndim_ptr * sizeof(*shape);
    // Apply hash over parameter copy_shape
    return starpu_hash_crc32c_be_n(shape, shape_size, 0);
}

StarpuCodelet randn_codelet_fp32("nntile_randn_fp32",
        randn_footprint<fp32_t>,
        {randn_cpu<fp32_t>},
        {});

StarpuCodelet randn_codelet_fp64("nntile_randn_fp64",
        randn_footprint<fp64_t>,
        {randn_cpu<fp64_t>},
        {});

void randn_restrict_where(uint32_t where)
{
    randn_codelet_fp32.restrict_where(where);
    randn_codelet_fp64.restrict_where(where);
}

void randn_restore_where()
{
    randn_codelet_fp32.restore_where();
    randn_codelet_fp64.restore_where();
}

template<typename T>
constexpr StarpuCodelet *randn_codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *randn_codelet<fp32_t>()
{
    return &randn_codelet_fp32;
}

template<>
constexpr StarpuCodelet *randn_codelet<fp64_t>()
{
    return &randn_codelet_fp64;
}

template<typename T>
void randn(Index ndim, Index nelems, unsigned long long seed,
        T mean, T stddev, const std::vector<Index> &start,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        const std::vector<Index> &underlying_shape, starpu_data_handle_t data,
        starpu_data_handle_t tmp_index)
{
    fp64_t nflops = 2 * nelems;
    // Submit task
    int ret = starpu_task_insert(randn_codelet<T>(),
            STARPU_VALUE, &ndim, sizeof(ndim),
            STARPU_VALUE, &nelems, sizeof(nelems),
            STARPU_VALUE, &seed, sizeof(seed),
            STARPU_VALUE, &mean, sizeof(mean),
            STARPU_VALUE, &stddev, sizeof(stddev),
            STARPU_VALUE, &start[0], ndim*sizeof(start[0]),
            STARPU_VALUE, &shape[0], ndim*sizeof(shape[0]),
            STARPU_VALUE, &stride[0], ndim*sizeof(stride[0]),
            STARPU_VALUE, &underlying_shape[0],
            ndim*sizeof(underlying_shape[0]),
            STARPU_W, data,
            STARPU_SCRATCH, tmp_index,
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in randn task submission");
    }
}

// Explicit instantiation
template
void randn<fp32_t>(Index ndim, Index nelems, unsigned long long seed,
        fp32_t mean, fp32_t stddev, const std::vector<Index> &start,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        const std::vector<Index> &underlying_shape, starpu_data_handle_t data,
        starpu_data_handle_t tmp_index);

template
void randn<fp64_t>(Index ndim, Index nelems, unsigned long long seed,
        fp64_t mean, fp64_t stddev, const std::vector<Index> &start,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        const std::vector<Index> &underlying_shape, starpu_data_handle_t data,
        starpu_data_handle_t tmp_index);

} // namespace starpu
} // namespace nntile

