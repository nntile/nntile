/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/norm.hh
 * Functions that compute different norms.
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tensor/norm.hh"

namespace nntile
{

template<typename T>
static void cpu_sum_ssq_accumulate(void *buffers[], void *cl_args)
{
    Index nelems;
    starpu_codelet_unpack_args(cl_args, &nelems);
    const T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    for(Index i = 0; i < nelems; i += 3)
    {
        // If maximal absolute value is 0 do no update to avoid division by 0
        if(src[i+1] == 0)
        {
            continue;
        }
        // Now src[i+1]>0
        dst[i] += src[i];
        if(dst[i+1] > src[i+1])
        {
            T tmp = src[i+1] / dst[i+1];
            dst[i+2] += src[i+2] * tmp * tmp;
        }
        else
        {
            // No division by 0 here since src[i+1]>0
            T tmp = dst[i+1] / src[i+1];
            dst[i+1] = src[i+1];
            dst[i+2] = dst[i+2]*tmp*tmp + src[i+2];
        }
    }
}

template<typename T>
void norm_sum_ssq_accumulate_async(const Tile<T> &sum_ssq,
        const Tile<T> &sum_ssq_total)
{
    static starpu_codelet codelet_sum_ssq_accumulate =
    {
        .cpu_funcs = {cpu_sum_ssq_accumulate<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_RW}
    };
    // Check inputs
    if(sum_ssq.ndim != sum_ssq_total.ndim)
    {
        throw std::runtime_error("sum_ssq.ndim != sum_ssq_total.ndim");
    }
    Index ndim = sum_ssq.ndim;
    for(Index i = 0; i < ndim; ++i)
    {
        if(sum_ssq.shape[i] != sum_ssq_total.shape[i])
        {
            throw std::runtime_error("sum_ssq.shape[i] != "
                    "sum_ssq_total.shape[i]");
        }
    }
    // Insert task
    starpu_task_insert(&codelet_sum_ssq_accumulate,
            STARPU_VALUE, &(sum_ssq.nelems), sizeof(sum_ssq.nelems),
            STARPU_R, static_cast<starpu_data_handle_t>(sum_ssq),
            STARPU_RW, static_cast<starpu_data_handle_t>(sum_ssq_total),
            0);
}

template
void norm_sum_ssq_accumulate_async(const Tile<fp32_t> &sum_ssq,
        const Tile<fp32_t> &sum_ssq_total);

template
void norm_sum_ssq_accumulate_async(const Tile<fp64_t> &sum_ssq,
        const Tile<fp64_t> &sum_ssq_total);

template<typename T>
void norm_sum_ssq_async(const Tensor<T> &src, const Tensor<T> &sum_ssq,
        const Tensor<T> &sum_ssq_work, const std::vector<Index> &axes)
{
    // Check inputs
    if(src.ndim+1 != sum_ssq.ndim+axes.size())
    {
        throw std::runtime_error("src.ndim+1 != sum_ssq.ndim+axes.size()");
    }
    // Treat special case of src.ndim=0
    if(src.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Treat special case of empty axes
    if(axes.size() == 0)
    {
        throw std::runtime_error("Empty axes");
    }
    // Check axes
    if(axes[0] < 0)
    {
        throw std::runtime_error("axes[0] < 0");
    }
    if(axes[axes.size()-1] >= src.ndim)
    {
        throw std::runtime_error("axes[axes.size()-1] >= src.ndim");
    }
    for(Index i = 1; i < axes.size(); ++i)
    {
        if(axes[i] <= axes[i-1])
        {
            throw std::runtime_error("axes[i] <= axes[i-1]");
        }
    }
    // Check shapes of src and sum_ssq
    if(sum_ssq.shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq.shape[0] != 3");
    }
    // Number of checked items in axes
    Index nchecked_axes = 0;
    for(Index i = 0; i < src.ndim; ++i)
    {
        if(nchecked_axes < axes.size() and i == axes[nchecked_axes])
        {
            ++nchecked_axes;
            continue;
        }
        if(src.shape[i] != sum_ssq.shape[i-nchecked_axes+1])
        {
            throw std::runtime_error("src.shape[i] != "
                    "sum_ssq.shape[i-nchecked_axes+1]");
        }
    }
}

template
void norm_sum_ssq_async(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &sum_ssq, const Tensor<fp32_t> &sum_ssq_work,
        const std::vector<Index> &axes);

template
void norm_sum_ssq_async(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &sum_ssq, const Tensor<fp64_t> &sum_ssq_work,
        const std::vector<Index> &axes);

} // namespace nntile

