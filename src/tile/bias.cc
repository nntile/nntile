/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/bias.cc
 * Bias operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/bias.hh"

namespace nntile
{

template<typename T>
static void cpu_bias(void *buffers[], void *cl_args)
{
    Index m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    const Index mk = m * k;
    const T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
//    Index dst_offset = 0;
//    for(Index i2 = 0; i2 < n; ++i2)
//    {
//        for(Index i1 = 0; i1 < k; ++i1)
//        {
//            Index src_offset = i2 * m;
//            for(Index i0 = 0; i0 < m; ++i0)
//            {
//                dst[dst_offset] += src[src_offset];
//                ++dst_offset;
//                ++src_offset;
//            }
//        }
//    }
    Index src_offset = 0;
    for(Index i2 = 0; i2 < n; ++i2)
    {
        for(Index i1 = 0; i1 < m; ++i1)
        {
            T *dst_slice = dst + i2*mk + i1;
            const T src_val = src[src_offset];
            ++src_offset;
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T &dst_val = dst_slice[i0*m];
                dst_val = dst_val + src_val;
            }
        }
    }
}

template<typename T>
void bias_async(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    // StarPU codelet
    constexpr auto commute_mode = static_cast<enum starpu_data_access_mode>(
            STARPU_RW | STARPU_COMMUTE);
    static struct starpu_codelet codelet_bias =
    {
        .cpu_funcs = {cpu_bias<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, commute_mode}
    };
    // Check dimensions
    if(dst.ndim != src.ndim+1)
    {
        throw std::runtime_error("dst.ndim != src.ndim+1");
    }
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    // Check shapes of input tiles
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i-1])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i-1]");
        }
    }
    // Reshape inputs for simplicity: src -> (m,n), dst -> (m,k,n)
    Index m, n, k;
    if(axis == 0)
    {
        m = 1;
        n = src.nelems;
        k = dst.shape[0];
    }
    else if(axis == dst.ndim-1)
    {
        m = src.nelems;
        n = 1;
        k = dst.shape[axis];
    }
    else
    {
        m = dst.stride[axis];
        n = dst.matrix_shape[axis+1][1];
        k = dst.shape[axis];
    }
    // Insert corresponding task
    starpu_task_insert(&codelet_bias,
            STARPU_VALUE, &m, sizeof(m),
            STARPU_VALUE, &n, sizeof(n),
            STARPU_VALUE, &k, sizeof(k),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            commute_mode, static_cast<starpu_data_handle_t>(dst),
            STARPU_FLOPS, static_cast<double>(dst.nelems),
            0);
}

template
void bias_async(const Tile<float> &src, const Tile<float> &dst,
        Index axis);

template
void bias_async(const Tile<double> &src, const Tile<double> &dst,
        Index axis);

template<typename T>
static void cpu_bias_avg_dev(void *buffers[], void *cl_args)
{
    Index m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    const Index mk = m * k;
    const T *avg_dev = reinterpret_cast<T *>(
            STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
//    Index dst_offset = 0;
//    for(Index i2 = 0; i2 < n; ++i2)
//    {
//        for(Index i1 = 0; i1 < k; ++i1)
//        {
//            Index src_offset = 3 * m * i2;
//            for(Index i0 = 0; i0 < m; ++i0)
//            {
//                T &val = dst[dst_offset];
//                const T &avg = avg_dev[src_offset];
//                const T &dev = avg_dev[src_offset+1];
//                val = (val-avg) / dev;
//                ++dst_offset;
//                src_offset += 3;
//            }
//        }
//    }
    Index src_offset = 0;
    for(Index i2 = 0; i2 < n; ++i2)
    {
        for(Index i1 = 0; i1 < m; ++i1)
        {
            T *dst_slice = dst + i2*mk + i1;
            const T &avg = avg_dev[src_offset], &dev = avg_dev[src_offset+1];
            const T inv_dev = T{1} / dev;
            const T frac = avg * inv_dev;
            src_offset += 2;
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T &val = dst_slice[i0*m];
                val = val*inv_dev - frac;
            }
        }
    }
}

template<typename T>
void bias_avg_dev_async(const Tile<T> &avg_dev, const Tile<T> &dst, Index axis)
{
    // StarPU codelet
    constexpr auto commute_mode = static_cast<enum starpu_data_access_mode>(
            STARPU_RW | STARPU_COMMUTE);
    static struct starpu_codelet codelet_bias_avg_dev =
    {
        .cpu_funcs = {cpu_bias_avg_dev<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, commute_mode}
    };
    // Check dimensions
    if(dst.ndim != avg_dev.ndim)
    {
        throw std::runtime_error("dst.ndim != avg_dev.ndim");
    }
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    if(avg_dev.shape[0] != 2)
    {
        throw std::runtime_error("avg_dev.shape[0] != 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != avg_dev.shape[i+1])
        {
            throw std::runtime_error("dst.shape[i] != avg_dev.shape[i+1]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != avg_dev.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
    }
    // Reshape inputs for simplicity: src -> (2,m,n), dst -> (m,k,n)
    Index m, n, k;
    if(axis == 0)
    {
        m = 1;
        n = avg_dev.nelems / 2; // 2 elements per single n
        k = dst.shape[0];
    }
    else if(axis == dst.ndim-1)
    {
        m = avg_dev.nelems / 2; // 2 elements per single m
        n = 1;
        k = dst.shape[axis];
    }
    else
    {
        m = dst.stride[axis];
        n = dst.matrix_shape[axis+1][1];
        k = dst.shape[axis];
    }
    // Insert corresponding task
    starpu_task_insert(&codelet_bias_avg_dev,
            STARPU_VALUE, &m, sizeof(m),
            STARPU_VALUE, &n, sizeof(n),
            STARPU_VALUE, &k, sizeof(k),
            STARPU_R, static_cast<starpu_data_handle_t>(avg_dev),
            commute_mode, static_cast<starpu_data_handle_t>(dst),
            STARPU_FLOPS, static_cast<double>(dst.nelems),
            0);
}

template
void bias_avg_dev_async(const Tile<float> &avg_dev, const Tile<float> &dst,
        Index axis);

template
void bias_avg_dev_async(const Tile<double> &avg_dev, const Tile<double> &dst,
        Index axis);

} // namespace nntile

