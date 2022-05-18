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

// CPU codelet for bias operation with a single axis provided
template<typename T>
static
void cpu_bias(void *buffers[], void *cl_args)
    noexcept
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    Index m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    const Index mk = m * k;
    const T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
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

// Bias operation over single axis
template<typename T>
void bias_work(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    // StarPU codelet
    static struct starpu_codelet codelet_bias =
    {
        .cpu_funcs = {cpu_bias<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, Starpu::STARPU_RW_COMMUTE},
        .name = "bias"
    };
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
    int ret = starpu_task_insert(&codelet_bias,
            STARPU_VALUE, &m, sizeof(m),
            STARPU_VALUE, &n, sizeof(n),
            STARPU_VALUE, &k, sizeof(k),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            Starpu::STARPU_RW_COMMUTE, static_cast<starpu_data_handle_t>(dst),
            STARPU_FLOPS, static_cast<double>(dst.nelems),
            0);
    if(ret != 0)
    {
        throw std::runtime_error("ret != 0");
    }
}

// Explicit instantiation of template
template
void bias_work(const Tile<fp32_t> &src, const Tile<fp32_t> &dst, Index axis);

// Explicit instantiation of template
template
void bias_work(const Tile<fp64_t> &src, const Tile<fp64_t> &dst, Index axis);

// CPU codelet for normalization over single axis
template<typename T>
static
void cpu_bias_avg_dev(void *buffers[], void *cl_args)
    noexcept
{
    // Source (avg_dev) is a 2-by-m-by-n tile, which contains mean and
    // deviation values
    // Destination is an m-by-k-by-n tile
    // Both source and destination are Fortran-contiguous
    Index m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    const T *avg_dev = reinterpret_cast<T *>(
            STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    Index dst_offset = 0;
    // Outer loop by the last mode of source and destination tiles
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Middle loop by the middle mode of destination tile
        for(Index i1 = 0; i1 < k; ++i1)
        {
            Index src_offset = 2 * m * i2;
            // Inner loop by the first mode of source and destination tiles
            for(Index i0 = 0; i0 < m; ++i0)
            {
                // Value-to-update
                T &val = dst[dst_offset];
                // Corresponding mean and deviation
                const T &avg = avg_dev[src_offset];
                const T &dev = avg_dev[src_offset+1];
                // Normalization
                val = (val-avg) / dev;
                // Update pointers
                ++dst_offset;
                src_offset += 2;
            }
        }
    }
}

// CPU codelet for normalization over single axis if m=1
template<typename T>
static
void cpu_bias_avg_dev_m1(void *buffers[], void *cl_args)
    noexcept
{
    // Source (avg_dev) is a 2-by-1-by-n tile, which contains mean and
    // deviation values
    // Destination is an 1-by-k-by-n tile
    // Both source and destination are Fortran-contiguous
    Index n, k;
    starpu_codelet_unpack_args(cl_args, &n, &k);
    const T *avg_dev = reinterpret_cast<T *>(
            STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    Index dst_offset = 0;
    // Outer loop by the last mode of source and destination tiles
    for(Index i2 = 0; i2 < n; ++i2)
    {
        Index src_offset = 2 * i2;
        const T &avg = avg_dev[src_offset];
        const T &dev = avg_dev[src_offset+1];
        // Middle loop by the middle mode of destination tile
        for(Index i1 = 0; i1 < k; ++i1)
        {
            // No inner loop as m=1
            // Value-to-update
            T &val = dst[dst_offset];
            val = (val-avg) / dev;
            // Update pointer
            ++dst_offset;
        }
    }
}

// Normalization operation over single axis
template<typename T>
void bias_avg_dev_work(const Tile<T> &avg_dev, const Tile<T> &dst, Index axis)
{
    // StarPU codelet
    static struct starpu_codelet codelet_bias_avg_dev =
    {
        .cpu_funcs = {cpu_bias_avg_dev<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_RW},
        .name = "normalize"
    };
    static struct starpu_codelet codelet_bias_avg_dev_m1 =
    {
        .cpu_funcs = {cpu_bias_avg_dev_m1<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_RW},
        .name = "normalize m=1"
    };
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
    int ret;
    if(m == 1)
    {
        ret = starpu_task_insert(&codelet_bias_avg_dev_m1,
                STARPU_VALUE, &n, sizeof(n),
                STARPU_VALUE, &k, sizeof(k),
                STARPU_R, static_cast<starpu_data_handle_t>(avg_dev),
                STARPU_RW, static_cast<starpu_data_handle_t>(dst),
                STARPU_FLOPS, static_cast<double>(dst.nelems),
                0);
    }
    else
    {
        ret = starpu_task_insert(&codelet_bias_avg_dev,
                STARPU_VALUE, &m, sizeof(m),
                STARPU_VALUE, &n, sizeof(n),
                STARPU_VALUE, &k, sizeof(k),
                STARPU_R, static_cast<starpu_data_handle_t>(avg_dev),
                STARPU_RW, static_cast<starpu_data_handle_t>(dst),
                STARPU_FLOPS, static_cast<double>(dst.nelems),
                0);
    }
    if(ret != 0)
    {
        throw std::runtime_error("ret != 0");
    }
}

// Explicit instantiation of template
template
void bias_avg_dev_work(const Tile<fp32_t> &avg_dev, const Tile<fp32_t> &dst,
        Index axis);

// Explicit instantiation of template
template
void bias_avg_dev_work(const Tile<fp64_t> &avg_dev, const Tile<fp64_t> &dst,
        Index axis);

} // namespace nntile

