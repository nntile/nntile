/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/randn.cc
 * Randn operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/randn.hh"
#include "random.h" // from external

namespace nntile
{

static inline float chameleon_randn(unsigned long long &seed, float mean,
        float stddev)
{
    return stddev*CORE_slaran(&seed) + mean;
}

static inline double chameleon_randn(unsigned long long &seed, double mean,
        double stddev)
{
    return stddev*CORE_dlaran(&seed) + mean;
}

template<typename T>
static void cpu_chameleon_randn_ndim0(void *buffers[], void *cl_args)
{
    unsigned long long seed;
    T mean, stddev;
    starpu_codelet_unpack_args(cl_args, &seed, &mean, &stddev);
    T *A = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    *A = chameleon_randn(seed, mean, stddev);
}

template<typename T>
static void cpu_chameleon_randn(void *buffers[], void *cl_args)
{
    Index ndim;
    starpu_codelet_unpack_args(cl_args, &ndim, 0);
    unsigned long long seed;
    T mean, stddev;
    Index nelems;
    std::vector<Index> underlying_stride(ndim), shape(ndim), stride(ndim);
    starpu_codelet_unpack_args(cl_args, &ndim, &nelems, &seed, &mean, &stddev,
            &(underlying_stride[0]), &(shape[0]), &(stride[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    // View tile as a matrix of shape (shape[0], prod(shape[1:ndim]))
    Index nrows = shape[0], ncols = nelems / nrows;
    for(Index i = 0; i < nrows; ++i)
    {
        *dst = chameleon_randn(seed, mean, stddev);
        ++dst;
    }
    std::vector<Index> index(ndim, 0);
    for(Index j = 1; j < ncols; ++j)
    {
        ++index[1];
        Index k = 1;
        Index shift = underlying_stride[1] - nrows;
        dst += stride[1] - nrows;
        while(index[k] == shape[k])
        {
            index[k] = 0;
            ++k;
            ++index[k];
            shift += underlying_stride[k] - underlying_stride[k-1]*shape[k-1];
            dst += stride[k] - stride[k-1]*shape[k-1];
        }
        seed = CORE_rnd64_jump(shift, seed);
        for(Index i = 0; i < nrows; ++i)
        {
            *dst = chameleon_randn(seed, mean, stddev);
            ++dst;
        }
    }
}

template<typename T>
void randn_async(const Tile<T> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, T mean, T stddev)
{
    static struct starpu_codelet codelet_randn =
    {
        .cpu_funcs = {cpu_chameleon_randn<T>},
        .nbuffers = 1,
        .modes = {STARPU_W}
    };
    static struct starpu_codelet codelet_randn_ndim0 =
    {
        .cpu_funcs = {cpu_chameleon_randn_ndim0<T>},
        .nbuffers = 1,
        .modes = {STARPU_W}
    };
    // Check inputs
    if(dst.ndim != offset.size())
    {
        throw std::runtime_error("dst.ndim != offset.size()");
    }
    if(dst.ndim != shape.size())
    {
        throw std::runtime_error("dst.ndim != shape.size()");
    }
    if(dst.ndim != stride.size())
    {
        throw std::runtime_error("dst.ndim != stride.size()");
    }
    // Treat special case of ndim=0
    if(dst.ndim == 0)
    {
        starpu_task_insert(&codelet_randn_ndim0,
                STARPU_VALUE, &seed, sizeof(seed),
                STARPU_VALUE, &mean, sizeof(mean),
                STARPU_VALUE, &stddev, sizeof(stddev),
                STARPU_W, static_cast<starpu_data_handle_t>(dst),
                0);
        return;
    }
    // Treat non-zero ndim
    if(offset[0] < 0)
    {
        throw std::runtime_error("offset[0] < 0");
    }
    if(offset[0]+dst.shape[0] > shape[0])
    {
        throw std::runtime_error("offset[0]+dst.shape[0] > shape[0]");
    }
    if(stride[0] != 1)
    {
        throw std::runtime_error("stride[0] != 1");
    }
    Index jump = offset[0]; // stride[0] = 1
    Index prod_shape = 1;
    for(Index i = 1; i < dst.ndim; ++i)
    {
        if(offset[i] < 0)
        {
            throw std::runtime_error("offset[i] < 0");
        }
        if(offset[i]+dst.shape[i] > shape[i])
        {
            throw std::runtime_error("offset[i]+dst.shape[i] > shape[i]");
        }
        prod_shape *= shape[i-1];
        if(stride[i] != prod_shape)
        {
            throw std::runtime_error("stride[i] != prod_shape");
        }
        jump += offset[i] * stride[i];
    }
    seed = CORE_rnd64_jump(jump, seed);
    starpu_task_insert(&codelet_randn,
            STARPU_VALUE, &(dst.ndim), sizeof(dst.ndim),
            STARPU_VALUE, &(dst.nelems), sizeof(dst.nelems),
            STARPU_VALUE, &seed, sizeof(seed),
            STARPU_VALUE, &mean, sizeof(mean),
            STARPU_VALUE, &stddev, sizeof(stddev),
            STARPU_VALUE, &(stride[0]), dst.ndim*sizeof(stride[0]),
            STARPU_VALUE, &(dst.shape[0]), dst.ndim*sizeof(dst.shape[0]),
            STARPU_VALUE, &(dst.stride[0]), dst.ndim*sizeof(dst.stride[0]),
            STARPU_W, static_cast<starpu_data_handle_t>(dst),
            0);
}

template
void randn_async(const Tile<float> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, float mean=0, float stddev=1);

template
void randn_async(const Tile<double> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, double mean=0, double stddev=1);

} // namespace nntile

