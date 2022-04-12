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
    size_t ndim;
    starpu_codelet_unpack_args(cl_args, &ndim, 0);
    unsigned long long seed;
    T mean, stddev;
    size_t nelems;
    std::vector<size_t> src_stride(ndim), randn_shape(ndim), dst_stride(ndim);
    starpu_codelet_unpack_args(cl_args, &ndim, &seed, &mean, &stddev,
            &(src_stride[0]), &(randn_shape[0]), &(dst_stride[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    size_t randn_nelems = 1;
    for(size_t i = 0; i < ndim; ++i)
    {
        randn_nelems *= randn_shape[i];
    }
    std::vector<size_t> index(ndim, 0);
    // View tile as a matrix of shape (shape[0], prod(shape[1:ndim]))
    size_t nrows = randn_shape[0], ncols = randn_nelems / nrows;
    for(size_t i = 0; i < nrows; ++i)
    {
        *dst = chameleon_randn(seed, mean, stddev);
        ++dst;
    }
    for(size_t j = 1; j < ncols; ++j)
    {
        ++index[1];
        size_t k = 1;
        size_t shift = src_stride[1] - nrows;
        dst += dst_stride[1] - nrows;
        while(index[k] == randn_shape[k])
        {
            index[k] = 0;
            ++k;
            ++index[k];
            shift += src_stride[k] - src_stride[k-1]*randn_shape[k-1];
            dst += dst_stride[k] - dst_stride[k-1]*randn_shape[k-1];
        }
        seed = CORE_rnd64_jump(shift, seed);
        for(size_t i = 0; i < nrows; ++i)
        {
            *dst = chameleon_randn(seed, mean, stddev);
            ++dst;
        }
    }
}

template<typename T>
void randn_async(const TileTraits &src, const Tile<T> &dst,
        const std::vector<size_t> &dst_coord, unsigned long long seed, T mean,
        T stddev)
{
    static struct starpu_codelet codelet_randn_rw =
    {
        .cpu_funcs = {cpu_chameleon_randn<T>},
        .nbuffers = 1,
        .modes = {STARPU_RW}
    };
    static struct starpu_codelet codelet_randn_w =
    {
        .cpu_funcs = {cpu_chameleon_randn<T>},
        .nbuffers = 1,
        .modes = {STARPU_W}
    };
    if(src.ndim != dst.ndim)
    {
        throw std::runtime_error("src.ndim != dst.ndim");
    }
    if(dst.ndim != dst_coord.size())
    {
        throw std::runtime_error("dst.ndim != dst_coord.size()");
    }
    size_t ndim = src.ndim;
    std::vector<size_t> randn_shape(dst.shape);
    bool full_overwrite = true;
    for(size_t i = 0; i < ndim; ++i)
    {
        // Do nothing if tiles do not intersect
        if(src.shape[i] <= dst_coord[i])
        {
            return;
        }
        size_t tmp = src.shape[i] - dst_coord[i];
        if(randn_shape[i] > tmp)
        {
            randn_shape[i] = tmp;
            full_overwrite = false;
        }
    }
    size_t jump = dst_coord[0]; // src.stride[0] = 1
    for(size_t i = 1; i < ndim; ++i)
    {
        jump += src.stride[i] * dst_coord[i];
    }
    seed = CORE_rnd64_jump(jump, seed);
    if(full_overwrite)
    {
        starpu_task_insert(&codelet_randn_w,
                STARPU_VALUE, &(ndim), sizeof(ndim),
                STARPU_VALUE, &seed, sizeof(seed),
                STARPU_VALUE, &mean, sizeof(mean),
                STARPU_VALUE, &stddev, sizeof(stddev),
                STARPU_VALUE, &(src.stride[0]), ndim*sizeof(src.stride[0]),
                STARPU_VALUE, &(randn_shape[0]), ndim*sizeof(randn_shape[0]),
                STARPU_VALUE, &(dst.stride[0]), ndim*sizeof(dst.stride[0]),
                STARPU_W, static_cast<starpu_data_handle_t>(dst),
                0);
    }
    else
    {
        starpu_task_insert(&codelet_randn_rw,
                STARPU_VALUE, &(ndim), sizeof(ndim),
                STARPU_VALUE, &seed, sizeof(seed),
                STARPU_VALUE, &mean, sizeof(mean),
                STARPU_VALUE, &stddev, sizeof(stddev),
                STARPU_VALUE, &(src.stride[0]), ndim*sizeof(src.stride[0]),
                STARPU_VALUE, &(randn_shape[0]), ndim*sizeof(randn_shape[0]),
                STARPU_VALUE, &(dst.stride[0]), ndim*sizeof(dst.stride[0]),
                STARPU_RW, static_cast<starpu_data_handle_t>(dst),
                0);
    }
}

template
void randn_async(const TileTraits &src, const Tile<float> &dst,
        const std::vector<size_t> &dst_coord, unsigned long long seed,
        float mean=0, float stddev=1);

template
void randn_async(const TileTraits &src, const Tile<double> &dst,
        const std::vector<size_t> &dst_coord, unsigned long long seed,
        double mean=0, double stddev=1);

} // namespace nntile

