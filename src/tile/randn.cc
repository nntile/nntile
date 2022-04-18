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
    std::vector<size_t> underlying_stride(ndim), shape(ndim), stride(ndim);
    starpu_codelet_unpack_args(cl_args, &ndim, &nelems, &seed, &mean, &stddev,
            &(underlying_stride[0]), &(shape[0]), &(stride[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    // View tile as a matrix of shape (shape[0], prod(shape[1:ndim]))
    size_t nrows = shape[0], ncols = nelems / nrows;
    for(size_t i = 0; i < nrows; ++i)
    {
        *dst = chameleon_randn(seed, mean, stddev);
        ++dst;
    }
    std::vector<size_t> index(ndim, 0);
    for(size_t j = 1; j < ncols; ++j)
    {
        ++index[1];
        size_t k = 1;
        size_t shift = underlying_stride[1] - nrows;
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
        for(size_t i = 0; i < nrows; ++i)
        {
            *dst = chameleon_randn(seed, mean, stddev);
            ++dst;
        }
    }
}

template<typename T>
void randn_async(const Tile<T> &dst, unsigned long long seed, T mean, T stddev)
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
    size_t jump = dst.offset[0]; // dst.underlying_stride[0] = 1
    for(size_t i = 1; i < dst.ndim; ++i)
    {
        jump += dst.underlying_stride[i] * dst.offset[i];
    }
    seed = CORE_rnd64_jump(jump, seed);
    starpu_task_insert(&codelet_randn,
            STARPU_VALUE, &(dst.ndim), sizeof(dst.ndim),
            STARPU_VALUE, &(dst.nelems), sizeof(dst.nelems),
            STARPU_VALUE, &seed, sizeof(seed),
            STARPU_VALUE, &mean, sizeof(mean),
            STARPU_VALUE, &stddev, sizeof(stddev),
            STARPU_VALUE, &(dst.underlying_stride[0]),
            dst.ndim*sizeof(dst.underlying_stride[0]),
            STARPU_VALUE, &(dst.shape[0]), dst.ndim*sizeof(dst.shape[0]),
            STARPU_VALUE, &(dst.stride[0]), dst.ndim*sizeof(dst.stride[0]),
            STARPU_W, static_cast<starpu_data_handle_t>(dst),
            0);
}

template
void randn_async(const Tile<float> &dst, unsigned long long seed,
        float mean=0, float stddev=1);

template
void randn_async(const Tile<double> &dst, unsigned long long seed,
        double mean=0, double stddev=1);

} // namespace nntile

