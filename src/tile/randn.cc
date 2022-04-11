#include "nntile/tile/randn.hh"
#include "random.h" // from external

namespace nntile
{

static inline float chameleon_randn(unsigned long long &seed, float mean,
        float stddev)
{
    return stddev*CORE_slaran(&seed)+mean;
}

static inline double chameleon_randn(unsigned long long &seed, double mean,
        double stddev)
{
    return stddev*CORE_dlaran(&seed)+mean;
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
    std::vector<size_t> shape(ndim), offset(ndim), stride(ndim);
    starpu_codelet_unpack_args(cl_args, &ndim, &seed, &mean, &stddev,
            &nelems, &(shape[0]), &(offset[0]), &(stride[0]));
    T *A = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    std::vector<size_t> index(ndim, 0);
    size_t global_offset = offset[0]; // stride[0] = 1
    for(size_t i = 1; i < ndim; ++i)
    {
        global_offset += offset[i] * stride[i];
    }
    seed = CORE_rnd64_jump(global_offset, seed);
    // View tile as a matrix of shape (shape[0], prod(shape[1:ndim]))
    size_t nrows = shape[0], ncols = nelems / nrows;
    for(size_t i = 0; i < nrows; ++i)
    {
        *A = chameleon_randn(seed, mean, stddev);
        //*A = global_offset;
        ++A;
        ++global_offset;
    }
    for(size_t j = 1; j < ncols; ++j)
    {
        size_t k = 1;
        ++index[k];
        size_t shift = stride[1] - shape[0];
        while(index[k] == shape[k])
        {
            index[k] = 0;
            ++k;
            ++index[k];
            shift += stride[k] - shape[k-1]*stride[k-1];
        }
        seed = CORE_rnd64_jump(shift, seed);
        global_offset += shift;
        for(size_t i = 0; i < nrows; ++i)
        {
            *A = chameleon_randn(seed, mean, stddev);
            //*A = global_offset;
            ++A;
            ++global_offset;
        }
    }
}

template<typename T>
void randn_async(const Tile<T> &A, const std::vector<size_t> &offset,
        const std::vector<size_t> &stride, unsigned long long &seed,
        T mean, T stddev)
{
    static struct starpu_codelet codelet_randn =
    {
        .cpu_funcs = {cpu_chameleon_randn<T>},
        .nbuffers = 1,
        .modes = {STARPU_W}
    };
    if(A.ndim != offset.size())
    {
        throw std::runtime_error("A.ndim != offset.size()");
    }
    if(A.ndim != stride.size())
    {
        throw std::runtime_error("A.ndim != stride.size()");
    }
    starpu_task_insert(&codelet_randn,
            STARPU_VALUE, &(A.ndim), sizeof(A.ndim),
            STARPU_VALUE, &seed, sizeof(seed),
            STARPU_VALUE, &mean, sizeof(mean),
            STARPU_VALUE, &stddev, sizeof(stddev),
            STARPU_VALUE, &(A.nelems), sizeof(A.nelems),
            STARPU_VALUE, &(A.shape[0]), A.ndim*sizeof(A.shape[0]),
            STARPU_VALUE, &(offset[0]), A.ndim*sizeof(offset[0]),
            STARPU_VALUE, &(stride[0]), A.ndim*sizeof(stride[0]),
            STARPU_W, static_cast<starpu_data_handle_t>(A),
            0);
    seed = CORE_rnd64_jump(A.nelems, seed);
}

template
void randn_async(const Tile<float> &A, const std::vector<size_t> &offset,
        const std::vector<size_t> &stride, unsigned long long &seed,
        float mean, float stddev);

template
void randn_async(const Tile<double> &A, const std::vector<size_t> &offset,
        const std::vector<size_t> &stride, unsigned long long &seed,
        double mean, double stddev);

} // namespace nntile

