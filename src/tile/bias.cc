#include "nntile/tile/bias.hh"

namespace nntile
{

template<typename T>
static void cpu_bias(void *buffers[], void *cl_args)
{
    size_t m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    const size_t mk = m * k;
    T *data = reinterpret_cast<T *>(STARPU_VECTOR_GET_PTR(buffers[0]));
    const T *bias = reinterpret_cast<T *>(STARPU_VECTOR_GET_PTR(buffers[1]));
    int data_offset = 0;
    for(size_t i2 = 0; i2 < n; ++i2)
    {
        for(size_t i1 = 0; i1 < k; ++i1)
        {
            size_t bias_offset = i2 * m;
            for(size_t i0 = 0; i0 < m; ++i0)
            {
                data[data_offset] += bias[bias_offset];
                ++data_offset;
                ++bias_offset;
            }
        }
    }
}

template<typename T>
void bias_async(const Tile<T> &A, const Tile<T> &bias, int batch_dim)
{
    static struct starpu_codelet codelet_bias =
    {
        .cpu_funcs = {cpu_bias<T>},
        .nbuffers = 2,
        .modes = {STARPU_RW, STARPU_R}
    };
    if(A.ndim != bias.ndim+1)
    {
        throw std::runtime_error("A.ndim != bias.ndim+1");
    }
    if(batch_dim < 0)
    {
        throw std::runtime_error("batch_dim < 0");
    }
    for(size_t i = 0; i < batch_dim; ++i)
    {
        if(A.shape[i] != bias.shape[i])
        {
            throw std::runtime_error("A.shape[i] != bias.shape[i]");
        }
    }
    for(size_t i = batch_dim+1; i < A.ndim; ++i)
    {
        if(A.shape[i] != bias.shape[i-1])
        {
            throw std::runtime_error("A.shape[i] != bias.shape[i-1]");
        }
    }
    size_t m, n, k;
    if(batch_dim == 0)
    {
        m = 1;
        n = bias.nelems;
        k = A.shape[0];
    }
    else if(batch_dim == A.ndim-1)
    {
        m = bias.nelems;
        n = 1;
        k = A.shape[A.ndim-1];
    }
    else
    {
        m = A.stride[batch_dim];
        n = A.matrix_shape[batch_dim][1];
        k = A.shape[batch_dim];
    }
    // Check that matrix sizes fit proper types for underlying CBLAS
#if defined(NNTILE_USE_CBLAS)
    if(static_cast<CBLAS_INT>(m) != m)
    {
        throw std::runtime_error("GEMM size M does not fit CBLAS_INT");
    }
    if(static_cast<CBLAS_INT>(n) != n)
    {
        throw std::runtime_error("GEMM size N does not fit CBLAS_INT");
    }
    if(static_cast<CBLAS_INT>(k) != k)
    {
        throw std::runtime_error("GEMM size K does not fit CBLAS_INT");
    }
#endif
    // Check that matrix sizes fit proper types for underlying CUBLAS
#if defined(NNTILE_USE_CUBLAS)
    if(static_cast<int>(m) != m)
    {
        throw std::runtime_error("GEMM size M does not fit int");
    }
    if(static_cast<int>(n) != n)
    {
        throw std::runtime_error("GEMM size N does not fit int");
    }
    if(static_cast<int>(k) != k)
    {
        throw std::runtime_error("GEMM size K does not fit int");
    }
#endif
    starpu_task_insert(&codelet_bias,
            STARPU_VALUE, &m, sizeof(m),
            STARPU_VALUE, &n, sizeof(n),
            STARPU_VALUE, &k, sizeof(k),
            STARPU_RW, static_cast<starpu_data_handle_t>(A),
            STARPU_R, static_cast<starpu_data_handle_t>(bias),
            0);
}

template
void bias_async(const Tile<float> &A, const Tile<float> &bias,
        int batch_dim);

template
void bias_async(const Tile<double> &A, const Tile<double> &bias,
        int batch_dim);

} // namespace nntile

