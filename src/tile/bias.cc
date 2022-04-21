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
    Index dst_offset = 0;
    for(Index i2 = 0; i2 < n; ++i2)
    {
        for(Index i1 = 0; i1 < k; ++i1)
        {
            Index src_offset = i2 * m;
            for(Index i0 = 0; i0 < m; ++i0)
            {
                dst[dst_offset] += src[src_offset];
                ++dst_offset;
                ++src_offset;
            }
        }
    }
}

template<typename T>
void bias_async(const Tile<T> &src, const Tile<T> &dst, Index batch_dim)
{
    constexpr auto commute_mode = static_cast<enum starpu_data_access_mode>(
            STARPU_RW | STARPU_COMMUTE);
    static struct starpu_codelet codelet_bias =
    {
        .cpu_funcs = {cpu_bias<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, commute_mode}
    };
    if(dst.ndim != src.ndim+1)
    {
        throw std::runtime_error("dst.ndim != src.ndim+1");
    }
    if(batch_dim < 0)
    {
        throw std::runtime_error("batch_dim < 0");
    }
    for(Index i = 0; i < batch_dim; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
    }
    for(Index i = batch_dim+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i-1])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i-1]");
        }
    }
    Index m, n, k;
    if(batch_dim == 0)
    {
        m = 1;
        n = src.nelems;
        k = dst.shape[0];
    }
    else if(batch_dim == dst.ndim-1)
    {
        m = src.nelems;
        n = 1;
        k = dst.shape[batch_dim];
    }
    else
    {
        m = dst.stride[batch_dim];
        n = dst.matrix_shape[batch_dim+1][1];
        k = dst.shape[batch_dim];
    }
    starpu_task_insert(&codelet_bias,
            STARPU_VALUE, &m, sizeof(m),
            STARPU_VALUE, &n, sizeof(n),
            STARPU_VALUE, &k, sizeof(k),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            commute_mode, static_cast<starpu_data_handle_t>(dst),
            0);
}

template
void bias_async(const Tile<float> &src, const Tile<float> &dst,
        Index batch_dim=1);

template
void bias_async(const Tile<double> &src, const Tile<double> &dst,
        Index batch_dim=1);

} // namespace nntile

