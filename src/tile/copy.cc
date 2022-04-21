#include "nntile/tile/copy.hh"

namespace nntile
{

template<typename T>
static void cpu_copy_intersection_ndim0(void *buffers[], void *cl_args)
{
    const T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    *dst = *src;
}

template<typename T>
static void cpu_copy_intersection(void *buffers[], void *cl_args)
{
    Index ndim;
    starpu_codelet_unpack_args(cl_args, &ndim, 0);
    std::vector<Index> src_start(ndim), src_stride(ndim), copy_shape(ndim),
        dst_start(ndim), dst_stride(ndim);
    starpu_codelet_unpack_args(cl_args, &ndim, &(src_start[0]),
            &(src_stride[0]), &(copy_shape[0]), &(dst_start[0]),
            &(dst_stride[0]));
    std::vector<Index> src_index(src_start), dst_index(dst_start);
    const T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    Index nelems = 1;
    for(Index i = 0; i < ndim; ++i)
    {
        nelems *= copy_shape[i];
    }
    Index src_offset = src_start[0]; // src_stride[0] = 1
    Index dst_offset = dst_start[0]; // src_stride[0] = 1
    for(Index i = 1; i < ndim; ++i)
    {
        src_offset += src_start[i] * src_stride[i];
        dst_offset += dst_start[i] * dst_stride[i];
    }
    dst[dst_offset] = src[src_offset];
    ++src_offset;
    ++dst_offset;
    for(Index i = 1; i < nelems; ++i)
    {
        ++src_index[0];
        ++dst_index[0];
        Index j = 0;
        while(src_index[j] == src_start[j]+copy_shape[j])
        {
            src_index[j] = src_start[j];
            ++j;
            ++src_index[j];
            src_offset += src_stride[j] - copy_shape[j-1]*src_stride[j-1];
        }
        j = 0;
        while(dst_index[j] == dst_start[j]+copy_shape[j])
        {
            dst_index[j] = dst_start[j];
            ++j;
            ++dst_index[j];
            dst_offset += dst_stride[j] - copy_shape[j-1]*dst_stride[j-1];
        }
        dst[dst_offset] = src[src_offset];
        ++src_offset;
        ++dst_offset;
    }
}

template<typename T>
void copy_intersection_async(const Tile<T> &src,
        const std::vector<Index> &src_offset, const Tile<T> &dst,
        const std::vector<Index> &dst_offset)
{
    static struct starpu_codelet codelet_copy_rw =
    {
        .cpu_funcs = {cpu_copy_intersection<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_RW}
    };
    static struct starpu_codelet codelet_copy_w =
    {
        .cpu_funcs = {cpu_copy_intersection<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_W}
    };
    static struct starpu_codelet codelet_copy_ndim0 =
    {
        .cpu_funcs = {cpu_copy_intersection_ndim0<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_W}
    };
    if(src.ndim != src_offset.size())
    {
        throw std::runtime_error("src.ndim != src_offset.size()");
    }
    if(src.ndim != dst.ndim)
    {
        throw std::runtime_error("src.ndim != dst.ndim");
    }
    if(dst.ndim != dst_offset.size())
    {
        throw std::runtime_error("dst.ndim != dst_offset.size()");
    }
    Index ndim = src.ndim;
    // Treat special case of ndim=0
    if(ndim == 0)
    {
        starpu_task_insert(&codelet_copy_ndim0,
                STARPU_R, static_cast<starpu_data_handle_t>(src),
                STARPU_W, static_cast<starpu_data_handle_t>(dst),
                0);
        return;
    }
    // Treat non-zero ndim
    std::vector<Index> src_start(ndim), dst_start(ndim), copy_shape(ndim);
    bool full_overwrite = true;
    for(Index i = 0; i < ndim; ++i)
    {
        // Do nothing if tiles do not intersect
        if((src_offset[i]+src.shape[i] <= dst_offset[i])
                or (dst_offset[i]+dst.shape[i] <= src_offset[i]))
        {
            return;
        }
        if(src_offset[i] < dst_offset[i])
        {
            dst_start[i] = 0;
            src_start[i] = dst_offset[i] - src_offset[i];
            copy_shape[i] = std::min(src.shape[i]-src_start[i],
                    dst.shape[i]);
        }
        else
        {
            dst_start[i] = src_offset[i] - dst_offset[i];
            src_start[i] = 0;
            copy_shape[i] = std::min(dst.shape[i]-dst_start[i],
                    src.shape[i]);
        }
        if(copy_shape[i] != dst.shape[i])
        {
            full_overwrite = false;
        }
    }
    if(full_overwrite)
    {
        starpu_task_insert(&codelet_copy_w,
                STARPU_VALUE, &(ndim), sizeof(ndim),
                STARPU_VALUE, &(src_start[0]), ndim*sizeof(src_start[0]),
                STARPU_VALUE, &(src.stride[0]), ndim*sizeof(src.stride[0]),
                STARPU_VALUE, &(copy_shape[0]), ndim*sizeof(copy_shape[0]),
                STARPU_VALUE, &(dst_start[0]), ndim*sizeof(dst_start[0]),
                STARPU_VALUE, &(dst.stride[0]), ndim*sizeof(dst.stride[0]),
                STARPU_R, static_cast<starpu_data_handle_t>(src),
                STARPU_W, static_cast<starpu_data_handle_t>(dst),
                0);
    }
    else
    {
        starpu_task_insert(&codelet_copy_rw,
                STARPU_VALUE, &(ndim), sizeof(ndim),
                STARPU_VALUE, &(src_start[0]), ndim*sizeof(src_start[0]),
                STARPU_VALUE, &(src.stride[0]), ndim*sizeof(src.stride[0]),
                STARPU_VALUE, &(copy_shape[0]), ndim*sizeof(copy_shape[0]),
                STARPU_VALUE, &(dst_start[0]), ndim*sizeof(dst_start[0]),
                STARPU_VALUE, &(dst.stride[0]), ndim*sizeof(dst.stride[0]),
                STARPU_R, static_cast<starpu_data_handle_t>(src),
                STARPU_RW, static_cast<starpu_data_handle_t>(dst),
                0);
    }
}

template
void copy_intersection_async(const Tile<float> &src,
        const std::vector<Index> &src_offset, const Tile<float> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async(const Tile<double> &src,
        const std::vector<Index> &src_offset, const Tile<double> &dst,
        const std::vector<Index> &dst_offset);

} // namespace nntile

