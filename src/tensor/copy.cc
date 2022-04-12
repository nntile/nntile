#include "nntile/tensor/copy.hh"
#include "nntile/tile/copy.hh"

namespace nntile
{

template<typename T>
void copy_async(const Tensor<T> &src, const std::vector<size_t> &src_coord,
        const Tensor<T> &dst, const std::vector<size_t> &dst_coord)
{
    std::vector<size_t> src_offset(src_coord), dst_offset(dst_coord),
        vzero(src.ndim, 0), src_index(vzero), dst_index(vzero);
    copy_async(src.get_tile(0), src_offset, dst.get_tile(0), dst_offset);
    for(size_t k = 1; k < dst.grid.nelems; ++k)
    {
        ++dst_index[0];
        size_t l = 0;
        while(dst_index[l] == dst.shape[l])
        {
            dst_index[l] = 0;
            dst_offset[l] = dst_coord[l];
            ++l;
            ++dst_index[l];
            dst_offset[l] += dst.basetile_shape[l];
        }
        copy_async(src.get_tile(0), src_offset, dst.get_tile(k), dst_offset);
    }
    for(size_t i = 1; i < src.grid.nelems; ++i)
    {
        dst_index = vzero;
        dst_offset = dst_coord;
        ++src_index[0];
        size_t j = 0;
        while(src_index[j] == src.shape[j])
        {
            src_index[j] = 0;
            src_offset[j] = src_coord[j];
            ++j;
            ++src_index[j];
            src_offset[j] += src.basetile_shape[j];
        }
        copy_async(src.get_tile(i), src_offset, dst.get_tile(0), dst_offset);
        for(size_t k = 1; k < dst.grid.nelems; ++k)
        {
            ++dst_index[0];
            size_t l = 0;
            while(dst_index[l] == dst.shape[l])
            {
                dst_index[l] = 0;
                dst_offset[l] = dst_coord[l];
                ++l;
                ++dst_index[l];
                dst_offset[l] += dst.basetile_shape[l];
            }
            copy_async(src.get_tile(i), src_offset, dst.get_tile(k),
                    dst_offset);
        }
    }
}

template
void copy_async(const Tensor<float> &src,
        const std::vector<size_t> &src_coord, const Tensor<float> &dst,
        const std::vector<size_t> &dst_coord);

template
void copy_async(const Tensor<double> &src,
        const std::vector<size_t> &src_coord, const Tensor<double> &dst,
        const std::vector<size_t> &dst_coord);

} // namespace nntile

