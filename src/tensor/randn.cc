#include "nntile/tensor/randn.hh"
#include "nntile/tile/randn.hh"

namespace nntile
{

template<typename T>
void randn_async(const Tensor<T> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, T mean, T stddev)
{
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
        randn_async(dst.get_tile(0), offset, shape, stride, seed, mean,
                stddev);
        return;
    }
    // Treat non-zero ndim (continue checking inputs)
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
    }
    // Now do the job
    std::vector<Index> tile_offset(offset), tile_index(dst.ndim);
    randn_async(dst.get_tile(0), offset, shape, stride, seed, mean, stddev);
    for(Index i = 1; i < dst.grid.nelems; ++i)
    {
        ++tile_index[0];
        tile_offset[0] += dst.basetile_shape[0];
        Index j = 0;
        while(tile_index[j] == dst.grid.shape[j])
        {
            tile_index[j] = 0;
            tile_offset[j] = offset[j];
            ++j;
            ++tile_index[j];
            tile_offset[j] += dst.basetile_shape[j];
        }
        randn_async(dst.get_tile(i), tile_offset, shape, stride, seed, mean,
                stddev);
    }
}

template
void randn_async(const Tensor<float> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, float mean=0, float stddev=1);

template
void randn_async(const Tensor<double> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, double mean=0, double stddev=1);

} // namespace nntile

