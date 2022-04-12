#include "nntile/tensor/randn.hh"
#include "nntile/tile/randn.hh"

namespace nntile
{

template<typename T>
void randn_async(const TileTraits &src, const Tensor<T> &dst,
        const std::vector<size_t> &dst_coord, unsigned long long seed, T mean,
        T stddev)
{
    std::vector<size_t> tile_coord(dst_coord), index(dst.ndim, 0);
    randn_async(src, dst.get_tile(0), tile_coord, seed, mean, stddev);
    for(size_t i = 1; i < dst.grid.nelems; ++i)
    {
        ++index[0];
        tile_coord[0] += dst.basetile_shape[0];
        size_t j = 0;
        while(index[j] == dst.grid.shape[j])
        {
            index[j] = 0;
            tile_coord[j] = dst_coord[j];
            ++j;
            ++index[j];
            tile_coord[j] += dst.basetile_shape[j];
        }
        randn_async(src, dst.get_tile(i), tile_coord, seed, mean, stddev);
    }
}

template
void randn_async(const TileTraits &src, const Tensor<float> &dst,
        const std::vector<size_t> &dst_coord, unsigned long long seed,
        float mean=0, float stddev=1);

template
void randn_async(const TileTraits &src, const Tensor<double> &dst,
        const std::vector<size_t> &dst_coord, unsigned long long seed,
        double mean=0, double stddev=1);

} // namespace nntile

