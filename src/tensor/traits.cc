#include "nntile/tensor/traits.hh"

namespace nntile
{

std::ostream &operator<<(std::ostream &os, const TensorTraits &traits)
{
    os << "TensorTraits object at " << &traits << "\n";
    os << "shape=(";
    if(traits.ndim > 0)
    {
        os << traits.shape[0];
        for(size_t i = 1; i < traits.ndim; ++i)
        {
            os << "," << traits.shape[i];
        }
    }
    os << ")\n";
    os << "basetile_shape=(";
    if(traits.ndim > 0)
    {
        os << traits.basetile_shape[0];
        for(size_t i = 1; i < traits.ndim; ++i)
        {
            os << "," << traits.basetile_shape[i];
        }
    }
    os << ")\n";
    os << "leftover_shape=(";
    if(traits.ndim > 0)
    {
        os << traits.leftover_shape[0];
        for(size_t i = 1; i < traits.ndim; ++i)
        {
            os << "," << traits.leftover_shape[i];
        }
    }
    os << ")\n";
    os << "grid\n" << traits.grid;
    os << "Tiles\n";
    for(size_t i = 0; i < traits.grid.nelems; ++i)
    {
        const auto index = traits.get_tile_index(i);
        os << "  " << i << "\n";
        os << "    index=(";
        if(traits.ndim > 0)
        {
            os << index[0];
            for(size_t j = 1; j < traits.ndim; ++j)
            {
                os << "," << index[j];
            }
        }
        os << ")\n";
        const auto shape = traits.get_tile_shape(index);
        os << "    shape=(";
        if(traits.ndim > 0)
        {
            os << shape[0];
            for(size_t j = 1; j < traits.ndim; ++j)
            {
                os << "," << shape[j];
            }
        }
        os << ")\n";
    }
    return os;
}

} // namespace nntile

