#include "nntile/tile/traits.hh"

namespace nntile
{

std::ostream &operator<<(std::ostream &os, const TileTraits &traits)
{
    os << "TileTraits object at " << &traits << "\n";
    os << "ndim=" << traits.ndim << "\n";
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
    os << "stride=(";
    if(traits.ndim > 0)
    {
        os << traits.stride[0];
        for(size_t i = 1; i < traits.ndim; ++i)
        {
            os << "," << traits.stride[i];
        }
    }
    os << ")\n";
    os << "nelems=" << traits.nelems << "\n";
    os << "matrix_shape=((" << traits.matrix_shape[0][0] <<
        "," << traits.matrix_shape[0][1] << ")";
    for(size_t i = 1; i <= traits.ndim; ++i)
    {
        os << ",(" << traits.matrix_shape[i][0] << "," <<
            traits.matrix_shape[i][1] << ")";
    }
    os << ")\n";
    return os;
}


} // namespace nntile

