#include "nntile/tensor/gelu.hh"
#include "nntile/tile/gelu.hh"

namespace nntile
{

template<typename T>
void gelu_async(const Tensor<T> &A)
{
    for(Index i = 0; i < A.grid.nelems; ++i)
    {
        gelu_async(A.get_tile(i));
    }
}

template
void gelu_async(const Tensor<float> &A);

template
void gelu_async(const Tensor<double> &A);

} // namespace nntile

