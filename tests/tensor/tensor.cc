#include "nntile/tensor/tensor.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void check_tensor(const Tensor<T> &A)
{
    TESTN(A.get_tile(-1));
    TESTN(A.get_tile(A.grid.nelems));
    auto index = A.grid.shape;
    for(Index i = 0; i < A.ndim; ++i)
    {
        --index[i];
    }
    if(A.ndim > 0)
    {
        ++index[0];
        TESTN(A.get_tile(index));
        for(Index i = 1; i < A.ndim; ++i)
        {
            TESTN(A.get_tile(index));
        }
        TESTN(A.get_tile(std::vector<Index>(A.ndim-1)));
    }
    TESTN(A.get_tile(std::vector<Index>(A.ndim+1)));
    for(Index i = 0; i < A.grid.nelems; ++i)
    {
        const auto tile_index = A.grid.linear_to_index(i);
        TESTA(A.get_tile_shape(tile_index) == A.get_tile(tile_index).shape);
        TESTA(&A.get_tile_traits(tile_index) == &A.get_tile_traits(i));
        TESTA(&A.get_tile(tile_index) == &A.get_tile(i));
    }
}

template<typename T>
void validate_tensor()
{
    std::vector<Index> empty;
    Tensor<T> scalar(empty, empty);
    check_tensor<T>(scalar);
    Tensor<T> vector({10}, {3});
    check_tensor<T>(vector);
    Tensor<T> matrix({3, 5}, {3, 5});
    check_tensor<T>(matrix);
    Tensor<T> t5d({11, 13, 15, 17, 19}, {100, 100, 100, 100, 100});
    check_tensor<T>(t5d);
    Tensor<T> t5d2({40, 40, 40, 40, 40}, {11, 13, 15, 17, 19});
    check_tensor<T>(t5d2);
}

int main(int argc, char ** argv)
{
    Starpu starpu;
    validate_tensor<float>();
    return 0;
}

