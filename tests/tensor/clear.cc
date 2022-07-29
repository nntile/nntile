#include "nntile/tensor/clear.hh"
#include "../testing.hh"

using namespace nntile;

Starpu starpu;

template<typename T>
void validate_clear()
{
    Starpu::pause();
    Tensor<T> A({4, 5, 6, 7}, {1, 2, 3, 4});
    for(Index i = 0; i < A.grid.nelems; ++i)
    {
        auto tile = A.get_tile(i);
        auto tile_local = tile.acquire(STARPU_W);
        for(Index j = 0; j < tile.nelems; ++j)
        {
            tile_local[j] = T(i+j+1);
        }
        tile_local.release();
    }
    Starpu::resume();
    clear(A);
    Starpu::pause();
    constexpr T zero = 0;
    for(Index i = 0; i < A.grid.nelems; ++i)
    {
        auto tile = A.get_tile(i);
        auto tile_local = tile.acquire(STARPU_R);
        for(Index j = 0; j < tile.nelems; ++j)
        {
            if(tile_local[j] != zero)
            {
                throw std::runtime_error("Data is not zero");
            }
        }
        tile_local.release();
    }
    Starpu::resume();
}

int main(int argc, char **argv)
{
    validate_clear<fp32_t>();
    validate_clear<fp64_t>();
    return 0;
}

