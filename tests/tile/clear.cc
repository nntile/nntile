#include "nntile/tile/clear.hh"
#include "../testing.hh"

using namespace nntile;

Starpu starpu;

template<typename T>
void validate_clear()
{
    Starpu::pause();
    TileTraits traits({4, 5, 6, 7});
    std::vector<T> buf(traits.nelems);
    for(Index i = 0; i < traits.nelems; ++i)
    {
        buf[i] = T(i+1);
    }
    Tile<T> A(traits, &buf[0], traits.nelems);
    Starpu::resume();
    clear(A);
    Starpu::pause();
    auto A_local = A.acquire(STARPU_R);
    constexpr T zero = 0;
    for(Index i = 0; i < traits.nelems; ++i)
    {
        if(A_local[i] != zero)
        {
            throw std::runtime_error("Data is not zero");
        }
    }
    Starpu::resume();
}

int main(int argc, char **argv)
{
    validate_clear<fp32_t>();
    validate_clear<fp64_t>();
    return 0;
}

