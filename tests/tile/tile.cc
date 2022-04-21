#include "nntile/tile/tile.hh"
#include "nntile/base_types.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void check_tile(const std::vector<Index> &shape)
{
    // Check tile with allocation done by StarPU
    Tile<T> tile1(shape);
    TESTA(static_cast<starpu_data_handle_t>(tile1) != nullptr);
    tile1.acquire(STARPU_W);
    TESTA(tile1.get_local_ptr() != nullptr);
    tile1.release();
    // Check copy construction
    Tile<T> tile2(tile1);
    TESTA(static_cast<starpu_data_handle_t>(tile2) ==
            static_cast<starpu_data_handle_t>(tile1));
    // Check constructor from TileTraits
    Tile<T> tile3(static_cast<TileTraits>(tile2));
    TESTA(static_cast<starpu_data_handle_t>(tile3) != nullptr);
    TESTA(static_cast<starpu_data_handle_t>(tile2) !=
            static_cast<starpu_data_handle_t>(tile3));
    tile2.acquire(STARPU_R);
    tile3.acquire(STARPU_W);
    TESTA(tile2.get_local_ptr() != tile3.get_local_ptr());
    tile2.release();
    tile3.release();
    // Check with shape and pointer
    T *ptr = new T[tile1.nelems];
    for(Index i = 0; i < tile1.nelems; ++i)
    {
        ptr[i] = static_cast<T>(i);
    }
    TESTN(Tile<T>(shape, ptr, tile1.nelems-1));
    Tile<T> tile4(tile1, ptr, tile1.nelems);
    TESTA(static_cast<starpu_data_handle_t>(tile4) != nullptr);
    TESTN(tile4.at_linear(-1));
    TESTN(tile4.at_linear(tile4.nelems));
    for(Index i = 0; i < tile4.nelems; ++i)
    {
        T value = static_cast<T>(i);
        const auto index = tile4.linear_to_index(i);
        TESTA(value == tile4.at_linear(i));
        TESTA(value == tile4.at_index(index));
    }
    // Check with TileTraits and pointer
    TESTN(Tile<T>(tile4, ptr, tile4.nelems-1));
    Tile<T> tile5(tile4, ptr, tile4.nelems);
    TESTA(static_cast<starpu_data_handle_t>(tile5) != nullptr);
    TESTA(static_cast<starpu_data_handle_t>(tile5) !=
            static_cast<starpu_data_handle_t>(tile4));
    TESTA(tile4.get_local_ptr() == tile5.get_local_ptr());
    TESTN(tile5.at_linear(-1));
    TESTN(tile5.at_linear(tile5.nelems));
    for(Index i = 0; i < tile5.nelems; ++i)
    {
        T value = static_cast<T>(i);
        const auto index = tile5.linear_to_index(i);
        TESTA(value == tile5.at_linear(i));
        TESTA(value == tile5.at_index(index));
    }
    delete[] ptr;
}

template<typename T>
void validate_tile()
{
    check_tile<T>({});
    check_tile<T>({3});
    check_tile<T>({3, 2});
    check_tile<T>({3, 2, 1});
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_tile<fp64_t>();
    validate_tile<fp32_t>();
    return 0;
}

