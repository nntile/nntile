#include "nntile/tile/bias.hh"
#include "nntile/tile/randn.hh"
#include "nntile/tile/copy.hh"
#include "../testing.hh"
#include <iomanip>

using namespace nntile;

template<typename T>
void check_bias(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    Tile<T> res(TileTraits(dst.shape));
    std::vector<Index> index(dst.ndim, 0);
    copy(dst, index, res, index);
    bias(src, res, axis);
    auto src_local = src.acquire(STARPU_R), dst_local = dst.acquire(STARPU_R),
         res_local = res.acquire(STARPU_R);
    if(src_local[0]+dst_local[0] != res_local[0])
    {
        throw std::runtime_error("src_local[0]+dst_local[0] != res_local[0]");
    }
    for(Index i = 1; i < dst.nelems; ++i)
    {
        ++index[0];
        Index j = 0;
        while(index[j] == dst.shape[j])
        {
            index[j] = 0;
            ++j;
            ++index[j];
        }
        Index src_offset = 0;
        for(Index k = 0; k < axis; ++k)
        {
            src_offset += index[k] * src.stride[k];
        }
        for(Index k = axis+1; k < dst.ndim; ++k)
        {
            src_offset += index[k] * src.stride[k-1];
        }
        if(src_local[src_offset]+dst_local[i] != res_local[i])
        {
            throw std::runtime_error("src_local[src_offset]+dst_local[i] != "
                    "dst_local[i]");
        }
    }
}

template<typename T>
void validate_bias()
{
    Tile<T> A({3, 4, 5, 6}), b0({4, 5, 6}), b1({3, 5, 6}), b2({3, 4, 6}),
        b3({3, 4, 5});
    unsigned long long A_seed = 100, b0_seed = 101, b1_seed = 102,
                  b2_seed = 103, b3_seed = 104;
    randn(A, A_seed);
    randn(b0, b0_seed);
    randn(b1, b1_seed);
    randn(b2, b2_seed);
    randn(b3, b3_seed);
    check_bias<T>(b0, A, 0);
    check_bias<T>(b1, A, 1);
    check_bias<T>(b2, A, 2);
    check_bias<T>(b3, A, 3);
    Tile<T> C({3});
    TESTN(bias(C, A, 0));
    TESTN(bias(b0, A, 1));
    TESTN(bias(b0, A, 2));
    TESTN(bias(b0, A, 3));
    TESTN(bias(b1, A, 0));
    TESTN(bias(b1, A, 2));
    TESTN(bias(b1, A, 3));
    TESTN(bias(b2, A, 0));
    TESTN(bias(b2, A, 1));
    TESTN(bias(b2, A, 3));
    TESTN(bias(b3, A, 0));
    TESTN(bias(b3, A, 1));
    TESTN(bias(b3, A, 2));
    TESTN(bias(b0, A, -1));
    TESTN(bias(b0, A, 4));
    Tile<T> fail_b0({4, 5, 5});
    TESTN(bias(fail_b0, A, 0));
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_bias<fp32_t>();
    validate_bias<fp64_t>();
    return 0;
}

