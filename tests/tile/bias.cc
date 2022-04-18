#include "nntile/tile/bias.hh"
#include "nntile/tile/randn.hh"
#include "nntile/tile/copy.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void check_bias(const Tile<T> &src, const Tile<T> &dst, int batch_dim)
{
    Tile<T> res(TileTraits(dst.shape));
    std::vector<size_t> index(dst.ndim, 0);
    copy_intersection(dst, index, res, index);
    bias(src, res, batch_dim);
    src.acquire(STARPU_R);
    dst.acquire(STARPU_R);
    res.acquire(STARPU_R);
    auto src_ptr = src.get_local_ptr(), dst_ptr = dst.get_local_ptr(),
         res_ptr = res.get_local_ptr();
    if(src_ptr[0]+dst_ptr[0] != res_ptr[0])
    {
        throw std::runtime_error("src_ptr[0]+dst_ptr[0] != res_ptr[0]");
    }
    for(size_t i = 1; i < dst.nelems; ++i)
    {
        ++index[0];
        size_t j = 0;
        while(index[j] == dst.shape[j])
        {
            index[j] = 0;
            ++j;
            ++index[j];
        }
        size_t src_offset = 0;
        for(size_t k = 0; k < batch_dim; ++k)
        {
            src_offset += index[k] * src.stride[k];
        }
        for(size_t k = batch_dim+1; k < dst.ndim; ++k)
        {
            src_offset += index[k] * src.stride[k-1];
        }
        if(src_ptr[src_offset]+dst_ptr[i] != res_ptr[i])
        {
            src.release();
            dst.release();
            res.release();
            throw std::runtime_error("src_ptr[src_offset]+dst_ptr[i] != "
                    "dst_ptr[i]");
        }
    }
    src.release();
    dst.release();
    res.release();
}

template<typename T>
void validate_bias()
{
    Tile<T> A({{3, 4, 5, 6}}), b0({{4, 5, 6}}), b1({{3, 5, 6}}),
        b2({{3, 4, 6}}), b3({{3, 4, 5}});
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
    Tile<T> C({{3}});
    TESTN(bias(C, A, 0));
    TESTN(bias(b2, A, 3));
    TESTN(bias(b0, A, -1));
    Tile<T> fail_b0({{4, 5, 5}});
    TESTN(bias(fail_b0, A, 0));
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_bias<float>();
    validate_bias<double>();
    return 0;
}

