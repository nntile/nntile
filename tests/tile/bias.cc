#include "nntile/tile/bias.hh"
#include "nntile/tile/randn.hh"
#include "nntile/tile/copy.hh"

using namespace nntile;

template<typename T>
void check_bias(const Tile<T> &A, const Tile<T> &b, int batch_dim)
{
    Tile<T> B(A.shape);
    std::vector<size_t> index(B.ndim, 0);
    copy(A, index, B, index);
    bias(B, b, batch_dim);
    auto A_ptr = A.get_local_ptr(), B_ptr = B.get_local_ptr(),
         b_ptr = b.get_local_ptr();
    if(A_ptr[0]+b_ptr[0] != B_ptr[0])
    {
        throw std::runtime_error("A_ptr[0]+b_ptr[0] != B_ptr[0]");
    }
    for(size_t i = 1; i < B.nelems; ++i)
    {
        ++index[0];
        size_t j = 0;
        while(index[j] == B.shape[j])
        {
            index[j] = 0;
            ++j;
            ++index[j];
        }
        size_t b_offset = 0;
        for(size_t k = 0; k < batch_dim; ++k)
        {
            b_offset += index[k] * b.stride[k];
        }
        for(size_t k = batch_dim+1; k < B.ndim; ++k)
        {
            b_offset += index[k] * b.stride[k-1];
        }
        if(A_ptr[i]+b_ptr[b_offset] != B_ptr[i])
        {
            throw std::runtime_error("A_ptr[i]+b_ptr[b_offset] != B_ptr[i]");
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
    check_bias<T>(A, b0, 0);
}

int main(int argc, char **argv)
{
    StarPU starpu;
    validate_bias<float>();
    validate_bias<double>();
    return 0;
}

