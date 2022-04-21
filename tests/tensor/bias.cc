#include "nntile/tensor/bias.hh"
#include "nntile/tile/bias.hh"
#include "nntile/tensor/randn.hh"
#include "nntile/tensor/copy.hh"
#include "check_tensors_intersection.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void check_bias(const Tensor<T> &src, const Tensor<T> &dst, Index batch_dim)
{
    Tensor<T> src_local(src.shape, src.shape), dst_local(dst.shape, dst.shape);
    copy_intersection(src, src_local);
    copy_intersection(dst, dst_local);
    bias_async(src, dst, batch_dim);
    starpu_task_wait_for_all();
    bias(src_local.get_tile(0), dst_local.get_tile(0), batch_dim);
    TESTA(check_tensors_intersection(dst, dst_local));
    TESTA(check_tensors_intersection(dst_local, dst));
}

template<typename T>
void validate_bias()
{
    Tensor<T> A({3, 4, 5, 6}, {1, 2, 3, 4}), b0({4, 5, 6}, {2, 3, 4}),
        b1({3, 5, 6}, {1, 3, 4}), b2({3, 4, 6}, {1, 2, 4}),
        b3({3, 4, 5}, {1, 2, 3});
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
    TESTN(bias(b0, A, -1));
    TESTN(bias(Tensor<T>({}, {}), Tensor<T>({}, {}), 0));
    TESTN(bias(Tensor<T>({4, 5, 7}, {2, 3, 4}), A, 0));
    TESTN(bias(Tensor<T>({4, 5, 6}, {2, 3, 3}), A, 0));
    TESTN(bias(Tensor<T>({3, 4, 6}, {1, 2, 3}), A, 3));
    TESTN(bias(Tensor<T>({3, 4, 5}, {1, 2, 4}), A, 3));
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_bias<float>();
    validate_bias<double>();
    return 0;
}

