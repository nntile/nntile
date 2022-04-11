#include <nntile/tensor/tensor.hh>

using namespace nntile;

template<typename T>
void validate_tensor()
{
    TensorTraits A1_traits({11, 12, 13, 14}, {5, 4, 6, 8}),
                 A1T_traits({14, 11, 12, 13}, {8, 5, 4, 6}),
                 B1_traits({14, 15, 16}, {8, 6, 4}),
                 B1T_traits({15, 16, 14}, {6, 4, 8}),
                 C1_traits({11, 12, 13, 15, 16}, {5, 4, 6, 6, 4}),
                 C1T_traits({15, 16, 11, 12, 13}, {6, 4, 5, 4, 6}),
                 A2_traits({8, 9, 10}, {2, 3, 4}),
                 A2T_traits({9, 10, 8}, {3, 4, 2}),
                 B2_traits({9, 10, 11, 12}, {3, 4, 5, 6}),
                 B2T_traits({11, 12, 9, 10}, {5, 6, 3, 4}),
                 C2_traits({8, 11, 12}, {2, 5, 6}),
                 C2T_traits({11, 12, 8}, {5, 6, 2});
    Tensor<T> A(A1_traits), B(A), C(Tensor<T>{A1_traits});
    // High-level API for tiles
    Tensor<T> A1(A1_traits),
        A1T(A1T_traits),
        B1(B1_traits),
        B1T(B1T_traits),
        C1(C1_traits),
        C1T(C1T_traits),
        A2(A2_traits),
        A2T(A2T_traits),
        B2(B2_traits),
        B2T(B2T_traits),
        C2(C2_traits),
        C2T(C2T_traits);
}

int main(int argc, char ** argv)
{
    StarPU starpu;
    validate_tensor<float>();
    return 0;
}

