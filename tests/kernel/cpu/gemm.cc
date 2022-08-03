/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/cpu/gemm.cc
 * Gemm operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-03
 * */

#include "nntile/kernel/cpu/gemm.hh"
#include "nntile/kernel/args/gemm.hh"
#include "nntile/starpu.hh"
#include <vector>
#include <stdexcept>
#include <numeric>

using namespace nntile;

// Templated validation
template<typename T>
void validate()
{
    // Init test input
    // A = [[0, 2, 4]      B = [[0, 3, 6,  9]      C = [[0, 2, 4, 6]
    //      [1, 3, 5]]          [1, 4, 7, 10]           [1, 3, 5, 7]]
    //                          [2, 5, 8, 11]]
    std::vector<T> A(6), B(12), C(8);
    std::iota(A.begin(), A.end(), 0);
    std::iota(B.begin(), B.end(), 0);
    std::iota(C.begin(), C.end(), 0);
    std::vector<T> C2(C);
    // StarPU interfaces
    StarpuVariableInterface A_interface(&A[0], 6*sizeof(T)),
            B_interface(&B[0], 12*sizeof(T)), C_interface(&C[0], 8*sizeof(T));
    void *buffers[3] = {&A_interface, &B_interface, &C_interface};
    // Codelet arguments
    gemm_starpu_args<T> args =
    {
        .transA = TransOp::NoTrans,
        .transB = TransOp::NoTrans,
        .m = 2,
        .n = 4,
        .k = 3,
        .alpha = T{2},
        .beta = T{-1}
    };
    // Launch codelet
    gemm_starpu_cpu<T>(buffers, &args);
    // Check it
    std::vector<T> D{20, 25, 54, 77, 88, 129, 122, 181};
    for(Index i = 0; i < 8; ++i)
    {
        if(C[i] != D[i])
        {
            throw std::runtime_error("Starpu codelet NN wrong result");
        }
    }
    // Check transpose A
    std::vector<T> AT{0, 2, 4, 1, 3, 5};
    args.transA = TransOp::Trans;
    StarpuVariableInterface AT_interface(&AT[0], 6*sizeof(T));
    buffers[0] = &AT_interface;
    std::memcpy(&C[0], &C2[0], 8*sizeof(T));
    // Launch codelet
    gemm_starpu_cpu<T>(buffers, &args);
    // Check it
    for(Index i = 0; i < 8; ++i)
    {
        if(C[i] != D[i])
        {
            throw std::runtime_error("Starpu codelet TN wrong result");
        }
    }
    // Check transpose B
    std::vector<T> BT{0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    args.transA = TransOp::NoTrans;
    args.transB = TransOp::Trans;
    StarpuVariableInterface BT_interface(&BT[0], 12*sizeof(T));
    buffers[0] = &A_interface;
    buffers[1] = &BT_interface;
    std::memcpy(&C[0], &C2[0], 8*sizeof(T));
    // Launch codelet
    gemm_starpu_cpu<T>(buffers, &args);
    // Check it
    for(Index i = 0; i < 8; ++i)
    {
        if(C[i] != D[i])
        {
            throw std::runtime_error("Starpu codelet TN wrong result");
        }
    }
    // Check transpose A and B
    args.transA = TransOp::Trans;
    buffers[0] = &AT_interface;
    std::memcpy(&C[0], &C2[0], 8*sizeof(T));
    // Launch codelet
    gemm_starpu_cpu<T>(buffers, &args);
    // Check it
    for(Index i = 0; i < 8; ++i)
    {
        if(C[i] != D[i])
        {
            throw std::runtime_error("Starpu codelet TT wrong result");
        }
    }
}

int main(int argc, char **argv)
{
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

