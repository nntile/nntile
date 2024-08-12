/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/forward.cc
 * This is a simple example with many matrix multiplications. It simulates
 * forward passes of a simple DeepLinear network.
 *
 * @version 1.1.0
 * */

#include <nntile.hh>
#include <chrono>

using namespace nntile;

Starpu starpu;

template<typename T>
void run_forward(const std::vector<Index> &shape,
        const std::vector<Index> &basetile_shape, Index nforward)
{
    // Init everything at first
    Index nlayers = shape.size()-2;
    std::vector<Tensor<T>> X, W, B;
    X.reserve(nlayers+1);
    W.reserve(nlayers);
    B.reserve(nlayers);
    unsigned long long seed = 1000000000000UL;
    std::vector<Index> X_shape={shape[0], shape[1]},
        X_basetile_shape={basetile_shape[0], basetile_shape[1]};
    X.emplace_back(X_shape, X_basetile_shape);
    for(Index i = 1; i <= nlayers; ++i)
    {
        std::vector<Index> W_shape{shape[i], shape[i+1]},
            W_basetile_shape{basetile_shape[i], basetile_shape[i+1]},
            X_shape{shape[0], shape[i+1]},
            X_basetile_shape{basetile_shape[0], basetile_shape[i+1]},
            B_shape{shape[i+1]},
            B_basetile_shape{basetile_shape[i+1]};
        W.emplace_back(W_shape, W_basetile_shape);
        randn(W[i-1], seed);
        seed = seed*seed + 1;
        X.emplace_back(X_shape, X_basetile_shape);
        B.emplace_back(B_shape, B_basetile_shape);
        randn(B[i-1], seed);
        seed = seed*seed + 1;
    }
    randn(X[0], seed);
    T one = 1, zero = 0;
    std::chrono::steady_clock clock;
    //starpu_profiling_init();
    starpu_profiling_worker_helper_display_summary();
    auto start = clock.now();
    for(Index i = 0; i < nforward; ++i)
    {
        //randn_async(X[0], seed);
        for(Index j = 0; j < nlayers; ++j)
        {
            gemm_async(one, TransOp::NoTrans, X[j], TransOp::NoTrans, W[j],
                    zero, X[j+1]);
            //X[j].wont_use();
            //W[j].wont_use();
            if(j > 0)
            {
                X[j].invalidate_submit();
            }
            W[j].wont_use();
            //bias_async(B[j], X[j+1], 0);
            //relu_async(X[j+1]);
        }
        //copy_intersection_async(X[nlayers], X[0]);
        //X[nlayers].wont_use();
        X[nlayers].invalidate_submit();
    }
    starpu_task_wait_for_all();
    starpu_profiling_worker_helper_display_summary();
    auto end = clock.now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Done in  " << diff.count() << " seconds\n";
    std::cout << "Gflops/s " << 2*shape[0]*shape[0]*shape[0]*1e-9*nlayers*nforward/diff.count() << "\n";
}

int main(int argc, char **argv)
{
    //Starpu starpu;
    std::vector<Index> shape{8192, 8192, 8192, 8192, 8192, 8192},
        basetile_shape{4000, 4000, 4000, 4000, 4000, 4000};
    //std::vector<Index> shape{4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096},
    //    basetile_shape{2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048};
    //    basetile_shape{1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    //    basetile_shape{500, 500, 500, 500, 500, 500, 500, 500};
    //std::vector<Index> shape{16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384},
    //    basetile_shape{4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096};
    Index nforward = 2;
    run_forward<fp32_t>(shape, basetile_shape, nforward);
    //run_forward<fp32_t>(shape, shape, nforward);
    return 0;
}
