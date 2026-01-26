/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/flash_sdpa_fwd_cudnn.cc
 * Test for compiled graph flash_sdpa_fwd_cudnn operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/flash_sdpa_fwd_cudnn.hh"
#include "nntile/graph/logical_graph_ops.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

struct FlashGraphCase
{
    Index head_size;
    Index n_seq;
    Index n_seq_tile;
    Index n_batch;
    Index batch_tile;
    Index kv_group_size;
    Index kv_group_tile;
    Index n_head_kv;
    Index head_kv_tile;
};

template<typename T>
void run_test(const nntile::Context& context)
{
    using Y = typename T::repr_t;
    DataType dtype = std::is_same_v<T, nntile::fp16_t> ? DataType::FP16 :
                     std::is_same_v<T, nntile::bf16_t> ? DataType::BF16 : DataType::FP32;

    // Define tensor shapes for single-tile test (similar to tensor test)
    Index head_size = 32;
    Index n_seq = 64;
    Index n_batch = 1;
    Index kv_group_size = 1;
    Index n_head_kv = 1;

    // Create tensor shapes
    std::vector<Index> kv_tensor_shape = {head_size, n_seq, n_batch, kv_group_size, n_head_kv};
    std::vector<Index> logsumexp_shape = {n_seq, n_batch, kv_group_size, n_head_kv};
    std::vector<Index> mask_shape = {n_seq, n_seq};

    auto build_graph = [&](LogicalGraph& g) {
        auto& K = g.tensor(kv_tensor_shape, "K", dtype);
        auto& Q = g.tensor(kv_tensor_shape, "Q", dtype);
        auto& mask = g.tensor(mask_shape, "mask", dtype);
        auto& logsumexp = g.tensor(logsumexp_shape, "logsumexp", DataType::FP32);
        auto& V = g.tensor(kv_tensor_shape, "V", dtype);
        auto& A = g.tensor(kv_tensor_shape, "A", dtype);

        flash_sdpa_fwd_cudnn(K, Q, mask, logsumexp, V, A);
    };

    auto run_tensor_direct = [&](std::map<std::string, std::vector<Y>>& inputs,
                               std::map<std::string, std::vector<Y>>& outputs,
                               const nntile::Context& ctx) {
        using TensorT = nntile::tensor::Tensor<T>;
        using TensorFp32 = nntile::tensor::Tensor<fp32_t>;

        // Create single-tile tensors for direct comparison
        nntile::tensor::TensorTraits traits(kv_tensor_shape, kv_tensor_shape);
        nntile::tensor::TensorTraits mask_traits(mask_shape, mask_shape);
        nntile::tensor::TensorTraits logsumexp_traits(logsumexp_shape, logsumexp_shape);

        TensorT K(traits);
        TensorT Q(traits);
        TensorT V(traits);
        TensorT A(traits);
        TensorT mask(mask_traits);
        TensorFp32 logsumexp(logsumexp_traits);

        write_tensor(K, inputs["K"]);
        write_tensor(Q, inputs["Q"]);
        write_tensor(V, inputs["V"]);
        write_tensor(A, inputs["A"]);
        write_tensor(mask, inputs["mask"]);

        // Initialize logsumexp to -inf
        auto logsumexp_tile = logsumexp.get_tile(0);
        auto logsumexp_local = logsumexp_tile.acquire(STARPU_W);
        for(Index i = 0; i < logsumexp_tile.nelems; ++i) {
            logsumexp_local[i] = -std::numeric_limits<float>::infinity();
        }
        logsumexp_local.release();

        // Run the tensor operation
        nntile::tensor::flash_sdpa_fwd_cudnn<T>(K, Q, mask, logsumexp, V, A);

        outputs["A"] = read_tensor(A);
        outputs["logsumexp"] = read_tensor(logsumexp);
    };

    // Create custom input data (similar to tensor test)
    std::map<std::string, std::vector<Y>> custom_inputs;

    // K, Q, V data
    size_t kv_size = head_size * n_seq * n_batch * kv_group_size * n_head_kv;
    custom_inputs["K"] = make_pattern<Y>(kv_size, static_cast<Y>(0.01));
    custom_inputs["Q"] = make_pattern<Y>(kv_size, static_cast<Y>(0.02));
    custom_inputs["V"] = make_pattern<Y>(kv_size, static_cast<Y>(0.03));
    custom_inputs["A"] = std::vector<Y>(kv_size, static_cast<Y>(0.0)); // Initialize to zero

    // Mask data - causal mask
    size_t mask_size = n_seq * n_seq;
    custom_inputs["mask"] = std::vector<Y>(mask_size, static_cast<Y>(0.0));
    for(Index i = 0; i < n_seq; ++i) {
        for(Index j = 0; j < n_seq; ++j) {
            Index idx = i * n_seq + j;
            if(j > i) {
                custom_inputs["mask"][idx] = -std::numeric_limits<Y>::infinity();
            }
        }
    }

    verify_graph_vs_tensor<T>(
        build_graph, run_tensor_direct,
        {"K", "Q", "mask", "V", "A"}, {"A", "logsumexp"}, context, custom_inputs
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph FlashSDPACUDNN vs Tensor",
    "[graph][verification]")
{
    // Only test FP16 and BF16 as per cuDNN limitations
    run_test<fp16_t>(context);
    run_test<bf16_t>(context);
}