/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/compiled_graph.hh
 * CompiledGraph class for executing logical graphs.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/logical_graph.hh>
#include <nntile/tensor/tensor.hh>
#include <memory>
#include <map>
#include <vector>

namespace nntile::graph {

//! Compiled graph - ready for execution
class CompiledGraph {
private:
    const LogicalGraph* logical_;

    // Runtime tensors (NNTile tensors, one tile each)
    std::map<std::string, std::shared_ptr<void>> tensors_;  // Type-erased tensor pointers
    std::map<std::string, DataType> tensor_dtypes_;

    // Execution order (topologically sorted ops)
    std::vector<const OpNode*> execution_order_;

public:
    //! Compile a logical graph
    static CompiledGraph compile(const LogicalGraph& logical);

    // ═══════════════════════════════════════════════════════════════
    // Data Binding
    // ═══════════════════════════════════════════════════════════════

    //! Bind data to a tensor (copies data)
    template<typename T>
    void bind_data(const std::string& name, const T* data, size_t count);

    //! Bind data from vector
    template<typename T>
    void bind_data(const std::string& name, const std::vector<T>& data);

    // ═══════════════════════════════════════════════════════════════
    // Execution
    // ═══════════════════════════════════════════════════════════════

    //! Execute the graph
    void execute();

    //! Wait for all operations to complete
    void wait();

    // ═══════════════════════════════════════════════════════════════
    // Output Retrieval
    // ═══════════════════════════════════════════════════════════════

    //! Get output data (copies data out)
    template<typename T>
    std::vector<T> get_output(const std::string& name);

    //! Get raw pointer to output (must call wait() first)
    template<typename T>
    const T* get_output_ptr(const std::string& name);

private:
    CompiledGraph() = default;

    //! Allocate NNTile tensors for all graph tensors
    void allocate_tensors();

    //! Compute topological order of operations
    void compute_execution_order();

    //! Execute a single operation
    void execute_op(const OpNode* op);

    //! Execute matmul operation
    void execute_matmul(const OpNode* op);

    //! Execute gelu operation
    void execute_gelu(const OpNode* op);

    //! Get typed tensor pointer
    template<typename T>
    nntile::tensor::Tensor<T>& get_tensor(const std::string& name);
};

} // namespace nntile::graph