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

// Include standard headers
#include <map>
#include <memory>
#include <vector>

// Include third-party headers

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>
#include <nntile/tensor/tensor.hh>

namespace nntile::graph
{

//! Operation execution information (extracted during compilation)
struct OpExecutionInfo
{
    OpType type;
    OpAttrs attrs;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

//! Compiled graph - ready for execution
class CompiledGraph
{
private:
    // Runtime tensors (NNTile tensors, one tile each)
    // Type-erased tensor pointers
    std::map<std::string, std::shared_ptr<void>> tensors_;
    std::map<std::string, DataType> tensor_dtypes_;

    // Execution order (topologically sorted ops with extracted info)
    std::vector<OpExecutionInfo> execution_order_;

public:
    //! Compile a logical graph
    static CompiledGraph compile(const LogicalGraph& logical);

    // -----------------------------------------------------------------
    // Data Binding
    // -----------------------------------------------------------------

    //! Bind data to a tensor (copies data)
    template<typename T>
    void bind_data(const std::string& name, const T* data, size_t count);

    //! Bind data from vector
    template<typename T>
    void bind_data(const std::string& name, const std::vector<T>& data);

    // -----------------------------------------------------------------
    // Execution
    // -----------------------------------------------------------------

    //! Execute the graph
    void execute();

    //! Wait for all operations to complete
    void wait();

    // -----------------------------------------------------------------
    // Output Retrieval
    // -----------------------------------------------------------------

    //! Get output data (copies data out)
    template<typename T>
    std::vector<T> get_output(const std::string& name);

    // -----------------------------------------------------------------
    // Internal Access (for operation implementations)
    // -----------------------------------------------------------------

    //! Get typed tensor pointer (used by operation implementations)
    template<typename T>
    nntile::tensor::Tensor<T>& get_tensor(const std::string& name);

    //! Get data type of tensor
    DataType get_dtype(const std::string& name) const
    {
        return tensor_dtypes_.at(name);
    }

private:
    CompiledGraph() = default;

    //! Allocate NNTile tensors for all graph tensors
    void allocate_tensors(const LogicalGraph& logical);

    //! Execute a single operation
    void execute_op(const OpExecutionInfo& op_info);
};

} // namespace nntile::graph
