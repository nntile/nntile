# LogicalGraph Implementation Plan

This document provides a step-by-step implementation plan for the NNTile 2.0 LogicalGraph system. The initial implementation is minimal: only `matmul` and `gelu` operations, no tiling, single node with single GPU worker.

---

## 1. Overview

### 1.1 Goal

Implement a minimal LogicalGraph that:
1. Defines tensors and operations (matmul, gelu)
2. Compiles to a CompiledGraph
3. Executes using existing NNTile tensor operations (`nntile::tensor::gemm`, `nntile::tensor::gelu`)
4. Works on a single node with a single GPU

### 1.2 Scope

**In scope (Phase 1)**:
- `LogicalGraph` class with `tensor()`, `matmul()`, `gelu()`, `mark_output()`
- `TensorNode` and `OpNode` classes
- `TensorSpec` (shape + dtype only)
- `CompiledGraph` with `compile()`, `bind_data()`, `execute()`, `get_output()`
- Single node, single worker (no distribution)
- No tiling (each tensor = 1 tile)

**Out of scope (Phase 1)**:
- Multiple nodes/workers
- Tiling strategies
- Distribution (FSDP, DDP)
- Other operations (add, softmax, etc.)
- Graph mutation (remove, rename)
- Cross-graph transfer

### 1.3 File Structure

```
include/nntile/graph/
├── tensor_spec.hh      # TensorSpec class
├── logical_graph.hh    # LogicalGraph + tensor/op node classes
├── compiled_graph.hh   # CompiledGraph class
└── graph.hh            # Convenience header (includes all)

src/graph/
├── tensor_spec.cc
├── logical_graph.cc
├── compiled_graph.cc
└── CMakeLists.txt
```

---

## 2. Implementation Steps

### Step 1: TensorSpec

**File**: `include/nntile/graph/tensor_spec.hh`

```cpp
#pragma once

#include <nntile/defs.hh>
#include <vector>
#include <string>

namespace nntile::graph {

//! Data types supported
enum class DataType {
    FP32,
    FP64,
    FP16,
    BF16,
    INT64,
    INT32,
    BOOL
};

//! Convert DataType to string
std::string dtype_to_string(DataType dtype);

//! Get size in bytes for DataType
size_t dtype_size(DataType dtype);

//! Tensor specification - shape and data type
class TensorSpec {
private:
    std::vector<Index> shape_;
    DataType dtype_;

public:
    //! Construct with shape and dtype
    TensorSpec(std::vector<Index> shape, DataType dtype = DataType::FP32);

    //! Accessors
    const std::vector<Index>& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    Index ndim() const { return static_cast<Index>(shape_.size()); }

    //! Get dimension at index (supports negative indexing)
    Index dim(int idx) const;

    //! Total number of elements
    Index nelems() const;

    //! Total size in bytes
    size_t size_bytes() const;

    //! Check if shapes are compatible for operations
    bool is_compatible(const TensorSpec& other) const;

    //! String representation
    std::string to_string() const;
};

} // namespace nntile::graph
```

**File**: `src/graph/tensor_spec.cc`

Implement all methods. Key points:
- `nelems()`: multiply all dimensions
- `dim(idx)`: handle negative indexing (dim(-1) = last dimension)
- `to_string()`: e.g., "TensorSpec([32, 768], FP32)"

**Test**: Create a simple test that constructs TensorSpec and verifies shape/dtype/nelems.

---

### Step 2: Tensor/Op Nodes (nested in LogicalGraph)

**File**: `include/nntile/graph/logical_graph.hh`

```cpp
namespace nntile::graph {

class LogicalGraph {
public:
    class OpNode;
    class TensorNode;

    class TensorNode {
        friend class LogicalGraph;
        friend class OpNode;
        // ...
    };

    class OpNode {
        friend class LogicalGraph;
        // ...
    };
};

} // namespace nntile::graph
```

**File**: `src/graph/logical_graph.cc`

Implement constructors and `to_string()` for both nested classes.

---

### Step 4: LogicalGraph

**File**: `include/nntile/graph/logical_graph.hh`

```cpp
#pragma once

#include <nntile/graph/logical_graph.hh>
#include <memory>
#include <vector>
#include <map>
#include <set>

namespace nntile::graph {

//! Logical graph - defines computation without physical details
class LogicalGraph {
private:
    std::string name_;
    std::vector<std::unique_ptr<TensorNode>> tensors_;
    std::vector<std::unique_ptr<OpNode>> ops_;
    std::map<std::string, TensorNode*> tensor_by_name_;
    std::set<std::string> output_names_;

    NodeId next_tensor_id_ = 0;
    NodeId next_op_id_ = 0;

public:
    explicit LogicalGraph(const std::string& name = "");

    // ═══════════════════════════════════════════════════════════════
    // Tensor Creation
    // ═══════════════════════════════════════════════════════════════

    //! Create a tensor
    TensorNode& tensor(const TensorSpec& spec, const std::string& name);

    //! Mark tensor as output
    void mark_output(const std::string& name);

    // ═══════════════════════════════════════════════════════════════
    // Operations
    // ═══════════════════════════════════════════════════════════════

    //! Matrix multiplication: C = A @ B
    //! Returns reference to output tensor
    TensorNode& matmul(
        TensorNode& a,
        TensorNode& b,
        const std::string& output_name,
        bool trans_a = false,
        bool trans_b = false
    );

    //! GeLU activation: y = gelu(x)
    TensorNode& gelu(
        TensorNode& x,
        const std::string& output_name
    );

    // ═══════════════════════════════════════════════════════════════
    // Queries
    // ═══════════════════════════════════════════════════════════════

    const std::string& name() const { return name_; }
    size_t num_tensors() const { return tensors_.size(); }
    size_t num_ops() const { return ops_.size(); }

    //! Get tensor by name (returns nullptr if not found)
    TensorNode* get_tensor(const std::string& name);
    const TensorNode* get_tensor(const std::string& name) const;

    //! Get all tensor names
    std::vector<std::string> tensor_names() const;

    //! Get output tensor names
    const std::set<std::string>& output_names() const { return output_names_; }

    //! Check if tensor is an output
    bool is_output(const std::string& name) const;

    //! Get all tensors (for iteration)
    const std::vector<std::unique_ptr<TensorNode>>& tensors() const { return tensors_; }

    //! Get all ops (for iteration)
    const std::vector<std::unique_ptr<OpNode>>& ops() const { return ops_; }

    //! String representation
    std::string to_string() const;

private:
    //! Internal: create output tensor for an operation
    TensorNode& create_op_output(
        OpNode& op,
        const TensorSpec& spec,
        const std::string& name
    );

    //! Compute output shape for matmul
    TensorSpec compute_matmul_output_spec(
        const TensorSpec& a,
        const TensorSpec& b,
        bool trans_a,
        bool trans_b
    );
};

} // namespace nntile::graph
```

**File**: `src/graph/logical_graph.cc`

Key implementations:

1. **`tensor()`**:
   - Check name doesn't already exist
   - Create TensorNode with unique ID
   - Store in tensors_ and tensor_by_name_
   - Return reference

2. **`matmul()`**:
   - Validate input shapes are compatible
   - Compute output shape based on trans_a, trans_b
   - Create OpNode with MatmulAttrs
   - Create output TensorNode
   - Wire up edges (inputs/outputs/producer/consumers)
   - Return reference to output tensor

3. **`gelu()`**:
   - Output shape = input shape
   - Create OpNode with GeluAttrs
   - Create output TensorNode
   - Wire up edges
   - Return reference to output tensor

4. **`compute_matmul_output_spec()`**:
   ```cpp
   // A: [M, K] (or [K, M] if trans_a)
   // B: [K, N] (or [N, K] if trans_b)
   // C: [M, N]
   Index M = trans_a ? a.dim(1) : a.dim(0);
   Index K_a = trans_a ? a.dim(0) : a.dim(1);
   Index K_b = trans_b ? b.dim(1) : b.dim(0);
   Index N = trans_b ? b.dim(0) : b.dim(1);

   if (K_a != K_b) throw std::invalid_argument("matmul: incompatible shapes");

   return TensorSpec({M, N}, a.dtype());
   ```

---

### Step 5: CompiledGraph

**File**: `include/nntile/graph/compiled_graph.hh`

```cpp
#pragma once

#include <nntile/graph/logical_graph.hh>
#include <nntile/tensor/tensor.hh>
#include <nntile/starpu/config.hh>
#include <memory>
#include <map>

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

    // StarPU config
    starpu::Config* config_ = nullptr;

public:
    //! Compile a logical graph
    //! For Phase 1: num_nodes=1, no tiling
    static CompiledGraph compile(
        const LogicalGraph& logical,
        starpu::Config& config
    );

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
```

**File**: `src/graph/compiled_graph.cc`

Key implementations:

1. **`compile()`**:
   ```cpp
   static CompiledGraph compile(const LogicalGraph& logical, starpu::Config& config) {
       CompiledGraph cg;
       cg.logical_ = &logical;
       cg.config_ = &config;

       cg.allocate_tensors();
       cg.compute_execution_order();

       return cg;
   }
   ```

2. **`allocate_tensors()`**:
   - For each TensorNode in logical graph:
   - Create NNTile tensor with shape = node's shape, tile_shape = full shape (no tiling)
   - Store in tensors_ map with type erasure

   ```cpp
   void CompiledGraph::allocate_tensors() {
       for (const auto& node : logical_->tensors()) {
           const auto& spec = node->spec();
           tensor_dtypes_[node->name()] = spec.dtype();

           // Create tensor with single tile (no tiling)
           std::vector<Index> shape = spec.shape();
           std::vector<Index> tile_shape = shape;  // Same as shape = 1 tile

           switch (spec.dtype()) {
               case DataType::FP32: {
                   auto t = std::make_shared<nntile::tensor::Tensor<float>>(
                       shape, tile_shape, *config_
                   );
                   tensors_[node->name()] = t;
                   break;
               }
               case DataType::FP64: {
                   auto t = std::make_shared<nntile::tensor::Tensor<double>>(
                       shape, tile_shape, *config_
                   );
                   tensors_[node->name()] = t;
                   break;
               }
               // Add other types as needed
           }
       }
   }
   ```

3. **`compute_execution_order()`**:
   - Topological sort of ops based on dependencies
   - Simple algorithm: ops with no unexecuted dependencies go first

   ```cpp
   void CompiledGraph::compute_execution_order() {
       execution_order_.clear();
       std::set<NodeId> executed_tensors;

       // Mark all input tensors (no producer) as executed
       for (const auto& t : logical_->tensors()) {
           if (!t->has_producer()) {
               executed_tensors.insert(t->id());
           }
       }

       // Keep adding ops whose inputs are all ready
       std::set<NodeId> executed_ops;
       while (execution_order_.size() < logical_->num_ops()) {
           for (const auto& op : logical_->ops()) {
               if (executed_ops.count(op->id())) continue;

               // Check if all inputs are ready
               bool ready = true;
               for (const auto* input : op->inputs()) {
                   if (!executed_tensors.count(input->id())) {
                       ready = false;
                       break;
                   }
               }

               if (ready) {
                   execution_order_.push_back(op.get());
                   executed_ops.insert(op->id());
                   for (const auto* output : op->outputs()) {
                       executed_tensors.insert(output->id());
                   }
               }
           }
       }
   }
   ```

4. **`execute()`**:
   ```cpp
   void CompiledGraph::execute() {
       for (const OpNode* op : execution_order_) {
           execute_op(op);
       }
   }
   ```

5. **`execute_op()`**:
   ```cpp
   void CompiledGraph::execute_op(const OpNode* op) {
       switch (op->type()) {
           case OpType::MATMUL:
               execute_matmul(op);
               break;
           case OpType::GELU:
               execute_gelu(op);
               break;
       }
   }
   ```

6. **`execute_matmul()`**:
   ```cpp
   void CompiledGraph::execute_matmul(const OpNode* op) {
       const auto& attrs = std::get<MatmulAttrs>(op->attrs());

       const std::string& a_name = op->input(0)->name();
       const std::string& b_name = op->input(1)->name();
       const std::string& c_name = op->output(0)->name();

       DataType dtype = tensor_dtypes_[a_name];

       if (dtype == DataType::FP32) {
           auto& a = get_tensor<float>(a_name);
           auto& b = get_tensor<float>(b_name);
           auto& c = get_tensor<float>(c_name);

           // Use nntile::tensor::gemm
           // gemm(alpha, trans_a, a, trans_b, b, beta, c)
           nntile::tensor::gemm<float>(
               static_cast<float>(attrs.alpha),
               attrs.trans_a ? nntile::TransOp::Trans : nntile::TransOp::NoTrans,
               a,
               attrs.trans_b ? nntile::TransOp::Trans : nntile::TransOp::NoTrans,
               b,
               static_cast<float>(attrs.beta),
               c
           );
       }
       // Add FP64 support similarly
   }
   ```

7. **`execute_gelu()`**:
   ```cpp
   void CompiledGraph::execute_gelu(const OpNode* op) {
       const std::string& x_name = op->input(0)->name();
       const std::string& y_name = op->output(0)->name();

       DataType dtype = tensor_dtypes_[x_name];

       if (dtype == DataType::FP32) {
           auto& x = get_tensor<float>(x_name);
           auto& y = get_tensor<float>(y_name);

           // Use nntile::tensor::gelu
           nntile::tensor::gelu<float>(x, y);
       }
       // Add FP64 support similarly
   }
   ```

8. **`bind_data()`** and **`get_output()`**:
   - Use existing tensor methods to copy data in/out
   - May need to implement scatter/gather for tensor tiles

---

### Step 6: CMakeLists.txt

**File**: `src/graph/CMakeLists.txt`

```cmake
# Graph library sources
set(GRAPH_SOURCES
    tensor_spec.cc
    logical_graph.cc
    compiled_graph.cc
)

# Create library
add_library(nntile_graph ${GRAPH_SOURCES})

target_include_directories(nntile_graph PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(nntile_graph PUBLIC
    nntile_tensor
    nntile_starpu
)

# Install
install(TARGETS nntile_graph DESTINATION lib)
```

Update `src/CMakeLists.txt` to include `add_subdirectory(graph)`.

---

### Step 7: Convenience Header

**File**: `include/nntile/graph/graph.hh`

```cpp
#pragma once

#include <nntile/graph/tensor_spec.hh>
#include <nntile/graph/logical_graph.hh>
#include <nntile/graph/nn_graph.hh>
#include <nntile/graph/compiled_graph.hh>
```

---

## 3. Testing

### Step 8: Basic Tests

**File**: `tests/graph/test_logical_graph.cc`

```cpp
#include <nntile/graph.hh>
#include <catch2/catch_test_macros.hpp>

using namespace nntile::graph;

TEST_CASE("LogicalGraph CreateTensor", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor(TensorSpec({32, 768}, DataType::FP32), "x");

    REQUIRE(x.name() == "x");
    REQUIRE(x.shape().size() == 2);
    REQUIRE(x.shape()[0] == 32);
    REQUIRE(x.shape()[1] == 768);
    REQUIRE(x.dtype() == DataType::FP32);
    REQUIRE_FALSE(x.has_producer());
}

TEST_CASE("LogicalGraph Gemm", "[graph]")
{
    LogicalGraph g("test");

    auto& a = g.tensor(TensorSpec({32, 768}, DataType::FP32), "a");
    auto& b = g.tensor(TensorSpec({768, 256}, DataType::FP32), "b");
    auto& c = gemm(a, b, "c");

    REQUIRE(c.shape()[0] == 32);
    REQUIRE(c.shape()[1] == 256);
    REQUIRE(c.has_producer());
    REQUIRE(c.producer()->type() == OpType::GEMM);
}

TEST_CASE("LogicalGraph GemmTranspose", "[graph]")
{
    LogicalGraph g("test");

    auto& a = g.tensor(
        TensorSpec({768, 32}, DataType::FP32),
        "a");  // Will be transposed
    auto& b = g.tensor(TensorSpec({768, 256}, DataType::FP32), "b");
    auto& c = gemm(
        a,
        b,
        "c",
        1.0,
        /*trans_a=*/true,
        /*trans_b=*/false);

    REQUIRE(c.shape()[0] == 32);   // M from A^T
    REQUIRE(c.shape()[1] == 256);  // N from B
}

TEST_CASE("LogicalGraph Gelu", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor(TensorSpec({32, 768}, DataType::FP32), "x");
    auto& y = gelu(x, "y");

    REQUIRE(y.shape() == x.shape());
    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::GELU);
}

TEST_CASE("LogicalGraph Chain", "[graph]")
{
    LogicalGraph g("mlp");

    auto& x = g.tensor(TensorSpec({32, 768}, DataType::FP32), "x");
    auto& w1 = g.tensor(TensorSpec({768, 3072}, DataType::FP32), "w1");
    auto& w2 = g.tensor(TensorSpec({3072, 768}, DataType::FP32), "w2");

    auto& h = gemm(x, w1, "h");
    auto& a = gelu(h, "a");
    auto& y = gemm(a, w2, "y");

    REQUIRE(g.num_tensors() == 6);  // x, w1, w2, h, a, y
    REQUIRE(g.num_ops() == 3);      // gemm, gelu, gemm
}
```

### Step 9: Execution Test

**File**: `tests/graph/test_compiled_graph.cc`

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>

#include "nntile/context.hh"
#include "nntile/graph.hh"

using namespace nntile::graph;

// Fixture to initialize NNTile context for graph tests
class GraphTestFixture
{
protected:
    nntile::Context context;
public:
    GraphTestFixture():
        context(
            1, 0, 0, "/tmp/nntile_ooc", 16777216, 0, "localhost", 5001, 0
        )
    {}
};

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph SimpleGemm",
    "[graph]"
)
{
    LogicalGraph g("test");

    auto& a = g.tensor(TensorSpec({2, 3}, DataType::FP32), "a");
    auto& b = g.tensor(TensorSpec({3, 4}, DataType::FP32), "b");
    auto& c = gemm(a, b, "c");

    auto compiled = CompiledGraph::compile(g);

    // NNTile tensors use column-major (Fortran) layout.
    // A = [[1,2,3], [4,5,6]]
    std::vector<float> a_data = {1, 4, 2, 5, 3, 6};
    // B = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    std::vector<float> b_data = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};

    compiled.bind_data("a", a_data);
    compiled.bind_data("b", b_data);

    compiled.execute();
    compiled.wait();

    auto c_data = compiled.get_output<float>("c");

    // C = A @ B = [[38,44,50,56], [83,98,113,128]]
    REQUIRE(c_data.size() == 8);
    REQUIRE(c_data[0] == 38.0f);
    REQUIRE(c_data[1] == 83.0f);
    REQUIRE(c_data[2] == 44.0f);
    REQUIRE(c_data[3] == 98.0f);
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph GeluActivation",
    "[graph]"
)
{
    LogicalGraph g("test");

    auto& x = g.tensor(TensorSpec({4}, DataType::FP32), "x");
    auto& y = gelu(x, "y");

    auto compiled = CompiledGraph::compile(g);

    std::vector<float> x_data = {-1.0f, 0.0f, 1.0f, 2.0f};
    compiled.bind_data("x", x_data);

    compiled.execute();
    compiled.wait();

    auto y_data = compiled.get_output<float>("y");

    // GELU(-1) ≈ -0.159, GELU(0) = 0, GELU(1) ≈ 0.841, GELU(2) ≈ 1.955
    REQUIRE_THAT(y_data[0], Catch::Approx(-0.159f).margin(0.01f));
    REQUIRE_THAT(y_data[1], Catch::Approx(0.0f).margin(0.01f));
    REQUIRE_THAT(y_data[2], Catch::Approx(0.841f).margin(0.01f));
    REQUIRE_THAT(y_data[3], Catch::Approx(1.955f).margin(0.01f));
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph MLP",
    "[graph]"
)
{
    LogicalGraph g("mlp");

    // x: [2, 4], w1: [4, 8], w2: [8, 4]
    auto& x = g.tensor(TensorSpec({2, 4}, DataType::FP32), "x");
    auto& w1 = g.tensor(TensorSpec({4, 8}, DataType::FP32), "w1");
    auto& w2 = g.tensor(TensorSpec({8, 4}, DataType::FP32), "w2");

    auto& h = gemm(x, w1, "h");
    auto& a = gelu(h, "a");
    auto& y = gemm(a, w2, "y");

    auto compiled = CompiledGraph::compile(g);

    // Initialize with simple values
    std::vector<float> x_data(8, 1.0f);
    std::vector<float> w1_data(32, 0.1f);
    std::vector<float> w2_data(32, 0.1f);

    compiled.bind_data("x", x_data);
    compiled.bind_data("w1", w1_data);
    compiled.bind_data("w2", w2_data);

    compiled.execute();
    compiled.wait();

    auto y_data = compiled.get_output<float>("y");

    REQUIRE(y_data.size() == 8);  // [2, 4]
    // Values should be non-zero and reasonable
    for (float v : y_data) {
        REQUIRE(v > 0.0f);
        REQUIRE(v < 10.0f);
    }
}
```

---

## 4. Integration Notes

### 4.1 Using Existing NNTile Operations

The CompiledGraph should use existing operations:

```cpp
#include <nntile/tensor/gemm.hh>  // For matmul
#include <nntile/tensor/gelu.hh>  // For gelu
```

Check existing signatures:
- `nntile::tensor::gemm<T>(alpha, trans_a, A, trans_b, B, beta, C)`
- `nntile::tensor::gelu<T>(src, dst)`

### 4.2 StarPU Initialization

Tests need proper StarPU setup:

```cpp
// In test main or fixture
nntile::starpu::Config config(ncpu, ncuda, 0);
// ... run tests ...
// Config destructor calls starpu_shutdown()
```

### 4.3 Memory Layout

NNTile tensors use **column-major** order (Fortran style). When binding data:
- If user data is row-major (C style), may need transpose
- Document this clearly in API

---

## 5. Future Extensions (Not in Phase 1)

After Phase 1 is complete and tested:

1. **More operations**: add, relu, softmax, layer_norm, etc.
2. **Tiling**: Allow tile_shape != shape
3. **Distribution**: TileDistribution for multi-node
4. **Backward ops**: gelu_backward, gemm for gradients
5. **Graph mutation**: remove, rename, clear
6. **Cross-graph transfer**: transfer_to()

---

## 6. Checklist

- [ ] Implement TensorSpec
- [ ] Implement TensorNode
- [ ] Implement OpNode
- [ ] Implement LogicalGraph with tensor(), matmul(), gelu(), mark_output()
- [ ] Implement CompiledGraph with compile(), bind_data(), execute(), get_output()
- [ ] Add CMakeLists.txt
- [ ] Write tests for LogicalGraph
- [ ] Write tests for CompiledGraph execution
- [ ] Verify matmul produces correct results
- [ ] Verify gelu produces correct results
- [ ] Verify MLP chain (matmul -> gelu -> matmul) works
- [ ] Document memory layout (column-major)
- [ ] Handle FP32 and FP64 data types
