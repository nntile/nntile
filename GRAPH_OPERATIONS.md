# Adding New Graph Operations to NNTile

This document describes how to add new operations to the NNTile graph system.

## Overview

The graph system has two components:
- **Logical Graph**: Defines the computation symbolically (shape inference, validation)
- **Compiled Graph**: Executes the computation using NNTile tensors

Each operation requires:
1. An `OpType` enum value and `OpAttrs` struct (in `logical_graph.hh`)
2. A logical operation (free function that adds the op to LogicalGraph)
3. A compiled operation (free function that executes the op on CompiledGraph)

## File Structure

```
include/nntile/graph/
├── graph.hh              # Convenience header (includes all)
├── logical_graph.hh      # LogicalGraph + nested tensor/op node classes
├── compiled_graph.hh     # CompiledGraph class
├── tensor_spec.hh        # TensorSpec class
├── logical/              # Logical operation headers
│   ├── gemm.hh
│   └── gelu.hh
└── compiled/             # Compiled operation headers
    ├── gemm.hh
    └── gelu.hh

src/graph/
├── logical_graph.cc      # LogicalGraph + tensor/op node implementation
├── compiled_graph.cc     # CompiledGraph implementation
├── tensor_spec.cc
├── logical/              # Logical operation implementations
│   ├── gemm.cc
│   └── gelu.cc
└── compiled/             # Compiled operation implementations
    ├── gemm.cc
    └── gelu.cc
```

## Step-by-Step Guide

### Step 1: Add OpType and OpAttrs

In `include/nntile/graph/logical_graph.hh`:

```cpp
// Add to OpType enum
enum class OpType {
    GEMM,
    GELU,
    YOUR_OP  // Add here
};

// Add attributes struct
struct YourOpAttrs
{
    // Operation-specific parameters (not tensors)
    float some_param = 0.0f;
    bool some_flag = false;
};

// Add to OpAttrs variant
using OpAttrs = std::variant<GemmAttrs, GeluAttrs, YourOpAttrs>;
```

In `src/graph/logical_graph.cc`, add the string conversion:

```cpp
std::string op_type_to_string(OpType type)
{
    switch(type)
    {
        case OpType::GEMM: return "GEMM";
        case OpType::GELU: return "GELU";
        case OpType::YOUR_OP: return "YOUR_OP";  // Add here
    }
    return "UNKNOWN";
}
```

### Step 2: Create Logical Operation

Create `include/nntile/graph/logical/your_op.hh`:

```cpp
#pragma once

#include <string>
#include <nntile/base_types.hh>
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Your operation description
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @param some_param Some parameter
//! @return Reference to the output tensor
LogicalGraph::TensorNode& your_op(
    LogicalGraph::TensorNode& x,
    const std::string& output_name,
    float some_param = 0.0f
);

} // namespace nntile::graph
```

Create `src/graph/logical/your_op.cc`:

```cpp
#include "nntile/graph/logical/your_op.hh"
#include "nntile/graph/logical_graph.hh"
#include <stdexcept>

namespace nntile::graph
{

LogicalGraph::TensorNode& your_op(
    LogicalGraph::TensorNode& x,
    const std::string& output_name,
    float some_param)
{
    // 1. Validate inputs (optional - add checks as needed)
    // Example: validate dtype
    if(x.dtype() != DataType::FP32 && x.dtype() != DataType::FP64)
    {
        throw std::invalid_argument(
            "your_op: only FP32 and FP64 are supported");
    }

    // 2. Compute output specification (shape and dtype)
    // Most ops preserve shape, some change it
    TensorSpec output_spec = TensorSpec(x.shape(), x.dtype());

    // 3. Create operation attributes
    OpAttrs attrs = YourOpAttrs{some_param};

    // 4. Create output tensor
    LogicalGraph::TensorNode& output = x.graph().tensor(output_spec, output_name);

    // 5. Add operation to graph using public builder API
    x.graph().add_op(
        OpType::YOUR_OP,
        attrs,
        {&x},           // Input tensors
        {&output}       // Output tensors (must be pre-created)
    );

    return output;
}

} // namespace nntile::graph
```

### Step 2b: In-Place Operations (Optional)

Some operations support both variants:
1. **Creating new tensor**: `y = f(x)` - creates output tensor and adds operation
2. **Accumulating into existing tensor**: `y = f(x) + beta * y` - modifies existing tensor

Example with gemm:

```cpp
// Variant 1: C = alpha * A @ B (creates new tensor)
LogicalGraph::TensorNode& gemm(LogicalGraph::TensorNode& a,
                             LogicalGraph::TensorNode& b,
                             const std::string& output_name,
                             ...);

// Variant 2: C = alpha * A @ B + beta * C (accumulates into existing tensor)
void gemm(LogicalGraph::TensorNode& a,
          LogicalGraph::TensorNode& b,
          LogicalGraph::TensorNode& c,
          Scalar alpha,
          Scalar beta,
          ...);
```

For in-place operations, use the unified `add_op()` with the output tensor in both inputs and outputs:

```cpp
void your_op_inplace(LogicalGraph::TensorNode& x,
                     LogicalGraph::TensorNode& y,
                     float some_param)
{
    OpAttrs attrs = YourOpAttrs{some_param};

    // For in-place operations, include the output tensor in both inputs and outputs
    x.graph().add_op(
        OpType::YOUR_OP,
        attrs,
        {&x, &y},    // All input tensors (including the one being modified)
        {&y}         // Output tensors (the tensor being modified)
    );
}
```

### Step 3: Create Compiled Operation

Create `include/nntile/graph/compiled/your_op.hh`:

```cpp
#pragma once

namespace nntile::graph
{
class CompiledGraph;
struct OpExecutionInfo;

//! Execute your_op on compiled graph
void execute_your_op(CompiledGraph& graph, const OpExecutionInfo& op_info);

} // namespace nntile::graph
```

Create `src/graph/compiled/your_op.cc`:

```cpp
#include "nntile/graph/compiled/your_op.hh"
#include "nntile/graph/compiled_graph.hh"
#include "nntile/base_types.hh"
#include "nntile/tensor/your_op.hh"  // NNTile tensor operation
#include <stdexcept>

namespace nntile::graph
{

void execute_your_op(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    // 1. Extract tensor names
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];

    // 2. Get attributes (if needed)
    const auto& attrs = std::get<YourOpAttrs>(op_info.attrs);

    // 3. Get data type and dispatch
    DataType dtype = graph.get_dtype(x_name);

    if(dtype == DataType::FP32)
    {
        auto& x = graph.get_tensor<nntile::fp32_t>(x_name);
        auto& y = graph.get_tensor<nntile::fp32_t>(y_name);
        nntile::tensor::your_op<nntile::fp32_t>(x, y, attrs.some_param);
    }
    else if(dtype == DataType::FP64)
    {
        auto& x = graph.get_tensor<nntile::fp64_t>(x_name);
        auto& y = graph.get_tensor<nntile::fp64_t>(y_name);
        nntile::tensor::your_op<nntile::fp64_t>(x, y, attrs.some_param);
    }
    // Add more dtype cases as needed...
    else
    {
        throw std::runtime_error("Unsupported data type for your_op");
    }
}

} // namespace nntile::graph
```

### Step 4: Register in CompiledGraph Dispatcher

In `src/graph/compiled_graph.cc`, add to `execute_op()`:

```cpp
void CompiledGraph::execute_op(const OpExecutionInfo& op_info)
{
    switch(op_info.type)
    {
        case OpType::GEMM:
            execute_gemm(*this, op_info);
            break;
        case OpType::GELU:
            execute_gelu(*this, op_info);
            break;
        case OpType::YOUR_OP:              // Add here
            execute_your_op(*this, op_info);
            break;
    }
}
```

Also add the include at the top:

```cpp
#include "nntile/graph/compiled/your_op.hh"
```

### Step 5: Update CMakeLists.txt

In `src/CMakeLists.txt`, add your source files:

```cmake
# Logical graph operations
set(GRAPH_LOGICAL_OPS_SRC
    "graph/logical/gemm.cc"
    "graph/logical/gelu.cc"
    "graph/logical/your_op.cc"  # Add here
)

# Compiled graph operations
set(GRAPH_COMPILED_OPS_SRC
    "graph/compiled/gemm.cc"
    "graph/compiled/gelu.cc"
    "graph/compiled/your_op.cc"  # Add here
)
```

### Step 6: Update Convenience Header

In `include/nntile/graph/graph.hh`, add includes:

```cpp
// Include logical graph operations
#include <nntile/graph/logical/gemm.hh>
#include <nntile/graph/logical/gelu.hh>
#include <nntile/graph/logical/your_op.hh>  // Add here

// Include compiled graph operations
#include <nntile/graph/compiled/gemm.hh>
#include <nntile/graph/compiled/gelu.hh>
#include <nntile/graph/compiled/your_op.hh>  // Add here
```

## Usage Example

After adding your operation, it can be used like this:

```cpp
#include <nntile/graph.hh>

using namespace nntile::graph;

// Create logical graph
LogicalGraph g("example");

// Create input tensors
auto& x = g.tensor(TensorSpec({128, 64}, DataType::FP32), "x");
auto& w = g.tensor(TensorSpec({64, 32}, DataType::FP32), "w");

// Apply operations (free functions - graph inferred from tensors)
auto& h = gemm(x, w, "h");           // C = A @ B (creates new tensor)
auto& y = gelu(h, "y");

// Compile and execute
auto compiled = CompiledGraph::compile(g);
compiled.bind_data("x", input_data);
compiled.bind_data("w", weight_data);
compiled.execute();
compiled.wait();
auto result = compiled.get_output<float>("y");
```

### In-Place Accumulation Example

```cpp
// Create tensors
auto& a = g.tensor(TensorSpec({32, 64}, DataType::FP32), "a");
auto& b = g.tensor(TensorSpec({64, 128}, DataType::FP32), "b");
auto& c = g.tensor(TensorSpec({32, 128}, DataType::FP32), "c");

// Accumulate: C = alpha * A @ B + beta * C
gemm(a, b, c, /*alpha=*/1.0, /*beta=*/0.5);  // Modifies c in-place
```

## Design Principles

1. **Free functions for operations**: Operations are free functions, not class methods.
   This enables better compilation parallelism and extensibility.

2. **Graph inferred from tensors**: Operations get the graph from input tensors via
   `tensor.graph()`, so no explicit graph parameter is needed.

3. **Two API patterns for operations**:
   - **Creating new tensor**: `LogicalGraph::TensorNode& op(inputs..., output_name)` - creates tensor then uses `add_op()`
   - **In-place accumulation**: `void op(inputs..., inout_tensor)` - uses `add_op()` with existing tensor

4. **Public builder API**: LogicalGraph exposes `add_op()` for operations to use,
   avoiding friend declarations. All output tensors must be pre-created.

5. **Separate files per operation**: Each operation has its own header and source
   files for parallel compilation.

6. **Type dispatching in compiled operations**: The compiled operation handles
   runtime type dispatching to call the appropriate NNTile tensor function.

## Data Type Support

When implementing compiled operations, consider supporting these data types:
- `FP32` (nntile::fp32_t)
- `FP32_FAST_TF32` (nntile::fp32_fast_tf32_t)
- `FP32_FAST_FP16` (nntile::fp32_fast_fp16_t)
- `FP32_FAST_BF16` (nntile::fp32_fast_bf16_t)
- `FP64` (nntile::fp64_t)
- `FP16` (nntile::fp16_t)
- `BF16` (nntile::bf16_t)

Not all operations support all types. Validate and throw appropriate errors.

## Validation in Compiled Graph

Optionally add dtype validation in `compiled_graph.cc`:

```cpp
void validate_operation_data_types(const LogicalGraph& logical)
{
    for(const auto& op : logical.ops())
    {
        DataType dtype = op->inputs()[0]->spec().dtype();

        if(op->type() == OpType::YOUR_OP)
        {
            if(dtype != DataType::FP32 && dtype != DataType::FP64)
            {
                throw std::runtime_error(
                    "YOUR_OP does not support data type " +
                    dtype_to_string(dtype));
            }
        }
    }
}
```
