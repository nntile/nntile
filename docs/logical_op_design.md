# OpNode Design: Operation Structures with execute()

This document describes the graph design where operations are represented as
structures inheriting from a base class `BaseOpNode<Graph>`, with virtual
`execute()` for dispatch instead of `OpType` enum. Inputs and outputs are
packed into each op; execution uses DataNode pointers and a mapping from
DataNode to runtime data. The templated `Graph` indicates the graph type
(TensorGraph, TileGraph); `DataNode` is `BaseDataNode<Graph>`.

Graphs derive from `BaseGraph<Graph>` (CRTP). TensorGraph and TileGraph are
thin derived classes that add type aliases and override `graph_type_name()`.

## Naming Conventions

| Concept | TensorGraph |
|---------|-------------|
| Data node | BaseDataNode\<TensorGraph\> (tensor/data node) |
| Op node | BaseOpNode\<TensorGraph\> (operation + graph node) |

## Graph Types

| Graph | DataNode | OpNode | Operates on |
|-------|----------|--------|-------------|
| TensorGraph | BaseDataNode\<TensorGraph\> | BaseOpNode\<TensorGraph\> | Tensors |
| TileGraph | BaseDataNode\<TileGraph\> | BaseOpNode\<TileGraph\> | Tiles (placeholder) |

## 1. BaseDataNode

Data nodes are templated on the graph type. The graph owns data nodes via
`unique_ptr`; ops hold raw pointers.

```cpp
// include/nntile/graph/base_data_node.hh

template<typename Graph>
class BaseDataNode
{
    friend Graph;

    NodeId id_;
    Graph* graph_;
    std::vector<Index> shape_;
    DataType dtype_;
    std::string name_;
    bool is_input_ = false;
    bool is_output_ = false;

public:
    BaseDataNode(NodeId id, Graph* graph, std::vector<Index> shape,
                 DataType dtype, const std::string& name = "");

    NodeId id() const;
    const std::string& name() const;
    DataType dtype() const;
    const std::vector<Index>& shape() const;
    Index ndim() const;
    Index dim(int idx) const;
    Index nelems() const;
    size_t size_bytes() const;
    bool is_compatible(const BaseDataNode* other) const;

    Graph* graph();
    const Graph* graph() const;

    bool is_input() const;
    bool is_output() const;
    void mark_input(bool v = true);
    void mark_output(bool v = true);

    std::string to_string() const;
};
```

TensorGraph defines `using DataNode = BaseDataNode<TensorGraph>;`.

## 2. DataType

DataType enum and utilities live in a standalone header:

```cpp
// include/nntile/graph/dtype.hh

namespace nntile::graph
{

enum class DataType
{
    FP32,
    FP32_FAST_TF32,
    FP32_FAST_FP16,
    FP32_FAST_BF16,
    FP64,
    FP16,
    BF16,
    INT64,
    INT32,
    BOOL
};

std::string dtype_to_string(DataType dtype);
size_t dtype_size(DataType dtype);

}
```

## 3. Execution Context

Instead of name-based lookup, execution uses DataNode pointers. The context
maps DataNode to runtime tensors. `ExecutionContext` is templated on the
DataNode type:

```cpp
// include/nntile/graph/execution_context.hh

template<typename DataNode>
class ExecutionContext
{
public:
    template<typename T>
    void register_tensor(
        const DataNode* node,
        std::shared_ptr<tensor::Tensor<T>> tensor);

    template<typename T>
    tensor::Tensor<T>& get_tensor(const DataNode* node);

    // Get dtype from DataNode::dtype() (no separate mapping)
    DataType get_dtype(const DataNode* node) const
    {
        return node->dtype();
    }

private:
    std::map<const DataNode*, std::shared_ptr<void>> tensor_map_;
};
```

- For TensorGraph: use `ExecutionContext<TensorGraph::DataNode>`.
- No name-based lookup; the op passes DataNode pointers directly.

---

## 4. Base Class: BaseOpNode

BaseOpNode is templated on `Graph`. It is both the operation descriptor and the
graph node. Graph is friend so it can assign `id_` when adding.

```cpp
// include/nntile/graph/base_op_node.hh

template<typename Graph>
class BaseOpNode
{
public:
    using NodeId = uint64_t;
    using DataNode = BaseDataNode<Graph>;

    virtual ~BaseOpNode() = default;

    virtual std::string op_name() const = 0;

    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }
    NodeId id() const { return id_; }

    const std::vector<DataNode*>& inputs() const { return inputs_; }
    const std::vector<DataNode*>& outputs() const { return outputs_; }

    virtual void execute(ExecutionContext<DataNode>& ctx) const = 0;
    virtual std::shared_ptr<BaseOpNode<Graph>> clone() const = 0;

protected:
    BaseOpNode() = default;

    NodeId id_ = -1;
    std::string name_;
    std::vector<DataNode*> inputs_;
    std::vector<DataNode*> outputs_;

    friend Graph;  // Graph assigns id_ when adding
};
```

- For TensorGraph: use `BaseOpNode<TensorGraph>`.
- Graph stores `std::vector<std::shared_ptr<BaseOpNode<Graph>>>` directly.
- No separate wrapper: BaseOpNode *is* the graph node (the operation and the
  node are one).

---

## 5. BaseGraph and TensorGraph

BaseGraph is a CRTP template with all common logic. TensorGraph derives from it.

```cpp
// include/nntile/graph/base_graph.hh

template<typename Graph>
class BaseGraph
{
public:
    using NodeId = uint64_t;
    using DataNode = BaseDataNode<Graph>;
    using OpNode = BaseOpNode<Graph>;

    explicit BaseGraph(const std::string& name = "");

    DataNode* data(
        std::vector<Index> shape,
        const std::string& name,
        DataType dtype = DataType::FP32);

    void add_op(
        std::shared_ptr<OpNode> op_node,
        const std::string& name = "");

    const std::string& name() const;
    size_t num_data() const;
    size_t num_ops() const;
    DataNode* get_data(const std::string& name);
    std::vector<std::string> data_names() const;
    const std::vector<std::unique_ptr<DataNode>>& data_nodes() const;
    const std::vector<std::shared_ptr<OpNode>>& ops() const;

    std::string to_string() const;
    std::string to_mermaid() const;

protected:
    virtual const char* graph_type_name() const
    {
        return "BaseGraph";
    }
};
```

```cpp
// include/nntile/graph/tensor_graph.hh

class TensorGraph : public BaseGraph<TensorGraph>
{
public:
    using DataNode = BaseDataNode<TensorGraph>;
    using OpNode = BaseOpNode<TensorGraph>;

    explicit TensorGraph(const std::string& name = "");

    void assign_op_id(OpNode* op, NodeId id);  // called by BaseGraph::add_op

protected:
    const char* graph_type_name() const override
    {
        return "TensorGraph";
    }
};
```

- BaseGraph validates that `op_node->inputs()` and `op_node->outputs()` belong
  to this graph.
- BaseGraph assigns `id_` via `Graph::assign_op_id()` and optional `name_` when
  adding.
- Data nodes: `unique_ptr`; ops: `shared_ptr`.
- Iterate over `ops()` directly; each element is an `BaseOpNode<TensorGraph>*`.

---

## 6. Example: Add Operation

### Header (`include/nntile/graph/tensor/add.hh`)

```cpp
struct TensorAddOp : BaseOpNode<TensorGraph>
{
    Scalar alpha = 1.0;
    Scalar beta = 0.0;
    TensorGraph::DataNode* x = nullptr;
    TensorGraph::DataNode* y = nullptr;
    TensorGraph::DataNode* z = nullptr;

    TensorAddOp() = default;
    TensorAddOp(
        TensorGraph::DataNode* x_,
        TensorGraph::DataNode* y_,
        TensorGraph::DataNode* z_,
        Scalar alpha_, Scalar beta_)
        : alpha(alpha_), beta(beta_), x(x_), y(y_), z(z_)
    {
        inputs_ = {x, y};
        outputs_ = {z};
    }

    std::string op_name() const override { return "ADD"; }

    void execute(
        ExecutionContext<TensorGraph::DataNode>& ctx) const override;

    std::shared_ptr<BaseOpNode<TensorGraph>> clone() const override
    {
        return std::make_shared<TensorAddOp>(*this);
    }
};

TensorGraph::DataNode* add(
    Scalar alpha,
    TensorGraph::DataNode* x,
    Scalar beta,
    TensorGraph::DataNode* y,
    const std::string& output_name);
```

### Implementation (`src/graph/tensor/add.cc`)

```cpp
TensorGraph::DataNode* add(
    Scalar alpha,
    TensorGraph::DataNode* x,
    Scalar beta,
    TensorGraph::DataNode* y,
    const std::string& output_name)
{
    // Validate inputs, create output tensor, build op, add to graph
    TensorGraph::DataNode* output = x->graph()->data(...);
    auto op = std::make_shared<TensorAddOp>(x, y, output, alpha, beta);
    x->graph()->add_op(op);
    return output;
}

void TensorAddOp::execute(
    ExecutionContext<TensorGraph::DataNode>& ctx) const
{
    DataType dtype = ctx.get_dtype(x);
    switch(dtype)
    {
        case DataType::FP32:
            run_add<nntile::fp32_t>(ctx, alpha, beta, x, y, z);
            break;
        // ... other dtypes ...
    }
}
```

---

## 7. Placeholder: TileGraph

```cpp
// include/nntile/graph/tile_graph.hh (placeholder)

class TileGraph : public BaseGraph<TileGraph>
{
public:
    using DataNode = BaseDataNode<TileGraph>;
    using OpNode = BaseOpNode<TileGraph>;

    void assign_op_id(OpNode* op, NodeId id);

protected:
    const char* graph_type_name() const override
    {
        return "TileGraph";
    }
};
```

- `BaseOpNode<TileGraph>` — operations on tile nodes.
- `ExecutionContext<TileGraph::DataNode>` — maps tile nodes to runtime tile data.

---

## 8. Summary

| Aspect | TensorGraph |
|--------|-------------|
| Graph base | `BaseGraph<Graph>` (CRTP) |
| Data node | `BaseDataNode<Graph>` |
| Op node | `BaseOpNode<Graph>` |
| ExecutionContext | `ExecutionContext<DataNode>` |
| add_op | `(shared_ptr<BaseOpNode<Graph>>, name)` |
| Inputs/outputs | In BaseOpNode |
| Data ownership | Data: `unique_ptr`; ops: `shared_ptr` |
| Tensor lookup | By DataNode* (`ctx.get_tensor(node)`) |
| Data type | `ctx.get_dtype(node)` from `node->dtype()` |
| Op storage | BaseOpNode directly; no separate wrapper |
