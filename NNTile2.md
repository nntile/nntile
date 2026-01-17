# NNTile 2.0 Architecture Design Plan

## Executive Summary

This document outlines the development plan for NNTile version 2.0, which introduces a **high-level computational graph abstraction** that enables automatic tensor distribution (FSDP/DDP-like functionality), task placement control, and multi-node distributed execution. The goal is to transform NNTile from a low-level tiled tensor framework into a production-ready distributed deep learning system while preserving the efficiency of the StarPU runtime.

---

## 1. Current Architecture Analysis

### 1.1 Existing Layer Structure

The current NNTile implementation consists of four abstraction layers:

| Layer | Location | Purpose | Current State |
|-------|----------|---------|---------------|
| **Kernel** | `src/kernel/` | Raw computational functions (CPU/CUDA) | ✅ Well-implemented |
| **StarPU** | `src/starpu/` | Task submission with data handles | ✅ Functional but lacks placement hints |
| **Tile** | `src/tile/` | Wrapper for single-tile operations | ⚠️ Largely redundant |
| **Tensor** | `src/tensor/` | Operations on tiled/distributed tensors | ⚠️ MPI disabled, manual distribution |
| **Python** | `wrappers/python/` | High-level layers, models, pipeline | ✅ Functional for single-node |

### 1.2 Identified Limitations

1. **MPI Support Disabled**: The `config.hh` file contains fake MPI functions:
   ```cpp
   static int starpu_mpi_world_size() { return 1; }
   static int starpu_mpi_world_rank() { return 0; }
   ```

2. **Manual Tensor Distribution**: Users must explicitly specify tile distribution vectors:
   ```cpp
   std::vector<int> tile_distr(grid.nelems, 0);  // All tiles on node 0
   ```

3. **No Task Placement Control**: StarPU decides task placement without hints from NNTile.

4. **Redundant Tile Layer**: Tensor-level routines bypass tile-level and call starpu-level directly:
   ```cpp
   // In tensor/gelu.cc - calls starpu directly
   starpu::gelu.submit<std::tuple<T>>(tile_traits.nelems, src_tile_handle, dst_tile_handle);
   ```

5. **No High-Level Graph Abstraction**: Users must manually handle tiling, distribution, and execution order.

---

## 2. NNTile 2.0 Architecture Vision

### 2.1 New Architectural Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Python User API (nntile.nn)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                    High-Level Graph (NEW - nntile.graph)                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │
│  │ComputeGraph│  │DistStrategy│  │ TensorSpec │  │ ExecutionPolicy        │ │
│  │   (DAG)    │  │ (FSDP/DDP) │  │(ShapeHints)│  │(Node/Worker Assignment)│ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                       Tensor Level (Enhanced)                               │
│     - Accepts ExecutionContext with node/worker hints                       │
│     - Automatic tile-to-node mapping based on DistStrategy                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                       StarPU Level (Enhanced)                               │
│     - Extended submit() with execution_node and execution_worker params     │
│     - Task priority hints, worker binding support                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                          Kernel Level (Unchanged)                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Design Principles

1. **Graph-First Design**: Define computation as a DAG before execution
2. **Declarative Distribution**: Specify distribution strategy (DDP/FSDP/TP) at model level
3. **Automatic Tiling**: Derive optimal tile shapes from tensor specs and hardware
4. **Explicit Placement**: Allow fine-grained control over task-to-node/worker mapping
5. **Backward Compatibility**: Existing code should work with minimal changes

---

## 3. Tile-Level Layer: Keep and Refactor

### 3.1 Decision: **Keep Tile-Level**

The tile-level layer provides important architectural benefits:

1. **Runtime Abstraction**: Enables future support for alternative runtime systems (e.g., TaskFlow, HPX) by isolating StarPU-specific code
2. **Single-Tile Testing**: Clean interface for unit testing and debugging individual tile operations
3. **Conceptual Clarity**: Maintains clear separation between single-tile and distributed tensor operations
4. **Validation Boundary**: Natural place for tile-specific invariant checks

### 3.2 Refactoring Goals

#### 3.2.1 Header-Only Implementation

Both **tile-level AND starpu-level** should become header-only where feasible:

**Rationale:**
- Enables inlining for zero-overhead abstraction
- Reduces compilation units and link times
- Template-heavy code benefits from header-only design
- Simplifies build system

**Structure:**
```
include/nntile/
├── kernel/          # Declarations only (implementations in .cc/.cu)
├── starpu/          # Header-only templates + inline functions
│   ├── gelu.hh      # Contains submit() implementation
│   └── ...
├── tile/            # Header-only, calls starpu::
│   ├── gelu.hh      # Contains gelu_async() and gelu()
│   └── ...
└── tensor/          # May remain .cc for complex logic
```

**Note:** Kernel-level cannot be header-only due to CUDA compilation requirements.

#### 3.2.2 Task Handle Return for Proper Blocking

**Problem with current approach:**
```cpp
// Current: waits for ALL tasks - too broad!
template<typename T>
void gelu(const Tile<T> &src, const Tile<T> &dst) {
    gelu_async<T>(src, dst);
    starpu_task_wait_for_all();  // BAD: blocks unrelated tasks
}
```

**Solution:** Return a task handle wrapper that allows waiting on the specific task.

### 3.3 StarPU Task Waiting Mechanism

StarPU provides `starpu_task_wait(struct starpu_task *task)` to wait for a specific task. However, the current `starpu_task_insert()` API doesn't return the task handle.

**Correct approach:** Use explicit task creation:

```cpp
// Instead of starpu_task_insert(), use:
struct starpu_task *task = starpu_task_create();
task->cl = &codelet;
task->handles[0] = src.get();
task->handles[1] = dst.get();
task->cl_arg = args;
task->cl_arg_size = sizeof(*args);
task->detach = 0;  // IMPORTANT: makes task waitable

int ret = starpu_task_submit(task);
// Now we can call: starpu_task_wait(task);
```

### 3.4 Task Handle Wrapper Design

```cpp
namespace nntile::starpu {

//! RAII wrapper for StarPU task handle
class TaskHandle {
private:
    struct starpu_task *task_ = nullptr;
    bool detached_ = false;

public:
    //! Construct from raw StarPU task
    explicit TaskHandle(struct starpu_task *task) : task_(task) {}
    
    //! Move-only semantics
    TaskHandle(TaskHandle&& other) noexcept 
        : task_(other.task_), detached_(other.detached_) {
        other.task_ = nullptr;
    }
    TaskHandle& operator=(TaskHandle&& other) noexcept {
        if (this != &other) {
            wait();  // Wait for current task before reassignment
            task_ = other.task_;
            detached_ = other.detached_;
            other.task_ = nullptr;
        }
        return *this;
    }
    
    //! No copy
    TaskHandle(const TaskHandle&) = delete;
    TaskHandle& operator=(const TaskHandle&) = delete;
    
    //! Destructor waits if not detached
    ~TaskHandle() {
        if (task_ && !detached_) {
            wait();
        }
    }
    
    //! Wait for this specific task to complete
    void wait() {
        if (task_) {
            starpu_task_wait(task_);
            task_ = nullptr;  // Task is freed after wait
        }
    }
    
    //! Detach task (fire-and-forget, StarPU manages lifetime)
    void detach() {
        detached_ = true;
        task_ = nullptr;
    }
    
    //! Check if task is complete without blocking
    bool is_complete() const {
        if (!task_) return true;
        return starpu_task_finished(task_) != 0;
    }
    
    //! Get raw handle (use with caution)
    struct starpu_task* get() const { return task_; }
    
    //! Check if handle is valid
    explicit operator bool() const { return task_ != nullptr; }
};

//! Collection of task handles for batch waiting
class TaskGroup {
private:
    std::vector<TaskHandle> tasks_;

public:
    void add(TaskHandle&& task) {
        tasks_.push_back(std::move(task));
    }
    
    //! Wait for all tasks in this group
    void wait_all() {
        for (auto& task : tasks_) {
            task.wait();
        }
        tasks_.clear();
    }
    
    //! Wait for any one task to complete, return its index
    size_t wait_any() {
        while (true) {
            for (size_t i = 0; i < tasks_.size(); ++i) {
                if (tasks_[i].is_complete()) {
                    tasks_[i].wait();  // Finalize
                    return i;
                }
            }
            // Brief yield to avoid busy-spinning
            starpu_do_schedule();
        }
    }
    
    size_t size() const { return tasks_.size(); }
    bool empty() const { return tasks_.empty(); }
};

} // namespace nntile::starpu
```

### 3.5 Updated StarPU-Level Submit Interface

```cpp
namespace nntile::starpu {

template<typename T>
class Gelu<std::tuple<T>> {
public:
    Codelet codelet;
    
    //! Submit task and return handle for optional waiting
    TaskHandle submit(Index nelems, Handle src, Handle dst,
                      const TaskExecutionHints& hints = {}) 
    {
        // Allocate arguments
        args_t *args = (args_t *)std::malloc(sizeof(*args));
        args->nelems = nelems;
        
        // Create task explicitly (not starpu_task_insert)
        struct starpu_task *task = starpu_task_create();
        task->cl = &codelet;
        task->detach = 0;  // Make waitable
        task->destroy = 1; // Auto-destroy after completion
        
        // Set data handles
        task->handles[0] = src.get();
        task->handles[1] = dst.get();
        task->modes[0] = STARPU_R;
        task->modes[1] = STARPU_W;
        task->nbuffers = 2;
        
        // Set codelet arguments
        task->cl_arg = args;
        task->cl_arg_size = sizeof(*args);
        task->cl_arg_free = 1;  // Auto-free args
        
        // Apply execution hints
        if (hints.target_node >= 0) {
            task->execute_on_a_specific_worker = 1;
            task->workerid = hints.target_worker;  // Or map node to worker
        }
        if (hints.priority != 0) {
            task->priority = hints.priority;
        }
        
        // Submit
        int ret = starpu_task_submit(task);
        if (ret != 0) {
            throw std::runtime_error("Error in gelu task submission");
        }
        
        return TaskHandle(task);
    }
};

} // namespace nntile::starpu
```

### 3.6 Updated Tile-Level Interface

```cpp
namespace nntile::tile {

//! Async tile-wise GeLU - returns task handle
template<typename T>
inline starpu::TaskHandle gelu_async(const Tile<T> &src, const Tile<T> &dst) {
    // Validation
    if (src.nelems != dst.nelems) {
        throw std::runtime_error("Tile size mismatch in gelu");
    }
    // Submit and return handle
    return starpu::gelu.submit<std::tuple<T>>(src.nelems, src.handle, dst.handle);
}

//! Blocking tile-wise GeLU - waits only for THIS task
template<typename T>
inline void gelu(const Tile<T> &src, const Tile<T> &dst) {
    auto task = gelu_async<T>(src, dst);
    task.wait();  // Wait for this specific task only!
}

} // namespace nntile::tile
```

### 3.7 Usage Patterns

```cpp
// Pattern 1: Fire-and-forget (async)
auto task = tile::gelu_async(src, dst);
task.detach();  // Don't wait

// Pattern 2: Wait immediately (blocking)
tile::gelu(src, dst);  // Blocks until THIS task completes

// Pattern 3: Batch submission then wait
starpu::TaskGroup group;
for (int i = 0; i < n_tiles; ++i) {
    group.add(tile::gelu_async(src[i], dst[i]));
}
group.wait_all();  // Wait for all tasks in this group

// Pattern 4: Pipeline with dependencies
auto task1 = tile::gelu_async(a, b);
auto task2 = tile::relu_async(b, c);  // StarPU handles dependency via data handles
task2.wait();  // Waiting for task2 implicitly waits for task1 due to data dependency

// Pattern 5: Tensor-level collecting tasks (for future synchronization)
template<typename T>
starpu::TaskGroup gelu_async(const Tensor<T>& src, const Tensor<T>& dst) {
    starpu::TaskGroup tasks;
    for (Index i = 0; i < src.grid.nelems; ++i) {
        auto task = starpu::gelu.submit<std::tuple<T>>(...);
        tasks.add(std::move(task));
    }
    return tasks;  // Caller can wait or let destructor wait
}
```

---

## 4. Detailed Component Design

### 4.1 High-Level Graph (`nntile::graph` namespace)

#### 4.1.1 ComputeGraph Class

```cpp
namespace nntile::graph {

class ComputeGraph {
public:
    // Graph building API
    TensorNode& input(const TensorSpec& spec, const std::string& name);
    TensorNode& parameter(const TensorSpec& spec, const std::string& name);
    TensorNode& output(TensorNode& node, const std::string& name);
    
    // Operation nodes (lazy - no computation yet)
    TensorNode& matmul(TensorNode& a, TensorNode& b, TransOp trans_a, TransOp trans_b);
    TensorNode& gelu(TensorNode& input);
    TensorNode& layernorm(TensorNode& input, TensorNode& gamma, TensorNode& beta);
    // ... more operations
    
    // Distribution configuration
    void set_distribution_strategy(DistributionStrategy strategy);
    void set_execution_policy(ExecutionPolicy policy);
    
    // Instantiation - creates actual tensors with proper tiling/distribution
    GraphInstance instantiate(const ExecutionContext& ctx);
    
    // Analysis
    std::vector<TensorNode*> topological_order() const;
    size_t estimate_memory_usage() const;
    
private:
    std::vector<std::unique_ptr<TensorNode>> nodes_;
    std::vector<std::unique_ptr<OpNode>> operations_;
    DistributionStrategy dist_strategy_;
    ExecutionPolicy exec_policy_;
};

} // namespace nntile::graph
```

#### 4.1.2 TensorSpec Class

```cpp
struct TensorSpec {
    std::vector<Index> shape;           // Logical shape
    std::vector<Index> tile_hint;       // Suggested tile size (optional)
    DataType dtype;                      // fp32, bf16, etc.
    DistributionHint dist_hint;         // REPLICATED, SHARDED_DIM0, etc.
    std::string name;                    // For debugging
    
    // Factory methods
    static TensorSpec parameter(std::vector<Index> shape, DataType dtype);
    static TensorSpec activation(std::vector<Index> shape, DataType dtype);
};
```

#### 4.1.3 DistributionStrategy Class

```cpp
enum class ParallelismMode {
    DDP,                 // Data Distributed Parallel - replicate model, shard data
    FSDP,                // Fully Sharded Data Parallel - shard everything
    TENSOR_PARALLEL,     // Shard specific tensor dimensions
    PIPELINE_PARALLEL,   // Shard layers across nodes
    HYBRID               // Combination of above
};

struct DistributionStrategy {
    ParallelismMode mode;
    
    // DDP settings
    int data_parallel_size = 1;
    
    // FSDP settings
    int shard_degree = 1;           // Number of shards per tensor
    bool shard_optimizer_states = true;
    bool shard_gradients = true;
    
    // Tensor parallel settings  
    std::vector<int> tensor_parallel_dims;  // Which dims to shard
    int tensor_parallel_size = 1;
    
    // Pipeline parallel settings
    int pipeline_stages = 1;
    
    // Factory methods
    static DistributionStrategy ddp(int world_size);
    static DistributionStrategy fsdp(int world_size, int shard_degree);
    static DistributionStrategy tensor_parallel(int world_size, std::vector<int> dims);
};
```

#### 4.1.4 ExecutionPolicy Class

```cpp
struct ExecutionPolicy {
    // Task placement preferences
    enum class PlacementStrategy {
        AUTO,           // Let StarPU decide
        OWNER_COMPUTES, // Task runs where output data lives
        AFFINITY_BASED, // Consider data locality
        EXPLICIT        // User specifies mapping
    };
    
    PlacementStrategy placement = PlacementStrategy::OWNER_COMPUTES;
    
    // Worker binding
    bool bind_to_gpu = true;
    std::vector<int> preferred_gpus;  // Empty = use all
    
    // Task priority
    int base_priority = 0;
    
    // Memory management
    bool enable_offloading = false;
    float offload_threshold = 0.8;  // Offload when GPU memory > 80%
    
    // Execution hints for StarPU
    bool enable_commute = false;    // Allow commutative task reordering
    bool prefetch_data = true;
};
```

### 4.2 Enhanced StarPU Level

#### 4.2.1 Extended Submit Interface

Current signature:
```cpp
void Gelu<std::tuple<T>>::submit(Index nelems, Handle src, Handle dst);
```

Enhanced signature:
```cpp
struct TaskExecutionHints {
    int target_node = -1;           // -1 = any node
    int target_worker = -1;         // -1 = any worker
    int priority = 0;
    bool prefetch_inputs = true;
    std::vector<Handle> prefetch_handles;  // Additional handles to prefetch
};

void Gelu<std::tuple<T>>::submit(
    Index nelems, 
    Handle src, 
    Handle dst,
    const TaskExecutionHints& hints = {}
);
```

#### 4.2.2 Implementation Changes

In `src/starpu/gelu.cc`:
```cpp
template<typename T>
void Gelu<std::tuple<T>>::submit(
    Index nelems,
    Handle src,
    Handle dst,
    const TaskExecutionHints& hints
) {
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    
    // Build task insertion arguments
    std::vector<int> task_args;
    task_args.push_back(STARPU_R);
    task_args.push_back(reinterpret_cast<int>(src.get()));
    task_args.push_back(STARPU_W);
    task_args.push_back(reinterpret_cast<int>(dst.get()));
    task_args.push_back(STARPU_CL_ARGS);
    task_args.push_back(reinterpret_cast<int>(args));
    task_args.push_back(sizeof(*args));
    
    // Add execution hints if specified
    if (hints.target_node >= 0) {
        task_args.push_back(STARPU_EXECUTE_ON_NODE);
        task_args.push_back(hints.target_node);
    }
    if (hints.target_worker >= 0) {
        task_args.push_back(STARPU_EXECUTE_ON_WORKER);
        task_args.push_back(hints.target_worker);
    }
    if (hints.priority != 0) {
        task_args.push_back(STARPU_PRIORITY);
        task_args.push_back(hints.priority);
    }
    
    task_args.push_back(0);  // Terminator
    
    int ret = starpu_task_insert(&codelet, /* variadic from task_args */);
    if(ret != 0) {
        throw std::runtime_error("Error in gelu task submission");
    }
}
```

### 4.3 Enhanced Tensor Level

#### 4.3.1 Execution Context

```cpp
struct ExecutionContext {
    int mpi_rank;
    int mpi_size;
    DistributionStrategy dist_strategy;
    ExecutionPolicy exec_policy;
    
    // Get target node for a tile based on distribution strategy
    int get_tile_node(const TensorTraits& tensor, Index tile_idx) const;
    
    // Get target worker for a tile
    int get_tile_worker(const TensorTraits& tensor, Index tile_idx) const;
    
    // Build execution hints for a tile
    TaskExecutionHints get_hints(const TensorTraits& tensor, Index tile_idx) const;
};
```

#### 4.3.2 Enhanced Tensor Operations

Current signature:
```cpp
template<typename T>
void gelu_async(const Tensor<T> &src, const Tensor<T> &dst);
```

Enhanced signature:
```cpp
template<typename T>
void gelu_async(
    const Tensor<T> &src, 
    const Tensor<T> &dst,
    const ExecutionContext& ctx = ExecutionContext::default_context()
);
```

Implementation:
```cpp
template<typename T>
void gelu_async(const Tensor<T> &src, const Tensor<T> &dst, const ExecutionContext& ctx) {
    // Validation
    if(dst.ndim != src.ndim) {
        throw std::runtime_error("dst.ndim != src.ndim");
    }
    // ... shape checks ...
    
    int mpi_rank = ctx.mpi_rank;
    for(Index i = 0; i < src.grid.nelems; ++i) {
        auto src_tile_handle = src.get_tile_handle(i);
        auto dst_tile_handle = dst.get_tile_handle(i);
        int dst_tile_rank = ctx.get_tile_node(dst, i);
        
        // Transfer data to target node
        src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        
        // Execute only on destination node
        if(mpi_rank == dst_tile_rank) {
            auto tile_traits = src.get_tile_traits(i);
            auto hints = ctx.get_hints(dst, i);
            starpu::gelu.submit<std::tuple<T>>(
                tile_traits.nelems, 
                src_tile_handle,
                dst_tile_handle,
                hints
            );
        }
        dst_tile_handle.mpi_flush();
    }
}
```

---

## 5. Python API Design

### 5.1 Graph Builder API

```python
# nntile/graph/__init__.py

class ComputeGraph:
    def __init__(self, name: str = ""):
        self._nodes = []
        self._ops = []
        self._dist_strategy = None
        self._exec_policy = None
    
    def input(self, shape: List[int], dtype: str = "fp32", name: str = "") -> TensorNode:
        """Declare an input tensor."""
        
    def parameter(self, shape: List[int], dtype: str = "fp32", name: str = "") -> TensorNode:
        """Declare a trainable parameter."""
        
    def constant(self, value: np.ndarray, name: str = "") -> TensorNode:
        """Declare a constant tensor."""
    
    # Operations
    def matmul(self, a: TensorNode, b: TensorNode, 
               trans_a: bool = False, trans_b: bool = False) -> TensorNode:
        """Matrix multiplication."""
        
    def gelu(self, x: TensorNode) -> TensorNode:
        """GeLU activation."""
        
    def layer_norm(self, x: TensorNode, gamma: TensorNode, 
                   beta: TensorNode, eps: float = 1e-5) -> TensorNode:
        """Layer normalization."""
    
    # Distribution
    def set_distribution(self, strategy: DistributionStrategy):
        """Set the distribution strategy for this graph."""
        
    def set_execution_policy(self, policy: ExecutionPolicy):
        """Set the execution policy."""
    
    # Instantiation
    def instantiate(self, ctx: Optional[ExecutionContext] = None) -> "GraphInstance":
        """Create actual tensors and prepare for execution."""


class DistributionStrategy:
    @staticmethod
    def ddp(world_size: int) -> "DistributionStrategy":
        """Data Distributed Parallel strategy."""
        
    @staticmethod  
    def fsdp(world_size: int, shard_degree: int = -1) -> "DistributionStrategy":
        """Fully Sharded Data Parallel strategy."""
        
    @staticmethod
    def tensor_parallel(world_size: int, dims: List[int]) -> "DistributionStrategy":
        """Tensor parallel strategy."""


class ExecutionPolicy:
    placement: str = "owner_computes"  # "auto", "owner_computes", "affinity", "explicit"
    preferred_gpus: List[int] = None
    enable_offloading: bool = False
    prefetch: bool = True
```

### 5.2 Usage Example

```python
import nntile
from nntile.graph import ComputeGraph, DistributionStrategy, ExecutionPolicy

# Initialize NNTile with MPI
nntile.init(mpi=True)

# Define computation graph
graph = ComputeGraph("transformer_block")

# Declare tensors
x = graph.input([seq_len, batch_size, embed_dim], name="input")
wq = graph.parameter([embed_dim, embed_dim], name="W_query")
wk = graph.parameter([embed_dim, embed_dim], name="W_key")
wv = graph.parameter([embed_dim, embed_dim], name="W_value")
wo = graph.parameter([embed_dim, embed_dim], name="W_out")
gamma = graph.parameter([embed_dim], name="ln_gamma")
beta = graph.parameter([embed_dim], name="ln_beta")

# Define computation
q = graph.matmul(x, wq)
k = graph.matmul(x, wk)
v = graph.matmul(x, wv)
attn = graph.scaled_dot_product_attention(q, k, v)
out = graph.matmul(attn, wo)
out = graph.add(out, x)  # Residual
out = graph.layer_norm(out, gamma, beta)

# Configure distribution for 8 GPUs across 2 nodes
dist = DistributionStrategy.fsdp(world_size=8, shard_degree=4)
graph.set_distribution(dist)

policy = ExecutionPolicy()
policy.placement = "owner_computes"
policy.enable_offloading = True
graph.set_execution_policy(policy)

# Instantiate - creates actual tensors with proper tiling
instance = graph.instantiate()

# Load weights
instance.load_parameters("checkpoint.pt")

# Execute
instance.forward()

# Get outputs
output = instance.get_tensor("output")

nntile.shutdown()
```

---

## 6. Development Roadmap

### Phase 1: Foundation (Parallel Track A) - Re-enable MPI Support

**Goal**: Restore StarPU-MPI functionality for multi-node execution

**Tasks**:
1. [ ] Remove fake MPI stubs from `config.hh`
2. [ ] Enable `starpu_mpi.h` includes
3. [ ] Implement proper `starpu_mpi_init()` / `starpu_mpi_shutdown()`
4. [ ] Test single-node MPI functionality
5. [ ] Implement proper MPI data transfers in tensor operations
6. [ ] Add MPI collective operations (allreduce, broadcast)

**Files to modify**:
- `include/nntile/starpu/config.hh`
- `src/context.cc`
- `src/tensor/*.cc` (all tensor operations)

### Phase 2: StarPU Enhancements (Parallel Track B)

**Goal**: Add task placement hints to StarPU level

**Tasks**:
1. [ ] Define `TaskExecutionHints` structure
2. [ ] Update `Codelet::submit()` interface to accept hints
3. [ ] Implement hint passing to `starpu_task_insert()`
4. [ ] Add worker binding support
5. [ ] Add task priority support
6. [ ] Update all StarPU-level operations (70+ files)

**Files to modify**:
- `include/nntile/starpu/codelet.hh`
- `src/starpu/*.cc` (all starpu operations)

### Phase 3: Tensor Level Enhancements (Parallel Track C)

**Goal**: Propagate execution hints through tensor operations

**Tasks**:
1. [ ] Define `ExecutionContext` structure
2. [ ] Update tensor operation signatures to accept context
3. [ ] Implement tile-to-node mapping logic
4. [ ] Add context-aware data transfers
5. [ ] Update all tensor operations

**Files to modify**:
- `include/nntile/tensor/*.hh`
- `src/tensor/*.cc`

### Phase 4: Graph Abstraction (Parallel Track D)

**Goal**: Implement high-level graph API

**Tasks**:
1. [ ] Define `TensorNode` and `OpNode` classes
2. [ ] Implement `ComputeGraph` class
3. [ ] Implement `TensorSpec` and shape inference
4. [ ] Implement `DistributionStrategy` class
5. [ ] Implement `ExecutionPolicy` class
6. [ ] Implement graph instantiation logic
7. [ ] Add automatic tiling algorithm

**New files**:
- `include/nntile/graph/tensor_node.hh`
- `include/nntile/graph/op_node.hh`
- `include/nntile/graph/compute_graph.hh`
- `include/nntile/graph/distribution_strategy.hh`
- `include/nntile/graph/execution_policy.hh`
- `src/graph/*.cc`

### Phase 5: Python Bindings (Parallel Track E)

**Goal**: Expose graph API to Python

**Tasks**:
1. [ ] Add pybind11 bindings for graph classes
2. [ ] Implement Python-friendly API wrappers
3. [ ] Add examples and documentation
4. [ ] Integrate with existing layer/model classes

**Files to modify**:
- `wrappers/python/nntile/nntile_core.cc`
- `wrappers/python/nntile/graph/__init__.py` (new)

### Phase 6: Distribution Strategies (Sequential, depends on Phase 4)

**Goal**: Implement FSDP/DDP/TP distribution strategies

**Tasks**:
1. [ ] Implement DDP strategy (replicate model, shard data)
2. [ ] Implement FSDP strategy (shard parameters and gradients)
3. [ ] Implement tensor parallelism
4. [ ] Implement pipeline parallelism
5. [ ] Add gradient synchronization primitives
6. [ ] Add optimizer state sharding

### Phase 7: Refactor to Header-Only and Task Handle Returns (Parallel Track F)

**Goal**: Make starpu-level and tile-level header-only with proper task waiting

**Tasks**:
1. [ ] Implement `TaskHandle` RAII wrapper class
2. [ ] Implement `TaskGroup` for batch task management
3. [ ] Refactor starpu-level `submit()` to use `starpu_task_create()` + `starpu_task_submit()`
4. [ ] Update starpu-level `submit()` to return `TaskHandle`
5. [ ] Move starpu-level implementations to headers (template + inline)
6. [ ] Update tile-level to return `TaskHandle` from async functions
7. [ ] Move tile-level implementations to headers
8. [ ] Update blocking tile functions to use `task.wait()` instead of `starpu_task_wait_for_all()`
9. [ ] Update tensor-level to optionally collect `TaskGroup` for synchronization
10. [ ] Remove empty `.cc` files from `src/starpu/` and `src/tile/`

**New files**:
- `include/nntile/starpu/task_handle.hh`

**Files to modify**:
- `include/nntile/starpu/*.hh` (move implementations from .cc)
- `include/nntile/tile/*.hh` (move implementations from .cc)
- `src/tensor/*.cc` (update to use TaskGroup where beneficial)

---

## 7. Parallel Development Strategy

The following tracks can be developed **in parallel** by different teams:

```
Track A (MPI)              ─────────────────────────────────┐
Track B (StarPU Hints)     ─────────────────────────────────┤
Track C (Tensor Ctx)       ───────────────────────┐         │
Track D (Graph API)        ───────────────────────┼─────────┼───> Integration
Track E (Python)           ───────────────────────┘         │
Track F (Header-Only+Task) ─────────────────────────────────┘
                                                            │
                                      Phase 6 (Dist) ───────┘
```

**Dependencies**:
- Track C depends on Track B (needs `TaskExecutionHints`)
- Track F (Header-Only + TaskHandle) is **independent** and can start immediately
- Phase 6 depends on Track A (MPI), Track C (ExecutionContext), Track D (DistributionStrategy)
- Track F should complete before Phase 6 to ensure proper task synchronization in distributed setting

**Recommended Priority for Track F**:
Track F (Header-Only + TaskHandle) is foundational and should be prioritized early because:
1. `TaskHandle` enables correct blocking semantics throughout the codebase
2. Header-only design reduces build times for all subsequent development
3. Other tracks will benefit from cleaner task management API

---

## 8. API Migration Guide

### 8.1 For Existing Users

**Before (NNTile 1.x)**:
```python
# Manual tensor creation with explicit distribution
traits = TensorTraits([seq_len, batch_size], [seq_tile, batch_tile])
distr = [0] * traits.grid.nelems  # All on node 0
x = Tensor_fp32(traits, distr)

# Manual forward/backward
model.forward_async()
loss.calc_async()
model.backward_async()
```

**After (NNTile 2.0)**:
```python
# Graph-based definition
graph = ComputeGraph()
x = graph.input([seq_len, batch_size])
# ... define model in graph ...

# Automatic distribution
graph.set_distribution(DistributionStrategy.fsdp(world_size=8))
instance = graph.instantiate()

# Same execution API
instance.forward()
instance.backward()
```

### 8.2 Backward Compatibility

NNTile 2.0 will maintain backward compatibility:
- Existing tensor-level API continues to work
- `ExecutionContext` parameter is optional with default values
- Python model/layer classes remain unchanged
- New graph API is additive, not replacing existing functionality

---

## 9. Testing Strategy

### 9.1 Unit Tests
- Test each new class independently
- Mock MPI for single-machine testing
- Verify hint propagation through layers

### 9.2 Integration Tests
- Multi-node tests with real MPI
- Distribution strategy correctness tests
- Performance regression tests

### 9.3 Benchmark Suite
- Compare against PyTorch DDP/FSDP
- Measure scaling efficiency
- Profile communication overhead

---

## 10. Open Questions and Future Work

1. **Memory Estimation**: How to accurately estimate memory for automatic tiling?
2. **Heterogeneous Clusters**: Support for mixed GPU types?
3. **Checkpoint Format**: Compatible with PyTorch/Megatron checkpoints?
4. **Dynamic Shapes**: Support for variable sequence lengths?
5. **Gradient Checkpointing**: Integration with activation checkpointing?
6. **Mixed Precision**: Automatic mixed precision in graph level?

---

## 11. Conclusion

NNTile 2.0 introduces a transformative high-level graph abstraction that enables:

1. **Automatic Distribution**: FSDP/DDP/TP without manual tensor partitioning
2. **Explicit Placement**: Fine-grained control over task execution location
3. **Multi-Node Scaling**: True distributed training across nodes
4. **Simplified API**: Declarative model definition with automatic optimization
5. **Proper Task Synchronization**: `TaskHandle` wrapper enables waiting on specific tasks instead of all tasks
6. **Runtime Abstraction**: Preserved tile-level enables future support for alternative runtimes (TaskFlow, HPX)

### Key Architectural Decisions

1. **Keep Tile-Level**: The tile-level is retained for runtime abstraction and clean single-tile testing interface
2. **Header-Only Design**: Both starpu-level and tile-level become header-only for zero-overhead abstraction and faster builds
3. **Task Handle Returns**: All async operations return `TaskHandle` for precise synchronization control

The phased approach allows parallel development across 6 tracks while maintaining backward compatibility. By implementing these changes, NNTile will evolve from a low-level tiled tensor library into a production-ready distributed deep learning framework competitive with state-of-the-art systems like DeepSpeed and Megatron-LM.
