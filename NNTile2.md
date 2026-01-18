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

#### 3.2.1 Keep C++ Source Files

Both **tile-level and runtime-level** (formerly starpu-level) keep their `.cc` source files:

**Rationale:**
- Overhead is negligible (does not interfere with actual computations in tasks)
- Maintains clear separation of interface and implementation
- Faster incremental builds (changes don't trigger full recompilation)
- Easier debugging with clear compilation units

#### 3.2.2 Runtime Abstraction Layer

Rename and restructure `starpu/` to a generic `runtime/` layer with pluggable backends:

**New Directory Structure:**
```
src/
├── kernel/              # Unchanged - CPU/CUDA implementations
├── runtime/             # NEW - Runtime abstraction layer
│   ├── backend.cc       # Abstract backend interface
│   ├── task_handle.cc   # Abstract task handle
│   ├── data_handle.cc   # Abstract data handle
│   ├── starpu/          # StarPU backend implementation
│   │   ├── backend.cc
│   │   ├── gelu.cc
│   │   ├── gemm.cc
│   │   └── ...
│   ├── taskflow/        # Future: TaskFlow backend
│   │   └── ...
│   └── serial/          # Future: Serial/debug backend
│       └── ...
├── tile/                # Unchanged - uses runtime:: interface
└── tensor/              # Unchanged - uses tile:: or runtime::

include/nntile/
├── kernel/              # Unchanged
├── runtime/             # NEW - Runtime-agnostic headers
│   ├── backend.hh       # Abstract backend interface
│   ├── task_handle.hh   # Abstract task handle
│   ├── data_handle.hh   # Abstract data handle
│   ├── codelet.hh       # Abstract codelet definition
│   ├── gelu.hh          # Runtime-agnostic operation interface
│   ├── gemm.hh
│   └── ...
│   ├── starpu/          # StarPU-specific headers
│   │   ├── backend.hh
│   │   └── ...
│   └── taskflow/        # Future
├── tile/                # Uses runtime:: namespace
└── tensor/              # Uses runtime:: or tile::
```

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
runtime::TaskGroup group;
for (int i = 0; i < n_tiles; ++i) {
    group.add(tile::gelu_async(src[i], dst[i]));
}
group.wait_all();  // Wait for all tasks in this group

// Pattern 4: Pipeline with dependencies
auto task1 = tile::gelu_async(a, b);
auto task2 = tile::relu_async(b, c);  // Runtime handles dependency via data handles
task2.wait();  // Waiting for task2 implicitly waits for task1 due to data dependency

// Pattern 5: Tensor-level collecting tasks (for future synchronization)
template<typename T>
runtime::TaskGroup gelu_async(const Tensor<T>& src, const Tensor<T>& dst) {
    runtime::TaskGroup tasks;
    for (Index i = 0; i < src.grid.nelems; ++i) {
        auto task = runtime::gelu.submit<std::tuple<T>>(...);
        tasks.add(std::move(task));
    }
    return tasks;  // Caller can wait or let destructor wait
}
```

---

## 3.8 Runtime Abstraction Layer Design

### 3.8.1 Design Goals

1. **Runtime Agnostic Interface**: Tile-level and tensor-level code should not know which runtime is being used
2. **Compile-Time Selection**: Choose runtime at build time for zero-overhead dispatch
3. **Runtime Selection** (Optional): Choose runtime at initialization for maximum flexibility
4. **Extensibility**: Easy to add new backends (TaskFlow, HPX, serial/debug)

### 3.8.2 Backend Selection Strategies

#### Option A: Compile-Time Selection (Recommended Default)

```cmake
# CMakeLists.txt
option(NNTILE_RUNTIME_STARPU "Use StarPU runtime" ON)
option(NNTILE_RUNTIME_TASKFLOW "Use TaskFlow runtime" OFF)
option(NNTILE_RUNTIME_SERIAL "Use Serial runtime (debug)" OFF)

# Only one can be enabled
if(NNTILE_RUNTIME_STARPU)
    add_compile_definitions(NNTILE_RUNTIME=starpu)
    add_subdirectory(src/runtime/starpu)
elseif(NNTILE_RUNTIME_TASKFLOW)
    add_compile_definitions(NNTILE_RUNTIME=taskflow)
    add_subdirectory(src/runtime/taskflow)
endif()
```

**Pros**: Zero overhead, direct function calls, dead code elimination
**Cons**: Must recompile to switch runtimes

#### Option B: Runtime Selection via Polymorphism

```cpp
// At initialization
nntile::runtime::set_backend(nntile::runtime::BackendType::StarPU);
// or
nntile::runtime::set_backend(nntile::runtime::BackendType::TaskFlow);
```

**Pros**: Single binary works with multiple runtimes, useful for testing/benchmarking
**Cons**: Virtual call overhead (negligible for task submission)

#### Option C: Hybrid Approach (Recommended)

- **Default**: Compile-time selection for production (Option A)
- **Optional**: Build with `NNTILE_RUNTIME_DYNAMIC=ON` for runtime selection (Option B)

```cmake
option(NNTILE_RUNTIME_DYNAMIC "Enable runtime backend selection" OFF)

if(NNTILE_RUNTIME_DYNAMIC)
    add_compile_definitions(NNTILE_RUNTIME_DYNAMIC)
    # Build all enabled backends
    add_subdirectory(src/runtime/starpu)
    add_subdirectory(src/runtime/taskflow)
endif()
```

### 3.8.3 Abstract Interface Design

#### Backend Interface

```cpp
// include/nntile/runtime/backend.hh
namespace nntile::runtime {

//! Supported backend types
enum class BackendType {
    StarPU,
    TaskFlow,
    Serial,  // For debugging/testing
    Auto     // Auto-detect best available
};

//! Abstract backend interface
class Backend {
public:
    virtual ~Backend() = default;
    
    //! Initialize the runtime
    virtual void init(int argc, char* argv[]) = 0;
    
    //! Shutdown the runtime
    virtual void shutdown() = 0;
    
    //! Get backend type
    virtual BackendType type() const = 0;
    
    //! Get number of workers (CPUs + GPUs)
    virtual int num_workers() const = 0;
    
    //! Get number of CPU workers
    virtual int num_cpu_workers() const = 0;
    
    //! Get number of GPU workers
    virtual int num_gpu_workers() const = 0;
    
    //! Wait for all submitted tasks
    virtual void wait_for_all() = 0;
    
    //! Pause task submission
    virtual void pause() = 0;
    
    //! Resume task submission
    virtual void resume() = 0;
    
    //! Factory for creating data handles
    virtual std::unique_ptr<DataHandle> create_data_handle(size_t size) = 0;
    virtual std::unique_ptr<DataHandle> create_data_handle(void* ptr, size_t size) = 0;
};

//! Global backend access
Backend& get_backend();
void set_backend(BackendType type);
void set_backend(std::unique_ptr<Backend> backend);

} // namespace nntile::runtime
```

#### Data Handle Interface

```cpp
// include/nntile/runtime/data_handle.hh
namespace nntile::runtime {

//! Data access modes
enum class AccessMode {
    Read,
    Write,
    ReadWrite,
    Reduce
};

//! Abstract data handle interface
class DataHandle {
public:
    virtual ~DataHandle() = default;
    
    //! Get raw pointer (backend-specific)
    virtual void* raw_handle() = 0;
    virtual const void* raw_handle() const = 0;
    
    //! Get data size in bytes
    virtual size_t size() const = 0;
    
    //! Acquire data for CPU access
    virtual void* acquire(AccessMode mode) = 0;
    
    //! Release data after CPU access
    virtual void release() = 0;
    
    //! Invalidate cached data
    virtual void invalidate() = 0;
    
    //! Hint that data won't be used soon
    virtual void wont_use() = 0;
    
    //! Unregister handle
    virtual void unregister() = 0;
    
    //! MPI-related (optional, may be no-op for some backends)
    virtual int mpi_get_rank() const { return 0; }
    virtual void mpi_transfer(int dst_rank, int src_rank) {}
    virtual void mpi_flush() {}
};

//! Convenient typed wrapper
template<typename T>
class TypedDataHandle : public DataHandle {
public:
    T* acquire_typed(AccessMode mode) {
        return static_cast<T*>(acquire(mode));
    }
};

} // namespace nntile::runtime
```

#### Task Handle Interface

```cpp
// include/nntile/runtime/task_handle.hh
namespace nntile::runtime {

//! Abstract task handle interface
class TaskHandle {
public:
    virtual ~TaskHandle() = default;
    
    //! Wait for task completion
    virtual void wait() = 0;
    
    //! Check if task is complete (non-blocking)
    virtual bool is_complete() const = 0;
    
    //! Detach task (don't wait on destruction)
    virtual void detach() = 0;
    
    //! Get raw handle (backend-specific)
    virtual void* raw_handle() = 0;
};

//! Owning task handle (RAII)
class TaskHandleOwner {
private:
    std::unique_ptr<TaskHandle> handle_;
    bool detached_ = false;

public:
    explicit TaskHandleOwner(std::unique_ptr<TaskHandle> h) : handle_(std::move(h)) {}
    
    TaskHandleOwner(TaskHandleOwner&&) = default;
    TaskHandleOwner& operator=(TaskHandleOwner&&) = default;
    TaskHandleOwner(const TaskHandleOwner&) = delete;
    TaskHandleOwner& operator=(const TaskHandleOwner&) = delete;
    
    ~TaskHandleOwner() {
        if (handle_ && !detached_) {
            handle_->wait();
        }
    }
    
    void wait() { if (handle_) handle_->wait(); }
    bool is_complete() const { return !handle_ || handle_->is_complete(); }
    void detach() { detached_ = true; }
    TaskHandle* get() { return handle_.get(); }
};

//! Task group for batch operations
class TaskGroup {
private:
    std::vector<TaskHandleOwner> tasks_;

public:
    void add(TaskHandleOwner task) {
        tasks_.push_back(std::move(task));
    }
    
    void wait_all() {
        for (auto& task : tasks_) {
            task.wait();
        }
        tasks_.clear();
    }
    
    size_t size() const { return tasks_.size(); }
};

} // namespace nntile::runtime
```

#### Codelet Interface

```cpp
// include/nntile/runtime/codelet.hh
namespace nntile::runtime {

//! CPU function signature
using CpuFunc = void (*)(void* buffers[], void* cl_args);

//! CUDA function signature
using CudaFunc = void (*)(void* buffers[], void* cl_args);

//! Abstract codelet definition
struct CodeletDef {
    std::string name;
    CpuFunc cpu_func = nullptr;
    CudaFunc cuda_func = nullptr;
    uint32_t (*footprint)(void* cl_args) = nullptr;
    
    bool can_run_on_cpu() const { return cpu_func != nullptr; }
    bool can_run_on_cuda() const { return cuda_func != nullptr; }
};

//! Abstract codelet interface (registered with backend)
class Codelet {
public:
    virtual ~Codelet() = default;
    
    //! Get codelet definition
    virtual const CodeletDef& definition() const = 0;
    
    //! Get raw backend-specific codelet
    virtual void* raw_codelet() = 0;
};

//! Codelet registry
class CodeletRegistry {
public:
    static CodeletRegistry& instance();
    
    //! Register a codelet definition
    void register_codelet(const std::string& name, const CodeletDef& def);
    
    //! Get codelet for current backend
    Codelet& get_codelet(const std::string& name);
    
private:
    std::unordered_map<std::string, CodeletDef> definitions_;
    std::unordered_map<std::string, std::unique_ptr<Codelet>> codelets_;
};

} // namespace nntile::runtime
```

#### Operation Submission Interface

```cpp
// include/nntile/runtime/gelu.hh
namespace nntile::runtime {

//! Execution hints for task placement
struct ExecutionHints {
    int target_node = -1;      // -1 = any
    int target_worker = -1;    // -1 = any
    int priority = 0;
    bool prefetch = true;
};

//! GELU operation - runtime agnostic interface
template<typename T>
class GeluOp {
public:
    //! Submit GELU task
    static TaskHandleOwner submit(
        Index nelems,
        DataHandle& src,
        DataHandle& dst,
        const ExecutionHints& hints = {}
    );
};

//! Convenience global instance
template<typename T>
inline TaskHandleOwner gelu_submit(Index nelems, DataHandle& src, DataHandle& dst,
                                    const ExecutionHints& hints = {}) {
    return GeluOp<T>::submit(nelems, src, dst, hints);
}

} // namespace nntile::runtime
```

### 3.8.4 StarPU Backend Implementation

```cpp
// include/nntile/runtime/starpu/backend.hh
namespace nntile::runtime::starpu {

class StarPUBackend : public Backend {
public:
    void init(int argc, char* argv[]) override;
    void shutdown() override;
    BackendType type() const override { return BackendType::StarPU; }
    int num_workers() const override;
    int num_cpu_workers() const override;
    int num_gpu_workers() const override;
    void wait_for_all() override;
    void pause() override;
    void resume() override;
    std::unique_ptr<DataHandle> create_data_handle(size_t size) override;
    std::unique_ptr<DataHandle> create_data_handle(void* ptr, size_t size) override;
};

class StarPUDataHandle : public DataHandle {
private:
    starpu_data_handle_t handle_;
    
public:
    explicit StarPUDataHandle(starpu_data_handle_t h) : handle_(h) {}
    
    void* raw_handle() override { return handle_; }
    const void* raw_handle() const override { return handle_; }
    
    size_t size() const override {
        return starpu_variable_get_elemsize(handle_);
    }
    
    void* acquire(AccessMode mode) override {
        starpu_data_access_mode smode;
        switch (mode) {
            case AccessMode::Read: smode = STARPU_R; break;
            case AccessMode::Write: smode = STARPU_W; break;
            case AccessMode::ReadWrite: smode = STARPU_RW; break;
            default: smode = STARPU_RW;
        }
        starpu_data_acquire(handle_, smode);
        return starpu_variable_get_local_ptr(handle_);
    }
    
    void release() override {
        starpu_data_release(handle_);
    }
    
    // ... other methods
};

class StarPUTaskHandle : public TaskHandle {
private:
    struct starpu_task* task_;
    
public:
    explicit StarPUTaskHandle(struct starpu_task* t) : task_(t) {}
    
    void wait() override {
        if (task_) {
            starpu_task_wait(task_);
            task_ = nullptr;
        }
    }
    
    bool is_complete() const override {
        return task_ == nullptr || starpu_task_finished(task_);
    }
    
    void detach() override {
        task_ = nullptr;
    }
    
    void* raw_handle() override { return task_; }
};

} // namespace nntile::runtime::starpu
```

```cpp
// src/runtime/starpu/gelu.cc
namespace nntile::runtime {

template<typename T>
TaskHandleOwner GeluOp<T>::submit(
    Index nelems,
    DataHandle& src,
    DataHandle& dst,
    const ExecutionHints& hints
) {
    // Get StarPU-specific handles
    auto src_handle = static_cast<starpu_data_handle_t>(src.raw_handle());
    auto dst_handle = static_cast<starpu_data_handle_t>(dst.raw_handle());
    
    // Get codelet
    auto& codelet = CodeletRegistry::instance().get_codelet("gelu_" + type_name<T>());
    auto* starpu_cl = static_cast<struct starpu_codelet*>(codelet.raw_codelet());
    
    // Allocate arguments
    struct args_t { Index nelems; };
    auto* args = new args_t{nelems};
    
    // Create task
    struct starpu_task* task = starpu_task_create();
    task->cl = starpu_cl;
    task->handles[0] = src_handle;
    task->handles[1] = dst_handle;
    task->cl_arg = args;
    task->cl_arg_size = sizeof(*args);
    task->cl_arg_free = 1;
    task->detach = 0;
    task->destroy = 1;
    
    // Apply hints
    if (hints.target_worker >= 0) {
        task->execute_on_a_specific_worker = 1;
        task->workerid = hints.target_worker;
    }
    task->priority = hints.priority;
    
    // Submit
    int ret = starpu_task_submit(task);
    if (ret != 0) {
        throw std::runtime_error("Failed to submit gelu task");
    }
    
    return TaskHandleOwner(std::make_unique<starpu::StarPUTaskHandle>(task));
}

// Explicit instantiations
template class GeluOp<fp32_t>;
template class GeluOp<fp64_t>;
// ...

} // namespace nntile::runtime
```

### 3.8.5 Future TaskFlow Backend (Sketch)

```cpp
// include/nntile/runtime/taskflow/backend.hh
namespace nntile::runtime::taskflow {

class TaskFlowBackend : public Backend {
private:
    tf::Executor executor_;
    tf::Taskflow taskflow_;
    
public:
    void init(int argc, char* argv[]) override {
        // Initialize with hardware concurrency
    }
    
    void shutdown() override {
        executor_.wait_for_all();
    }
    
    // ... implement other methods
};

class TaskFlowTaskHandle : public TaskHandle {
private:
    tf::Future<void> future_;
    
public:
    void wait() override {
        future_.wait();
    }
    
    bool is_complete() const override {
        return future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
    
    // ...
};

} // namespace nntile::runtime::taskflow
```

### 3.8.6 Compile-Time Backend Selection (Zero Overhead)

For compile-time selection, use type aliases and conditional compilation:

```cpp
// include/nntile/runtime/current_backend.hh
namespace nntile::runtime {

#if defined(NNTILE_RUNTIME_STARPU)
    using CurrentBackend = starpu::StarPUBackend;
    using CurrentDataHandle = starpu::StarPUDataHandle;
    using CurrentTaskHandle = starpu::StarPUTaskHandle;
#elif defined(NNTILE_RUNTIME_TASKFLOW)
    using CurrentBackend = taskflow::TaskFlowBackend;
    using CurrentDataHandle = taskflow::TaskFlowDataHandle;
    using CurrentTaskHandle = taskflow::TaskFlowTaskHandle;
#else
    #error "No runtime backend selected"
#endif

// Direct function calls (no virtual dispatch)
inline CurrentBackend& backend() {
    static CurrentBackend instance;
    return instance;
}

} // namespace nntile::runtime
```

### 3.8.7 Runtime Backend Selection (Optional)

When `NNTILE_RUNTIME_DYNAMIC` is defined:

```cpp
// include/nntile/runtime/dynamic_backend.hh
namespace nntile::runtime {

#ifdef NNTILE_RUNTIME_DYNAMIC

//! Factory for creating backends
std::unique_ptr<Backend> create_backend(BackendType type) {
    switch (type) {
        case BackendType::StarPU:
            return std::make_unique<starpu::StarPUBackend>();
        case BackendType::TaskFlow:
            return std::make_unique<taskflow::TaskFlowBackend>();
        case BackendType::Serial:
            return std::make_unique<serial::SerialBackend>();
        case BackendType::Auto:
            // Prefer StarPU if available, then TaskFlow, then Serial
            #ifdef NNTILE_HAS_STARPU
                return std::make_unique<starpu::StarPUBackend>();
            #elif defined(NNTILE_HAS_TASKFLOW)
                return std::make_unique<taskflow::TaskFlowBackend>();
            #else
                return std::make_unique<serial::SerialBackend>();
            #endif
        default:
            throw std::runtime_error("Unknown backend type");
    }
}

//! Global backend storage
class BackendManager {
private:
    std::unique_ptr<Backend> backend_;
    
public:
    static BackendManager& instance() {
        static BackendManager mgr;
        return mgr;
    }
    
    void set(BackendType type) {
        backend_ = create_backend(type);
    }
    
    void set(std::unique_ptr<Backend> b) {
        backend_ = std::move(b);
    }
    
    Backend& get() {
        if (!backend_) {
            backend_ = create_backend(BackendType::Auto);
        }
        return *backend_;
    }
};

inline Backend& get_backend() {
    return BackendManager::instance().get();
}

inline void set_backend(BackendType type) {
    BackendManager::instance().set(type);
}

#endif // NNTILE_RUNTIME_DYNAMIC

} // namespace nntile::runtime
```

### 3.8.8 Tile-Level Using Runtime Abstraction

```cpp
// include/nntile/tile/gelu.hh
namespace nntile::tile {

template<typename T>
runtime::TaskHandleOwner gelu_async(const Tile<T>& src, const Tile<T>& dst) {
    // Validation
    if (src.nelems != dst.nelems) {
        throw std::runtime_error("Tile size mismatch in gelu");
    }
    
    // Submit via runtime-agnostic interface
    return runtime::gelu_submit<T>(src.nelems, src.handle(), dst.handle());
}

template<typename T>
void gelu(const Tile<T>& src, const Tile<T>& dst) {
    gelu_async<T>(src, dst).wait();
}

} // namespace nntile::tile
```

### 3.8.9 Summary: Choosing Selection Strategy

| Scenario | Recommendation |
|----------|----------------|
| Production deployment | Compile-time (zero overhead) |
| Development/testing | Runtime selection for flexibility |
| Benchmarking runtimes | Runtime selection to compare |
| Library distribution | Build multiple variants or dynamic |

**Default recommendation**: Compile-time selection with StarPU, with optional `NNTILE_RUNTIME_DYNAMIC` for testing.

---

## 4. Detailed Component Design

### 4.1 High-Level Graph API Design

#### 4.1.1 Graph Construction Approaches: Analysis

There are several approaches to building computational graphs. We analyze each with pros, cons, and usage examples.

---

##### Approach A: Implicit Graph (PyTorch Style)

Operations execute immediately, building the graph implicitly through automatic differentiation tape.

**How it works:**
- Tensors carry gradient information
- Operations record themselves onto a tape when executed
- Backward pass replays the tape in reverse

**Example Usage:**
```python
import nntile

# Initialize
nntile.init()

# Create tensors - they exist immediately
x = nntile.tensor([seq_len, batch_size, embed_dim], dtype="fp32")
w = nntile.parameter([embed_dim, hidden_dim], dtype="fp32", requires_grad=True)

# Operations execute immediately, graph recorded implicitly
y = nntile.matmul(x, w)          # Executes NOW, records to tape
z = nntile.gelu(y)               # Executes NOW, records to tape
loss = nntile.sum(z)             # Executes NOW, records to tape

# Backward traverses recorded tape
loss.backward()                   # Replays operations in reverse

# Gradients available
print(w.grad)
```

**Pros:**
| Advantage | Description |
|-----------|-------------|
| Intuitive | Matches imperative programming mental model |
| Dynamic shapes | Easy to handle variable-length sequences |
| Debugging | Can inspect intermediate values at any point |
| Familiarity | PyTorch users already know this pattern |
| Control flow | Python if/for naturally become part of graph |

**Cons:**
| Disadvantage | Description |
|--------------|-------------|
| No ahead-of-time optimization | Can't optimize full graph before execution |
| Distribution difficulty | Hard to analyze graph for FSDP/DDP partitioning |
| Repeated tracing | Must re-record tape for each forward pass |
| Memory overhead | Must keep all intermediate tensors for backward |
| No tiling analysis | Can't determine optimal tiling before execution |

---

##### Approach B: Explicit Graph with Deferred Execution (cuDNN Frontend Style)

Graph is fully constructed first, then compiled/instantiated, then executed.

**How it works:**
- Create symbolic tensor nodes (no data yet)
- Define operations as graph edges
- Compile graph with specific configuration (tiling, distribution)
- Execute compiled graph with actual data

**Example Usage:**
```python
import nntile
from nntile.graph import Graph, TensorSpec, DistributionStrategy

nntile.init()

# Phase 1: Define graph structure (no computation)
graph = Graph("transformer_block")

# Declare symbolic tensors
x = graph.input(TensorSpec([seq_len, batch_size, embed_dim], dtype="fp32"), name="input")
w = graph.parameter(TensorSpec([embed_dim, hidden_dim], dtype="fp32"), name="weight")

# Define operations (symbolic, no execution)
y = graph.matmul(x, w)           # Returns TensorNode, no computation
z = graph.gelu(y)                # Returns TensorNode, no computation
out = graph.output(z, name="output")

# Phase 2: Configure and compile
graph.set_distribution(DistributionStrategy.fsdp(world_size=8))
graph.set_tiling({"weight": [embed_dim_tile, hidden_dim_tile]})

compiled = graph.compile()       # Analyze, optimize, allocate

# Phase 3: Execute with actual data
compiled.set_input("input", input_data)
compiled.set_parameter("weight", weight_data)
compiled.forward()
result = compiled.get_output("output")

# Can re-instantiate with different configuration
compiled_ddp = graph.compile(DistributionStrategy.ddp(world_size=8))
```

**Pros:**
| Advantage | Description |
|-----------|-------------|
| Full graph visibility | Can analyze entire computation before execution |
| Optimization opportunities | Dead code elimination, fusion, scheduling |
| Distribution planning | Can partition tensors optimally for FSDP/DDP |
| Tiling analysis | Determine optimal tile sizes based on full graph |
| Reusable definition | Same graph, multiple instantiations |
| Memory planning | Allocate exact memory needed, plan reuse |

**Cons:**
| Disadvantage | Description |
|--------------|-------------|
| Less intuitive | Two-phase (define, then execute) mental model |
| Dynamic shapes harder | Must handle shape variability explicitly |
| Debugging complexity | Can't easily inspect intermediate values |
| Boilerplate | More code to set up graph structure |

---

##### Approach C: Lazy Tensor / Tracing JIT (PyTorch 2.0 / JAX Style)

Hybrid approach: write eager code, but trace it into a graph for optimization.

**How it works:**
- Operations are recorded but not immediately executed
- Graph is built lazily as operations are called
- Explicit "sync point" triggers compilation and execution
- Can re-trace when shapes change

**Example Usage:**
```python
import nntile
from nntile import lazy

nntile.init()

# Mark tensors as lazy
x = lazy.tensor([seq_len, batch_size, embed_dim], dtype="fp32")
w = lazy.parameter([embed_dim, hidden_dim], dtype="fp32")

# Operations are recorded, not executed
y = lazy.matmul(x, w)            # Recorded
z = lazy.gelu(y)                 # Recorded
loss = lazy.sum(z)               # Recorded

# Explicit sync triggers compilation and execution
loss.sync()                       # Compile + Execute everything

# Or use context manager
with lazy.trace() as trace:
    y = lazy.matmul(x, w)
    z = lazy.gelu(y)

# Compile traced graph with distribution
compiled = trace.compile(distribution=DistributionStrategy.fsdp(8))
compiled.execute()
```

**Pros:**
| Advantage | Description |
|-----------|-------------|
| Familiar syntax | Looks like eager code |
| Optimization possible | Full graph available at sync point |
| Flexibility | Can mix lazy and eager operations |
| Gradual adoption | Easy to convert existing code |

**Cons:**
| Disadvantage | Description |
|--------------|-------------|
| Implicit graph boundaries | Unclear when graph ends |
| Re-tracing overhead | May re-trace on shape changes |
| Hidden complexity | Debugging traced vs eager execution differs |
| Limited control | Less explicit control over compilation |

---

##### Approach D: Functional Transformations (JAX Style)

Pure functions transformed by decorators for autodiff, distribution, etc.

**How it works:**
- Define computation as pure functions
- Apply transformations (grad, vmap, pmap) as decorators
- Transformations compose and are explicit

**Example Usage:**
```python
import nntile
from nntile import functional as F
from nntile.transforms import grad, distribute, jit

nntile.init()

# Define pure function
def forward(params, x):
    y = F.matmul(x, params['w'])
    z = F.gelu(y)
    return F.sum(z)

# Transform for gradients
grad_fn = grad(forward, argnums=0)  # Gradient w.r.t. params

# Transform for distribution
dist_forward = distribute(forward, strategy=DistributionStrategy.fsdp(8))

# Transform for JIT compilation
fast_forward = jit(forward)

# Compose transformations
fast_grad = jit(grad(forward))

# Use
params = {'w': nntile.randn([embed_dim, hidden_dim])}
x = nntile.randn([seq_len, batch_size, embed_dim])

loss = fast_forward(params, x)
grads = fast_grad(params, x)
```

**Pros:**
| Advantage | Description |
|-----------|-------------|
| Composable | Transformations stack cleanly |
| Explicit | Clear what each transformation does |
| Functional purity | Easier to reason about, parallelize |
| Powerful abstractions | vmap for batching, pmap for parallelism |

**Cons:**
| Disadvantage | Description |
|--------------|-------------|
| Different paradigm | Requires functional programming mindset |
| State management | Must explicitly thread state through functions |
| Learning curve | Unfamiliar to PyTorch users |
| Less flexible | Harder to do imperative control flow |

---

##### Approach E: Multi-Stage Graph (Recommended for NNTile)

Combines explicit graph with multiple instantiation stages.

**How it works:**
1. **Logical Graph**: Define operations and tensor relationships
2. **Physical Graph**: Apply tiling and distribution decisions
3. **Executable Graph**: Bind to actual memory and runtime

This separation allows the same logical graph to have multiple physical realizations.

**Example Usage:**
```python
import nntile
from nntile.graph import LogicalGraph, PhysicalGraph, ExecutableGraph
from nntile.graph import TensorSpec, OpSpec, DistributionStrategy, TilingStrategy

nntile.init(mpi=True)

# ═══════════════════════════════════════════════════════════════
# Stage 1: Logical Graph (what to compute)
# ═══════════════════════════════════════════════════════════════
logical = LogicalGraph("transformer_layer")

# Declare logical tensors (shapes only, no tiling/distribution)
x = logical.input(
    TensorSpec(shape=[seq_len, batch_size, embed_dim], dtype="fp32"),
    name="input"
)
wq = logical.parameter(
    TensorSpec(shape=[embed_dim, embed_dim], dtype="fp32"),
    name="W_query"
)
wk = logical.parameter(
    TensorSpec(shape=[embed_dim, embed_dim], dtype="fp32"),
    name="W_key"
)
wv = logical.parameter(
    TensorSpec(shape=[embed_dim, embed_dim], dtype="fp32"),
    name="W_value"
)
wo = logical.parameter(
    TensorSpec(shape=[embed_dim, embed_dim], dtype="fp32"),
    name="W_out"
)

# Define operations (logical, no tiling decisions)
q = logical.matmul(x, wq, name="query_proj")
k = logical.matmul(x, wk, name="key_proj")
v = logical.matmul(x, wv, name="value_proj")
attn = logical.scaled_dot_product_attention(q, k, v, name="attention")
out = logical.matmul(attn, wo, name="output_proj")
out = logical.add(out, x, name="residual")  # Residual connection

output = logical.output(out, name="output")

# Graph analysis available
print(f"Operations: {logical.num_operations()}")
print(f"Parameters: {logical.num_parameters()}")
print(f"FLOPs: {logical.estimate_flops()}")

# ═══════════════════════════════════════════════════════════════
# Stage 2: Physical Graph (how to tile and distribute)
# ═══════════════════════════════════════════════════════════════

# Option A: FSDP - shard parameters across all GPUs
physical_fsdp = PhysicalGraph(logical)
physical_fsdp.set_distribution(DistributionStrategy.fsdp(
    world_size=8,
    shard_degree=8,
    shard_optimizer_states=True
))
physical_fsdp.set_tiling(TilingStrategy.auto(
    target_tile_size_mb=64,
    constraints={
        "input": {"batch_dim": "shard"},      # Shard batch across GPUs
        "W_query": {"both_dims": "shard"},    # FSDP shards weights
    }
))
physical_fsdp.compile()  # Compute tile shapes, distribution map

# Option B: DDP - replicate parameters, shard data
physical_ddp = PhysicalGraph(logical)
physical_ddp.set_distribution(DistributionStrategy.ddp(world_size=8))
physical_ddp.set_tiling(TilingStrategy.auto(
    target_tile_size_mb=64,
    constraints={
        "input": {"batch_dim": "shard"},      # Shard batch
        "W_query": {"both_dims": "replicate"} # Replicate weights
    }
))
physical_ddp.compile()

# Option C: Tensor Parallel - shard hidden dimension
physical_tp = PhysicalGraph(logical)
physical_tp.set_distribution(DistributionStrategy.tensor_parallel(
    world_size=8,
    parallel_dim=1  # Shard output dimension of projections
))
physical_tp.set_tiling(TilingStrategy.manual({
    "input": [seq_tile, batch_tile, embed_tile],
    "W_query": [embed_tile, embed_tile // 8],  # Column parallel
    "W_key": [embed_tile, embed_tile // 8],
    "W_value": [embed_tile, embed_tile // 8],
    "W_out": [embed_tile // 8, embed_tile],    # Row parallel
}))
physical_tp.compile()

# Compare memory usage
print(f"FSDP memory per GPU: {physical_fsdp.memory_per_device() / 1e9:.2f} GB")
print(f"DDP memory per GPU: {physical_ddp.memory_per_device() / 1e9:.2f} GB")
print(f"TP memory per GPU: {physical_tp.memory_per_device() / 1e9:.2f} GB")

# ═══════════════════════════════════════════════════════════════
# Stage 3: Executable Graph (bind to runtime and execute)
# ═══════════════════════════════════════════════════════════════

# Create executable from physical graph
exe = ExecutableGraph(physical_fsdp)

# Load parameters (distributed according to physical graph)
exe.load_parameters("checkpoint.pt")

# Or initialize randomly
exe.init_parameters("xavier_uniform")

# Bind input data
exe.bind_input("input", input_tensor)

# Execute forward pass
exe.forward()

# Get output
output = exe.get_output("output")

# Execute backward pass (if gradients needed)
exe.bind_output_grad("output", output_grad)
exe.backward()

# Access gradients
w_query_grad = exe.get_parameter_grad("W_query")

# ═══════════════════════════════════════════════════════════════
# Advanced: Dynamic shape handling
# ═══════════════════════════════════════════════════════════════

# For variable sequence lengths, use symbolic dimensions
logical_dynamic = LogicalGraph("transformer_dynamic")
x_dyn = logical_dynamic.input(
    TensorSpec(
        shape=["seq_len", batch_size, embed_dim],  # "seq_len" is symbolic
        dtype="fp32"
    ),
    name="input"
)
# ... define operations ...

# Specialize for specific shapes
physical_512 = PhysicalGraph(logical_dynamic, shape_bindings={"seq_len": 512})
physical_1024 = PhysicalGraph(logical_dynamic, shape_bindings={"seq_len": 1024})
```

**Pros:**
| Advantage | Description |
|-----------|-------------|
| Clear separation | Logical vs physical vs executable clearly delineated |
| Multiple instantiations | Same logical graph, different physical realizations |
| Distribution flexibility | Try FSDP, DDP, TP with same graph definition |
| Memory planning | Know exact memory per device before execution |
| Optimization opportunities | Full graph visible at each stage |
| Reusability | Logical graph is a reusable template |

**Cons:**
| Disadvantage | Description |
|--------------|-------------|
| More complex API | Three stages instead of one |
| Boilerplate | More setup code required |
| Learning curve | New concepts (logical vs physical) to learn |

---

#### 4.1.2 Recommendation: Multi-Stage Explicit Graph (Approach E)

**Rationale:**

1. **Distribution Experimentation**: The primary goal is to try different instantiations (FSDP, DDP, TP). Multi-stage approach makes this natural - same logical graph, different physical graphs.

2. **Tiling Decisions**: NNTile's core value is efficient tiling. Separating logical shape from physical tiling allows:
   - Automatic tiling based on hardware
   - Manual override for specific tensors
   - Comparison of tiling strategies

3. **Memory Planning**: With explicit physical graph, we can:
   - Compute exact memory per device before execution
   - Reject infeasible configurations early
   - Plan tensor reuse and offloading

4. **StarPU Integration**: StarPU expects a task graph. Explicit graph maps naturally to StarPU's model.

5. **Backward Compatibility**: Existing tensor/tile code can be used inside `ExecutableGraph` - no rewrite needed.

---

#### 4.1.3 Detailed Multi-Stage Graph Design

#### 4.1.4 Core Data Structures

##### TensorSpec - Logical Tensor Specification

```cpp
namespace nntile::graph {

//! Dimension can be concrete or symbolic
using Dimension = std::variant<Index, std::string>;

//! Data types supported
enum class DataType {
    FP16, BF16, FP32, FP64,
    FP32_FAST_TF32, FP32_FAST_FP16, FP32_FAST_BF16,
    INT32, INT64, BOOL
};

//! Tensor role in computation
enum class TensorRole {
    Input,       // Fed from outside
    Output,      // Produced for outside consumption
    Parameter,   // Trainable weight
    Buffer,      // Intermediate activation
    Constant     // Fixed value
};

//! Logical tensor specification (no tiling/distribution info)
struct TensorSpec {
    std::vector<Dimension> shape;  // Can mix concrete and symbolic dims
    DataType dtype;
    TensorRole role;
    std::string name;
    bool requires_grad = false;
    
    // Constructors
    TensorSpec(std::vector<Index> shape, DataType dtype, std::string name = "");
    TensorSpec(std::vector<Dimension> shape, DataType dtype, std::string name = "");
    
    // Queries
    Index ndim() const { return shape.size(); }
    bool has_symbolic_dims() const;
    std::vector<std::string> symbolic_dim_names() const;
    
    // Create concrete spec by binding symbolic dims
    TensorSpec bind(const std::map<std::string, Index>& bindings) const;
    
    // Compute number of elements (only if fully concrete)
    std::optional<Index> nelems() const;
};

} // namespace nntile::graph
```

##### TensorNode - Node in Logical Graph

```cpp
namespace nntile::graph {

//! Unique identifier for nodes
using NodeId = uint64_t;

//! Forward declarations
class OpNode;
class LogicalGraph;

//! A tensor node in the logical graph
class TensorNode {
    friend class LogicalGraph;
    friend class OpNode;
    
private:
    NodeId id_;
    TensorSpec spec_;
    LogicalGraph* graph_;
    
    // Edges
    OpNode* producer_ = nullptr;           // Op that creates this tensor
    std::vector<OpNode*> consumers_;       // Ops that use this tensor
    
public:
    TensorNode(NodeId id, TensorSpec spec, LogicalGraph* graph);
    
    // Accessors
    NodeId id() const { return id_; }
    const TensorSpec& spec() const { return spec_; }
    const std::string& name() const { return spec_.name; }
    DataType dtype() const { return spec_.dtype; }
    const std::vector<Dimension>& shape() const { return spec_.shape; }
    
    // Graph queries
    bool is_input() const { return spec_.role == TensorRole::Input; }
    bool is_output() const { return spec_.role == TensorRole::Output; }
    bool is_parameter() const { return spec_.role == TensorRole::Parameter; }
    OpNode* producer() const { return producer_; }
    const std::vector<OpNode*>& consumers() const { return consumers_; }
    
    // Fluent API for building graph (returns new TensorNode)
    TensorNode& matmul(TensorNode& other, bool trans_a = false, bool trans_b = false);
    TensorNode& add(TensorNode& other);
    TensorNode& gelu();
    TensorNode& relu();
    TensorNode& softmax(int axis = -1);
    TensorNode& layer_norm(TensorNode& gamma, TensorNode& beta, float eps = 1e-5f);
    // ... more operations
};

} // namespace nntile::graph
```

##### OpNode - Operation Node in Logical Graph

```cpp
namespace nntile::graph {

//! Operation types
enum class OpType {
    // Elementwise
    Add, Sub, Mul, Div, Gelu, Relu, Silu, Tanh, Sigmoid, Sqrt, Pow,
    // Reductions
    Sum, Mean, Max, Min, Softmax, LogSumExp,
    // Linear algebra
    MatMul, BatchedMatMul,
    // Normalization
    LayerNorm, RMSNorm, BatchNorm,
    // Attention
    ScaledDotProductAttention,
    // Data movement
    Transpose, Reshape, Slice, Concat, Gather, Scatter,
    // Special
    Embedding, EmbeddingBackward,
    // Optimizer operations
    AdamStep, SGDStep
};

//! Operation attributes (varies by op type)
struct OpAttributes {
    // MatMul
    bool trans_a = false;
    bool trans_b = false;
    
    // Reduction/Normalization
    int axis = -1;
    float epsilon = 1e-5f;
    
    // Attention
    bool causal_mask = false;
    float dropout = 0.0f;
    
    // Generic
    std::map<std::string, std::variant<int, float, bool, std::string>> extra;
};

//! An operation node in the logical graph
class OpNode {
    friend class LogicalGraph;
    
private:
    NodeId id_;
    OpType type_;
    OpAttributes attrs_;
    LogicalGraph* graph_;
    
    // Edges
    std::vector<TensorNode*> inputs_;
    std::vector<TensorNode*> outputs_;
    
public:
    OpNode(NodeId id, OpType type, OpAttributes attrs, LogicalGraph* graph);
    
    // Accessors
    NodeId id() const { return id_; }
    OpType type() const { return type_; }
    const OpAttributes& attributes() const { return attrs_; }
    
    // Graph structure
    const std::vector<TensorNode*>& inputs() const { return inputs_; }
    const std::vector<TensorNode*>& outputs() const { return outputs_; }
    
    // Analysis
    std::string name() const;  // Human-readable name
    Index estimate_flops() const;  // FLOPs for this operation
    
    // Gradient information
    bool has_backward() const;
    std::vector<OpNode*> backward_ops() const;  // Ops for gradient computation
};

} // namespace nntile::graph
```

##### LogicalGraph - Stage 1: What to Compute

```cpp
namespace nntile::graph {

//! Logical graph - defines computation without physical decisions
class LogicalGraph {
private:
    std::string name_;
    std::vector<std::unique_ptr<TensorNode>> tensor_nodes_;
    std::vector<std::unique_ptr<OpNode>> op_nodes_;
    
    // Special node lists
    std::vector<TensorNode*> inputs_;
    std::vector<TensorNode*> outputs_;
    std::vector<TensorNode*> parameters_;
    
    // Node ID counter
    NodeId next_id_ = 0;
    
    // Helper to create nodes
    TensorNode& create_tensor(TensorSpec spec);
    OpNode& create_op(OpType type, OpAttributes attrs,
                      std::vector<TensorNode*> inputs,
                      std::vector<TensorSpec> output_specs);

public:
    explicit LogicalGraph(const std::string& name = "");
    
    // ═══════════════════════════════════════════════════════════
    // Tensor Creation
    // ═══════════════════════════════════════════════════════════
    
    //! Declare an input tensor
    TensorNode& input(const TensorSpec& spec, const std::string& name = "");
    
    //! Declare a trainable parameter
    TensorNode& parameter(const TensorSpec& spec, const std::string& name = "");
    
    //! Declare a constant tensor
    TensorNode& constant(const TensorSpec& spec, const std::string& name = "");
    
    //! Mark a tensor as output
    TensorNode& output(TensorNode& tensor, const std::string& name = "");
    
    // ═══════════════════════════════════════════════════════════
    // Operations
    // ═══════════════════════════════════════════════════════════
    
    // Elementwise
    TensorNode& add(TensorNode& a, TensorNode& b, const std::string& name = "");
    TensorNode& mul(TensorNode& a, TensorNode& b, const std::string& name = "");
    TensorNode& gelu(TensorNode& x, const std::string& name = "");
    TensorNode& relu(TensorNode& x, const std::string& name = "");
    TensorNode& silu(TensorNode& x, const std::string& name = "");
    
    // Linear algebra
    TensorNode& matmul(TensorNode& a, TensorNode& b,
                       bool trans_a = false, bool trans_b = false,
                       const std::string& name = "");
    
    // Normalization
    TensorNode& layer_norm(TensorNode& x, TensorNode& gamma, TensorNode& beta,
                           float eps = 1e-5f, const std::string& name = "");
    TensorNode& rms_norm(TensorNode& x, TensorNode& gamma,
                         float eps = 1e-5f, const std::string& name = "");
    
    // Attention
    TensorNode& scaled_dot_product_attention(
        TensorNode& q, TensorNode& k, TensorNode& v,
        bool causal = false, float dropout = 0.0f,
        const std::string& name = "");
    
    // Data movement
    TensorNode& transpose(TensorNode& x, std::vector<int> perm,
                          const std::string& name = "");
    TensorNode& reshape(TensorNode& x, std::vector<Dimension> new_shape,
                        const std::string& name = "");
    
    // Embedding
    TensorNode& embedding(TensorNode& indices, TensorNode& table,
                          const std::string& name = "");
    
    // ═══════════════════════════════════════════════════════════
    // Analysis
    // ═══════════════════════════════════════════════════════════
    
    //! Get topological order of operations
    std::vector<OpNode*> topological_order() const;
    
    //! Get all tensor nodes
    const std::vector<std::unique_ptr<TensorNode>>& tensors() const;
    
    //! Get all operation nodes
    const std::vector<std::unique_ptr<OpNode>>& operations() const;
    
    //! Query methods
    size_t num_operations() const { return op_nodes_.size(); }
    size_t num_tensors() const { return tensor_nodes_.size(); }
    size_t num_parameters() const { return parameters_.size(); }
    const std::vector<TensorNode*>& inputs() const { return inputs_; }
    const std::vector<TensorNode*>& outputs() const { return outputs_; }
    const std::vector<TensorNode*>& parameters() const { return parameters_; }
    
    //! Estimate total FLOPs (forward pass)
    Index estimate_flops() const;
    
    //! Estimate memory for activations (in bytes, single precision)
    Index estimate_activation_memory() const;
    
    //! Estimate memory for parameters (in bytes)
    Index estimate_parameter_memory() const;
    
    //! Check if graph has symbolic dimensions
    bool has_symbolic_dims() const;
    
    //! Get all symbolic dimension names
    std::set<std::string> symbolic_dim_names() const;
    
    // ═══════════════════════════════════════════════════════════
    // Serialization
    // ═══════════════════════════════════════════════════════════
    
    //! Save graph to file (JSON or binary)
    void save(const std::string& path) const;
    
    //! Load graph from file
    static LogicalGraph load(const std::string& path);
    
    //! Export to visualization format (DOT, etc.)
    std::string to_dot() const;
};

} // namespace nntile::graph
```

##### PhysicalGraph - Stage 2: How to Tile and Distribute

```cpp
namespace nntile::graph {

//! Tiling specification for a tensor
struct TilingSpec {
    std::vector<Index> tile_shape;
    
    // Computed from shape and tile_shape
    std::vector<Index> grid_shape;
    Index num_tiles;
    
    static TilingSpec from_tile_shape(const std::vector<Index>& shape,
                                       const std::vector<Index>& tile_shape);
};

//! Distribution specification for a tensor
struct DistributionSpec {
    enum class Pattern {
        Replicated,     // Same data on all devices
        Sharded,        // Split across devices
        Partial         // Partial results on different devices
    };
    
    Pattern pattern;
    int shard_dim = -1;              // Which dimension to shard (-1 = none)
    std::vector<int> device_mapping;  // tile_idx -> device_id
    
    static DistributionSpec replicated(int num_devices);
    static DistributionSpec sharded(int dim, int num_devices);
};

//! Physical tensor - logical tensor + tiling + distribution
struct PhysicalTensor {
    TensorNode* logical;
    TilingSpec tiling;
    DistributionSpec distribution;
    
    // Computed properties
    Index memory_per_device() const;
    std::vector<Index> tiles_on_device(int device_id) const;
};

//! Physical operation - logical op + execution decisions
struct PhysicalOp {
    OpNode* logical;
    
    // Execution hints per tile
    struct TileExecution {
        Index tile_idx;
        int device_id;
        int worker_id = -1;  // -1 = any worker on device
        int priority = 0;
    };
    std::vector<TileExecution> tile_executions;
};

//! Physical graph - logical graph + physical decisions
class PhysicalGraph {
private:
    LogicalGraph* logical_;
    std::map<std::string, Index> shape_bindings_;  // Symbolic dim bindings
    
    // Physical decisions
    std::map<NodeId, PhysicalTensor> tensor_physical_;
    std::map<NodeId, PhysicalOp> op_physical_;
    
    // Strategy
    DistributionStrategy dist_strategy_;
    TilingStrategy tiling_strategy_;
    
    // Computed
    bool compiled_ = false;
    Index total_memory_ = 0;
    std::map<int, Index> memory_per_device_;

public:
    //! Create physical graph from logical graph
    PhysicalGraph(LogicalGraph& logical,
                  const std::map<std::string, Index>& shape_bindings = {});
    
    // ═══════════════════════════════════════════════════════════
    // Configuration
    // ═══════════════════════════════════════════════════════════
    
    //! Set distribution strategy
    void set_distribution(const DistributionStrategy& strategy);
    
    //! Set tiling strategy
    void set_tiling(const TilingStrategy& strategy);
    
    //! Override tiling for specific tensor
    void set_tensor_tiling(const std::string& name, const TilingSpec& spec);
    
    //! Override distribution for specific tensor
    void set_tensor_distribution(const std::string& name, const DistributionSpec& spec);
    
    // ═══════════════════════════════════════════════════════════
    // Compilation
    // ═══════════════════════════════════════════════════════════
    
    //! Compile physical decisions (tiling, distribution, scheduling)
    void compile();
    
    //! Check if compiled
    bool is_compiled() const { return compiled_; }
    
    // ═══════════════════════════════════════════════════════════
    // Analysis (available after compile)
    // ═══════════════════════════════════════════════════════════
    
    //! Memory per device (bytes)
    Index memory_per_device(int device_id) const;
    
    //! Maximum memory across all devices
    Index max_memory_per_device() const;
    
    //! Total memory (sum across devices)
    Index total_memory() const;
    
    //! Get physical tensor info
    const PhysicalTensor& physical_tensor(const std::string& name) const;
    
    //! Get physical op info
    const PhysicalOp& physical_op(const OpNode& op) const;
    
    //! Communication volume estimate (bytes)
    Index estimate_communication() const;
    
    //! Export tiling/distribution decisions for debugging
    std::string dump_physical_plan() const;
};

} // namespace nntile::graph
```

##### ExecutableGraph - Stage 3: Bind to Runtime and Execute

```cpp
namespace nntile::graph {

//! Executable graph - physical graph bound to actual memory and runtime
class ExecutableGraph {
private:
    PhysicalGraph* physical_;
    
    // Actual tensor storage (maps to runtime::DataHandle)
    std::map<NodeId, std::unique_ptr<runtime::DataHandle>> data_handles_;
    
    // Execution state
    enum class State { Uninitialized, Ready, Executing, Completed };
    State state_ = State::Uninitialized;
    
    // Task handles for async execution
    runtime::TaskGroup forward_tasks_;
    runtime::TaskGroup backward_tasks_;
    
    // Gradient storage (if backward needed)
    std::map<NodeId, std::unique_ptr<runtime::DataHandle>> grad_handles_;
    bool gradients_enabled_ = false;

public:
    //! Create executable from physical graph
    explicit ExecutableGraph(PhysicalGraph& physical);
    
    ~ExecutableGraph();
    
    // ═══════════════════════════════════════════════════════════
    // Initialization
    // ═══════════════════════════════════════════════════════════
    
    //! Initialize parameters with specific initializer
    void init_parameters(const std::string& initializer);  // "zeros", "xavier", etc.
    
    //! Load parameters from checkpoint file
    void load_parameters(const std::string& path);
    
    //! Save parameters to checkpoint file
    void save_parameters(const std::string& path);
    
    //! Enable gradient computation
    void enable_gradients(bool enable = true);
    
    // ═══════════════════════════════════════════════════════════
    // Input/Output Binding
    // ═══════════════════════════════════════════════════════════
    
    //! Bind input data (copies data into internal storage)
    void bind_input(const std::string& name, const void* data, size_t size);
    
    //! Bind input from numpy array (Python interface)
    void bind_input_numpy(const std::string& name, /* numpy array */);
    
    //! Bind input from existing tensor (zero-copy if possible)
    void bind_input_tensor(const std::string& name, const tensor::Tensor<auto>& t);
    
    //! Get output data (copies data out)
    void get_output(const std::string& name, void* data, size_t size);
    
    //! Get output as new tensor
    template<typename T>
    tensor::Tensor<T> get_output_tensor(const std::string& name);
    
    //! Bind output gradient (for backward pass)
    void bind_output_grad(const std::string& name, const void* data, size_t size);
    
    //! Get parameter gradient
    void get_parameter_grad(const std::string& name, void* data, size_t size);
    
    // ═══════════════════════════════════════════════════════════
    // Execution
    // ═══════════════════════════════════════════════════════════
    
    //! Execute forward pass (async)
    void forward_async();
    
    //! Execute forward pass (blocking)
    void forward();
    
    //! Execute backward pass (async)
    void backward_async();
    
    //! Execute backward pass (blocking)
    void backward();
    
    //! Wait for all operations to complete
    void synchronize();
    
    //! Clear intermediate activations (free memory)
    void clear_activations();
    
    //! Clear gradients
    void clear_gradients();
    
    // ═══════════════════════════════════════════════════════════
    // Advanced
    // ═══════════════════════════════════════════════════════════
    
    //! Access underlying data handle (for advanced use)
    runtime::DataHandle& data_handle(const std::string& tensor_name);
    
    //! Get execution statistics
    struct ExecutionStats {
        double forward_time_ms;
        double backward_time_ms;
        Index bytes_transferred;
        Index flops_executed;
    };
    ExecutionStats get_stats() const;
};

} // namespace nntile::graph
```

#### 4.1.5 DistributionStrategy Class

```cpp
namespace nntile::graph {

enum class ParallelismMode {
    DDP,                 // Data Distributed Parallel - replicate model, shard data
    FSDP,                // Fully Sharded Data Parallel - shard everything
    TENSOR_PARALLEL,     // Shard specific tensor dimensions
    PIPELINE_PARALLEL,   // Shard layers across nodes
    HYBRID               // Combination of above
};

struct DistributionStrategy {
    ParallelismMode mode;
    int world_size = 1;
    
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
    std::vector<int> stage_to_device;  // stage_idx -> device_id
    
    // Factory methods
    static DistributionStrategy ddp(int world_size) {
        DistributionStrategy s;
        s.mode = ParallelismMode::DDP;
        s.world_size = world_size;
        s.data_parallel_size = world_size;
        return s;
    }
    
    static DistributionStrategy fsdp(int world_size, int shard_degree = -1) {
        DistributionStrategy s;
        s.mode = ParallelismMode::FSDP;
        s.world_size = world_size;
        s.shard_degree = (shard_degree < 0) ? world_size : shard_degree;
        return s;
    }
    
    static DistributionStrategy tensor_parallel(int world_size, std::vector<int> dims) {
        DistributionStrategy s;
        s.mode = ParallelismMode::TENSOR_PARALLEL;
        s.world_size = world_size;
        s.tensor_parallel_size = world_size;
        s.tensor_parallel_dims = dims;
        return s;
    }
    
    static DistributionStrategy pipeline(int world_size, int num_stages) {
        DistributionStrategy s;
        s.mode = ParallelismMode::PIPELINE_PARALLEL;
        s.world_size = world_size;
        s.pipeline_stages = num_stages;
        return s;
    }
    
    static DistributionStrategy hybrid(
        int data_parallel_size,
        int tensor_parallel_size,
        int pipeline_stages = 1
    ) {
        DistributionStrategy s;
        s.mode = ParallelismMode::HYBRID;
        s.world_size = data_parallel_size * tensor_parallel_size * pipeline_stages;
        s.data_parallel_size = data_parallel_size;
        s.tensor_parallel_size = tensor_parallel_size;
        s.pipeline_stages = pipeline_stages;
        return s;
    }
};

} // namespace nntile::graph
```

#### 4.1.6 TilingStrategy Class

```cpp
namespace nntile::graph {

//! Tiling constraint for a specific dimension
struct DimTilingConstraint {
    enum class Type {
        Auto,           // Let the system decide
        Fixed,          // Use exact tile size
        Multiple,       // Tile size must be multiple of this
        Divisor,        // Tile size must divide this evenly
        Full,           // No tiling (tile = full dimension)
        Shard           // Tile across devices (for distribution)
    };
    
    Type type = Type::Auto;
    Index value = 0;  // Meaning depends on type
};

//! Tiling constraints for a tensor
struct TensorTilingConstraints {
    std::string tensor_name;
    std::map<int, DimTilingConstraint> dim_constraints;  // dim_idx -> constraint
    
    // Convenience setters
    TensorTilingConstraints& dim(int idx, DimTilingConstraint::Type type, Index value = 0) {
        dim_constraints[idx] = {type, value};
        return *this;
    }
    
    TensorTilingConstraints& full_dim(int idx) {
        return dim(idx, DimTilingConstraint::Type::Full);
    }
    
    TensorTilingConstraints& shard_dim(int idx) {
        return dim(idx, DimTilingConstraint::Type::Shard);
    }
    
    TensorTilingConstraints& fixed_dim(int idx, Index tile_size) {
        return dim(idx, DimTilingConstraint::Type::Fixed, tile_size);
    }
};

//! Overall tiling strategy
class TilingStrategy {
private:
    enum class Mode { Auto, Manual, Constrained };
    Mode mode_;
    
    // Auto mode settings
    Index target_tile_bytes_ = 64 * 1024 * 1024;  // 64 MB default
    Index min_tile_size_ = 64;
    Index max_tile_size_ = 8192;
    
    // Manual mode settings
    std::map<std::string, std::vector<Index>> manual_tilings_;
    
    // Constrained mode settings
    std::vector<TensorTilingConstraints> constraints_;

public:
    //! Auto tiling - system determines tile sizes
    static TilingStrategy auto_tiling(
        Index target_tile_bytes = 64 * 1024 * 1024,
        Index min_tile_size = 64,
        Index max_tile_size = 8192
    ) {
        TilingStrategy s;
        s.mode_ = Mode::Auto;
        s.target_tile_bytes_ = target_tile_bytes;
        s.min_tile_size_ = min_tile_size;
        s.max_tile_size_ = max_tile_size;
        return s;
    }
    
    //! Manual tiling - user specifies exact tile sizes
    static TilingStrategy manual(
        const std::map<std::string, std::vector<Index>>& tilings
    ) {
        TilingStrategy s;
        s.mode_ = Mode::Manual;
        s.manual_tilings_ = tilings;
        return s;
    }
    
    //! Constrained tiling - auto with constraints
    static TilingStrategy constrained(
        const std::vector<TensorTilingConstraints>& constraints,
        Index target_tile_bytes = 64 * 1024 * 1024
    ) {
        TilingStrategy s;
        s.mode_ = Mode::Constrained;
        s.constraints_ = constraints;
        s.target_tile_bytes_ = target_tile_bytes;
        return s;
    }
    
    //! Compute tile shapes for all tensors in graph
    std::map<std::string, std::vector<Index>> compute_tilings(
        const LogicalGraph& graph,
        const DistributionStrategy& dist
    ) const;
};

} // namespace nntile::graph
```

#### 4.1.7 ExecutionPolicy Class

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

### 4.2 Graph API Comparison Summary

| Aspect | Implicit (PyTorch) | Explicit (cuDNN) | Lazy (JAX) | Functional | **Multi-Stage (Recommended)** |
|--------|-------------------|------------------|------------|------------|-------------------------------|
| Ease of use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Optimization | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Distribution | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Tiling control | ⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Dynamic shapes | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Memory planning | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Debugging | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Reusability | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Recommendation**: Multi-Stage Explicit Graph (Approach E) provides the best balance for NNTile's goals of experimenting with different distribution strategies while maintaining full control over tiling decisions.

---

### 4.3 Enhanced Runtime Level (formerly StarPU Level)

#### 4.3.1 Extended Submit Interface

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

#### 4.3.2 Implementation Changes

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

### 4.4 Enhanced Tensor Level

#### 4.4.1 Execution Context

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

#### 4.4.2 Enhanced Tensor Operations

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

### Phase 7: Runtime Abstraction Layer (Parallel Track F)

**Goal**: Create runtime-agnostic interface supporting multiple backends (StarPU, TaskFlow, etc.)

**Tasks**:
1. [ ] Create `include/nntile/runtime/` directory structure
2. [ ] Implement abstract `Backend` interface class
3. [ ] Implement abstract `DataHandle` interface class
4. [ ] Implement abstract `TaskHandle` interface class with RAII wrapper (`TaskHandleOwner`)
5. [ ] Implement `TaskGroup` for batch task management
6. [ ] Implement `CodeletDef` and `Codelet` abstract interface
7. [ ] Implement `CodeletRegistry` for codelet management
8. [ ] Implement `ExecutionHints` structure
9. [ ] Create runtime-agnostic operation interfaces (e.g., `GeluOp<T>`)
10. [ ] Move `src/starpu/` to `src/runtime/starpu/`
11. [ ] Implement `StarPUBackend` class
12. [ ] Implement `StarPUDataHandle` class
13. [ ] Implement `StarPUTaskHandle` class
14. [ ] Refactor StarPU operations to use `starpu_task_create()` + `starpu_task_submit()`
15. [ ] Update all StarPU operations to return `TaskHandleOwner`
16. [ ] Implement compile-time backend selection via CMake
17. [ ] (Optional) Implement runtime backend selection with `NNTILE_RUNTIME_DYNAMIC`
18. [ ] Update tile-level to use `runtime::` namespace instead of `starpu::`
19. [ ] Update tile-level async functions to return `TaskHandleOwner`
20. [ ] Update blocking tile functions to use `task.wait()` instead of `wait_for_all()`
21. [ ] Update tensor-level to use runtime abstraction
22. [ ] Add serial backend for debugging/testing
23. [ ] (Future) Add TaskFlow backend skeleton

**New files**:
- `include/nntile/runtime/backend.hh`
- `include/nntile/runtime/data_handle.hh`
- `include/nntile/runtime/task_handle.hh`
- `include/nntile/runtime/codelet.hh`
- `include/nntile/runtime/execution_hints.hh`
- `include/nntile/runtime/gelu.hh` (and other operations)
- `include/nntile/runtime/starpu/backend.hh`
- `include/nntile/runtime/starpu/data_handle.hh`
- `include/nntile/runtime/starpu/task_handle.hh`
- `src/runtime/backend.cc`
- `src/runtime/starpu/backend.cc`
- `src/runtime/starpu/gelu.cc` (and other operations)

**Files to modify**:
- `CMakeLists.txt` (add runtime selection options)
- `include/nntile/tile/*.hh` (use runtime:: namespace)
- `src/tile/*.cc` (use runtime:: namespace)
- `src/tensor/*.cc` (use runtime:: or tile::, add TaskGroup support)

---

## 7. Parallel Development Strategy

The following tracks can be developed **in parallel** by different teams:

```
Track A (MPI)                  ─────────────────────────────────┐
Track B (Execution Hints)      ─────────────────────────────────┤
Track C (Tensor Ctx)           ───────────────────────┐         │
Track D (Graph API)            ───────────────────────┼─────────┼───> Integration
Track E (Python)               ───────────────────────┘         │
Track F (Runtime Abstraction)  ═════════════════════════════════╪═══> Foundation
                                                                │
                                        Phase 6 (Dist) ─────────┘
```

**Dependencies**:
- **Track F is foundational** - should be prioritized first as other tracks depend on its interfaces
- Track B (Execution Hints) integrates into Track F's `ExecutionHints` structure
- Track C depends on Track F (uses `runtime::` interfaces)
- Phase 6 depends on Track A (MPI), Track C (ExecutionContext), Track D (DistributionStrategy)

**Recommended Development Order**:

1. **Track F (Runtime Abstraction)** - Start first, defines interfaces for all other tracks
   - Abstract interfaces (`Backend`, `DataHandle`, `TaskHandle`)
   - StarPU implementation
   - TaskHandle RAII wrapper
   
2. **Tracks A, B, D, E** - Can proceed in parallel once Track F interfaces are defined
   - Track A: MPI support integrates into `DataHandle::mpi_*` methods
   - Track B: Execution hints integrate into `ExecutionHints` struct
   - Track D: Graph API uses runtime interfaces
   - Track E: Python bindings wrap runtime API

3. **Track C** - Depends on Track B and Track F being stable

4. **Phase 6** - Final integration of distribution strategies

**Why Track F is Critical**:
1. Defines `TaskHandleOwner` for correct blocking semantics
2. Enables future TaskFlow/HPX backends without changing tile/tensor code
3. Establishes clean abstraction boundary between runtime and computation
4. `ExecutionHints` structure consolidates task placement from Track B

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

## 10. Design Refinements and Open Questions

### 10.1 Unified Tensor Concept

**Problem with current design**: Separate `input()`, `parameter()`, `constant()` methods create artificial distinctions. Everything should be a tensor.

**Refined design**: Single `tensor()` method with attributes determining role:

```cpp
namespace nntile::graph {

//! Tensor attributes that determine its role
struct TensorAttributes {
    bool requires_grad = false;      // Needs gradient computation
    bool persistent = false;         // Saved in checkpoints, survives across batches
    bool external = false;           // Data comes from outside the graph
    
    // Derived role
    TensorRole role() const {
        if (external && !persistent) return TensorRole::Input;
        if (external && persistent) return TensorRole::Constant;
        if (persistent && requires_grad) return TensorRole::Parameter;
        return TensorRole::Buffer;  // Intermediate activation
    }
};

class LogicalGraph {
public:
    //! Universal tensor creation - role determined by attributes
    TensorNode& tensor(
        const TensorSpec& spec,
        const std::string& name,
        TensorAttributes attrs = {}
    );
    
    // Convenience methods (call tensor() internally)
    TensorNode& input(const TensorSpec& spec, const std::string& name) {
        return tensor(spec, name, {.external = true});
    }
    
    TensorNode& parameter(const TensorSpec& spec, const std::string& name) {
        return tensor(spec, name, {.requires_grad = true, .persistent = true});
    }
    
    TensorNode& constant(const TensorSpec& spec, const std::string& name) {
        return tensor(spec, name, {.persistent = true, .external = true});
    }
    
    //! Mark tensor as output (can be any tensor)
    void mark_output(TensorNode& t, const std::string& output_name = "");
};

} // namespace nntile::graph
```

**Usage:**
```python
graph = LogicalGraph("model")

# All are just tensors with different attributes
x = graph.tensor(TensorSpec([seq, batch, embed]), "x", external=True)
w = graph.tensor(TensorSpec([embed, hidden]), "w", requires_grad=True, persistent=True)
scale = graph.tensor(TensorSpec([hidden]), "scale", persistent=True, external=True)  # Constant

# Operations create new tensors automatically
y = graph.matmul(x, w)  # y is a buffer (not external, not persistent)
z = graph.gelu(y)

# Any tensor can be marked as output
graph.mark_output(z, "logits")
```

### 10.2 Merging PhysicalGraph and ExecutableGraph

**Problem**: The separation between PhysicalGraph and ExecutableGraph is artificial. In practice, you always go Physical → Executable immediately.

**Analysis of why they were separate:**
1. Analyze memory before allocating (can do this before creating executable)
2. Create multiple executables from same physical (rare use case)

**Refined design**: Merge into single `CompiledGraph` with lazy allocation:

```cpp
namespace nntile::graph {

//! Compilation result - combines physical decisions + execution capability
class CompiledGraph {
private:
    LogicalGraph* logical_;
    
    // Physical decisions (computed at compile time)
    std::map<NodeId, TilingSpec> tilings_;
    std::map<NodeId, DistributionSpec> distributions_;
    DistributionStrategy dist_strategy_;
    
    // Execution state (allocated lazily on first use)
    enum class AllocationState { NotAllocated, Allocated };
    AllocationState alloc_state_ = AllocationState::NotAllocated;
    std::map<NodeId, std::unique_ptr<runtime::DataHandle>> handles_;
    
    void ensure_allocated();

public:
    //! Compile logical graph with distribution/tiling
    static CompiledGraph compile(
        LogicalGraph& logical,
        const DistributionStrategy& dist,
        const TilingStrategy& tiling = TilingStrategy::auto_tiling(),
        const std::map<std::string, Index>& shape_bindings = {}
    );
    
    // ═══════════════════════════════════════════════════════════
    // Analysis (available immediately after compile, before allocation)
    // ═══════════════════════════════════════════════════════════
    
    Index memory_per_device(int device_id) const;
    Index max_memory_per_device() const;
    Index estimate_communication() const;
    const TilingSpec& tiling(const std::string& tensor_name) const;
    const DistributionSpec& distribution(const std::string& tensor_name) const;
    std::string dump_plan() const;  // Human-readable plan
    
    // ═══════════════════════════════════════════════════════════
    // Execution (triggers allocation on first call)
    // ═══════════════════════════════════════════════════════════
    
    void bind_input(const std::string& name, const void* data, size_t size);
    void forward();
    void backward();
    void get_output(const std::string& name, void* data, size_t size);
    
    // ... rest of execution API
};

} // namespace nntile::graph
```

**Usage becomes simpler:**
```python
# Define
graph = LogicalGraph("model")
# ... build graph ...

# Compile with strategy (no allocation yet)
compiled = CompiledGraph.compile(
    graph,
    DistributionStrategy.fsdp(world_size=8),
    TilingStrategy.auto_tiling()
)

# Analyze before running
print(f"Memory per GPU: {compiled.memory_per_device(0) / 1e9:.2f} GB")
print(compiled.dump_plan())

# Execute (allocates on first call)
compiled.bind_input("x", data)
compiled.forward()
result = compiled.get_output("output")
```

### 10.3 Graph Composition (Stacking Logical Graphs)

**Requirement**: Build complex models by composing smaller logical graphs (like functions).

```cpp
namespace nntile::graph {

class LogicalGraph {
public:
    //! Embed another graph as a subgraph
    //! Returns mapping of subgraph outputs to nodes in this graph
    std::map<std::string, TensorNode*> embed(
        const LogicalGraph& subgraph,
        const std::map<std::string, TensorNode*>& input_bindings,
        const std::string& prefix = ""  // Prefix for tensor names
    );
    
    //! Create a reusable "function" from a graph
    //! (Clones the graph structure each time it's called)
    static std::function<std::map<std::string, TensorNode*>(
        LogicalGraph&,
        const std::map<std::string, TensorNode*>&
    )> as_function(const LogicalGraph& template_graph);
};

} // namespace nntile::graph
```

**Usage:**
```python
# Define attention block as reusable graph
def make_attention_block():
    block = LogicalGraph("attention")
    x = block.input(TensorSpec([seq, batch, embed]), "x")
    wq = block.parameter(TensorSpec([embed, embed]), "Wq")
    wk = block.parameter(TensorSpec([embed, embed]), "Wk")
    wv = block.parameter(TensorSpec([embed, embed]), "Wv")
    wo = block.parameter(TensorSpec([embed, embed]), "Wo")
    
    q = block.matmul(x, wq)
    k = block.matmul(x, wk)
    v = block.matmul(x, wv)
    attn = block.scaled_dot_product_attention(q, k, v)
    out = block.matmul(attn, wo)
    out = block.add(out, x)  # Residual
    
    block.mark_output(out, "output")
    return block

attention_template = make_attention_block()

# Build transformer by stacking
transformer = LogicalGraph("transformer")
x = transformer.input(TensorSpec([seq, batch, embed]), "input")

# Stack 12 attention blocks
hidden = x
for i in range(12):
    outputs = transformer.embed(
        attention_template,
        input_bindings={"x": hidden},
        prefix=f"layer_{i}_"  # layer_0_Wq, layer_0_Wk, etc.
    )
    hidden = outputs["output"]

transformer.mark_output(hidden, "output")

# All 12 blocks share the same structure but have separate parameters
print(f"Parameters: {transformer.num_parameters()}")  # 12 * 4 weight matrices
```

### 10.4 Missing Components Analysis

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| **Backward operations** | Explicit in graph | Done | User defines backward ops explicitly, no autograd engine |
| **Optimizer operations** | Not designed | High | Adam, SGD as explicit graph operations |
| **Loss function ops** | Not designed | High | CrossEntropy, MSE as graph operations |
| **Gradient checkpointing** | Natural fit | Medium | Just re-run forward ops in backward section of graph |
| **Mixed precision** | Mentioned briefly | High | Cast ops in graph, FP16/BF16 forward, FP32 accumulation |
| **Data loading pipeline** | Not designed | Medium | How data gets to inputs |
| **Profiling/debugging** | Not designed | Medium | Trace visualization, performance analysis |
| **Serialization format** | Not designed | Medium | Checkpoint compatibility |
| **Error handling** | Not designed | Medium | What happens when OOM, NaN, etc. |
| **Communication primitives** | Implicit | High | AllReduce, AllGather - explicit ops or hidden in distribution? |
| **Memory pool management** | Not designed | High | Reuse allocations across iterations |

### 10.5 Implementation Limitations and Risks

#### StarPU-Related Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Task granularity overhead** | Small tiles → scheduling overhead dominates | Auto-tune tile size, minimum tile threshold |
| **Limited reduction support** | STARPU_REDUX can be tricky | Implement explicit reduction trees |
| **MPI+StarPU complexity** | starpu_mpi is less mature | Thorough testing, fallback to manual MPI |
| **Worker binding limitations** | Can't always force GPU affinity | Use execution hints, accept some flexibility |
| **Profiling overhead** | FxT traces are large | Selective profiling, sampling |

#### Scalability Concerns

| Concern | At Scale | Mitigation |
|---------|----------|------------|
| **Graph construction time** | O(ops) - may be slow for huge models | Lazy construction, parallel building |
| **Compilation time** | Tiling analysis can be expensive | Cache compiled graphs, incremental recompilation |
| **Memory for graph metadata** | Many nodes = significant overhead | Use indices instead of pointers, compact representation |
| **Python/C++ boundary** | Many small calls are expensive | Batch operations, minimize crossings |

#### Correctness Risks

| Risk | Consequence | Mitigation |
|------|-------------|------------|
| **Tiling correctness** | Wrong results if tile boundaries mishandled | Extensive edge-case testing, formal verification for critical ops |
| **Distribution correctness** | Gradients wrong under sharding | Gradient checking, comparison with single-GPU |
| **Race conditions** | StarPU handles dependencies, but custom code might not | Careful API design, dependency analysis |
| **Numerical stability** | Tiled operations may accumulate differently | Test numerical equivalence, configurable precision |

### 10.6 Integration with Existing NNTile Code

**Challenge**: NNTile already has tensor/tile/layer code. How does graph API integrate?

**Strategy**: Graph API generates calls to existing tensor operations:

```cpp
// During ExecutableGraph::forward(), for a matmul op:
void execute_matmul(const PhysicalOp& op, ExecutionContext& ctx) {
    // Get existing NNTile tensors from handles
    auto& A = get_tensor<T>(op.inputs[0]);
    auto& B = get_tensor<T>(op.inputs[1]);
    auto& C = get_tensor<T>(op.outputs[0]);
    
    // Call existing tensor-level gemm
    tensor::gemm_async<T>(
        op.attrs.alpha, op.attrs.trans_a, A,
        op.attrs.trans_b, B,
        op.attrs.beta, C,
        op.attrs.ndim, op.attrs.batch_ndim,
        ctx.redux_mode
    );
}
```

**Benefits**:
- Reuse battle-tested tensor operations
- Graph API is "orchestration layer" on top of existing code
- Gradual migration - existing code still works

### 10.7 Explicit Backward Operations (No Autograd Engine)

**Design Decision**: No automatic differentiation engine. Users explicitly define backward operations in the logical graph.

**Rationale**:
1. **Simplicity**: No complex autograd machinery to implement and maintain
2. **Control**: User decides exactly which backward operations to use
3. **Flexibility**: Easy to implement custom backward passes, gradient checkpointing, etc.
4. **Transparency**: No "magic" - what you write is what executes
5. **Optimization**: Can fuse or reorganize backward ops manually for efficiency

**Example: Explicit Forward and Backward**:

```python
graph = LogicalGraph("mlp_with_backward")

# ═══════════════════════════════════════════════════════════════
# Forward tensors
# ═══════════════════════════════════════════════════════════════
x = graph.tensor(TensorSpec([batch, input_dim]), "x", external=True)
w1 = graph.tensor(TensorSpec([input_dim, hidden_dim]), "w1", 
                  requires_grad=True, persistent=True)
w2 = graph.tensor(TensorSpec([hidden_dim, output_dim]), "w2",
                  requires_grad=True, persistent=True)

# Forward operations
h = graph.matmul(x, w1, name="h")           # h = x @ w1
a = graph.gelu(h, name="a")                  # a = gelu(h)
y = graph.matmul(a, w2, name="y")           # y = a @ w2
graph.mark_output(y, "output")

# ═══════════════════════════════════════════════════════════════
# Backward tensors (gradients)
# ═══════════════════════════════════════════════════════════════
# Gradient of loss w.r.t. output (fed externally)
dy = graph.tensor(TensorSpec([batch, output_dim]), "dy", external=True)

# Gradient tensors for parameters (accumulated)
dw1 = graph.tensor(TensorSpec([input_dim, hidden_dim]), "dw1",
                   persistent=True)  # Accumulates across batches
dw2 = graph.tensor(TensorSpec([hidden_dim, output_dim]), "dw2",
                   persistent=True)

# ═══════════════════════════════════════════════════════════════
# Backward operations (explicit!)
# ═══════════════════════════════════════════════════════════════
# dy/dw2 = a^T @ dy  (gradient of w2)
dw2_batch = graph.matmul(a, dy, trans_a=True, name="dw2_batch")
graph.add_inplace(dw2, dw2_batch, name="dw2_accum")  # Accumulate

# dy/da = dy @ w2^T  (gradient flowing back through second matmul)
da = graph.matmul(dy, w2, trans_b=True, name="da")

# dy/dh = da * gelu'(h)  (gradient through gelu)
dh = graph.gelu_backward(da, h, name="dh")

# dy/dw1 = x^T @ dh  (gradient of w1)
dw1_batch = graph.matmul(x, dh, trans_a=True, name="dw1_batch")
graph.add_inplace(dw1, dw1_batch, name="dw1_accum")  # Accumulate

# Mark gradient outputs
graph.mark_output(dw1, "grad_w1")
graph.mark_output(dw2, "grad_w2")
```

**Execution**:
```python
compiled = CompiledGraph.compile(graph, DistributionStrategy.fsdp(8))

# Forward pass
compiled.bind_input("x", batch_data)
compiled.forward()
output = compiled.get_output("output")

# Compute loss externally and get gradient
loss, dy = compute_loss_and_grad(output, targets)

# Backward pass (explicit operations in same graph)
compiled.bind_input("dy", dy)
compiled.forward()  # Runs the backward ops too (they're just ops!)

# Get gradients
grad_w1 = compiled.get_output("grad_w1")
grad_w2 = compiled.get_output("grad_w2")
```

**Key Insight**: Forward and backward are just operations in the same graph. The "backward pass" is simply executing the backward operations, which happen to depend on forward tensors.

**Organizing Forward vs Backward**:

For clarity, users can use helper methods or conventions:

```python
class ModelGraph:
    """Helper to organize forward/backward in logical graph."""
    
    def __init__(self, name):
        self.graph = LogicalGraph(name)
        self.forward_ops = []
        self.backward_ops = []
    
    def forward_matmul(self, a, b, **kwargs):
        """Add matmul to forward pass."""
        result = self.graph.matmul(a, b, **kwargs)
        self.forward_ops.append(result.producer())
        return result
    
    def backward_matmul_dA(self, dC, B, **kwargs):
        """dA = dC @ B^T"""
        return self.graph.matmul(dC, B, trans_b=True, **kwargs)
    
    def backward_matmul_dB(self, A, dC, **kwargs):
        """dB = A^T @ dC"""
        return self.graph.matmul(A, dC, trans_a=True, **kwargs)
```

**Comparison with Autograd**:

| Aspect | Explicit Backward | Autograd Engine |
|--------|-------------------|-----------------|
| Implementation complexity | Low | High |
| User control | Full | Limited |
| Custom backward ops | Easy | Requires hooks |
| Gradient checkpointing | Explicit in graph | Separate mechanism |
| Code clarity | What you see is what runs | Hidden transformations |
| Debugging | Straightforward | Need to understand internals |
| Performance tuning | Direct control | Depends on autograd impl |

### 10.8 Revised Open Questions

1. **Dynamic shapes in compiled graphs**: Re-compile on shape change, or handle dynamically?
2. **Multi-GPU within single node**: Is it one CompiledGraph or multiple coordinated ones?
3. **Checkpoint format**: Custom binary, or compatibility layer with PyTorch/SafeTensors?
4. **Lazy vs eager tensor creation**: Should `graph.tensor()` create the node immediately?
5. **Error recovery**: What state is the graph in after a failed execution?
6. **Graph mutation**: Can you modify a LogicalGraph after operations are added?
7. **Thread safety**: Can multiple threads build the same LogicalGraph?
8. **Communication ops**: Should AllReduce/AllGather be explicit graph ops or hidden in distribution strategy?
9. **Execution order**: How to specify that backward ops run after forward ops in same graph?
10. **Gradient accumulation**: Best pattern for accumulating gradients across micro-batches?

---

## 11. Conclusion

NNTile 2.0 introduces a transformative high-level graph abstraction that enables:

1. **Automatic Distribution**: FSDP/DDP/TP without manual tensor partitioning
2. **Explicit Placement**: Fine-grained control over task execution location
3. **Multi-Node Scaling**: True distributed training across nodes
4. **Simplified API**: Declarative model definition with automatic optimization
5. **Proper Task Synchronization**: `TaskHandleOwner` wrapper enables waiting on specific tasks instead of all tasks
6. **Runtime Abstraction**: Pluggable backend system supports StarPU, TaskFlow, and future runtimes

### Key Architectural Decisions

1. **Keep Tile-Level**: The tile-level is retained for runtime abstraction and clean single-tile testing interface
2. **Keep C++ Sources**: Both tile-level and runtime-level keep `.cc` files (overhead is negligible)
3. **Runtime Abstraction Layer**: New `nntile::runtime` namespace with abstract interfaces:
   - `Backend` - runtime initialization and management
   - `DataHandle` - data registration and access
   - `TaskHandle` - task submission and synchronization
   - `Codelet` - kernel registration
4. **Flexible Backend Selection**:
   - Compile-time selection for zero-overhead production builds
   - Optional runtime selection for testing and benchmarking
5. **Directory Restructure**: `src/starpu/` → `src/runtime/starpu/` with room for `taskflow/`, `serial/`, etc.
6. **Multi-Stage Graph API**:
   - `LogicalGraph` - defines what to compute (operations, shapes, data flow)
   - `CompiledGraph` - combines physical decisions with lazy execution (merged Physical+Executable)
7. **Unified Tensor Concept**: Single `tensor()` method with attributes (`requires_grad`, `persistent`, `external`) determining role
8. **Graph Composition**: `embed()` method allows stacking smaller graphs into larger ones

### Design Refinements from Review

| Original Design | Refined Design | Rationale |
|-----------------|----------------|-----------|
| Separate `input()`, `parameter()`, `constant()` | Unified `tensor()` with attributes | Everything is a tensor; role derived from attributes |
| Separate PhysicalGraph + ExecutableGraph | Merged `CompiledGraph` | Simpler API; allocation is lazy on first execution |
| No graph composition | `embed()` for stacking graphs | Build transformers by composing attention blocks |

### Identified Risks and Mitigations

| Risk Category | Key Risks | Mitigations |
|---------------|-----------|-------------|
| **StarPU** | Task granularity overhead, MPI complexity | Auto-tune tile sizes, thorough MPI testing |
| **Scalability** | Graph construction time, compilation time | Lazy construction, cached compilation |
| **Correctness** | Tiling edge cases, distributed gradients | Extensive testing, gradient checking |
| **Integration** | Existing code compatibility | Graph API orchestrates existing tensor ops |

### Key Design Choices

- **Explicit backward operations** - No autograd engine; users define backward ops in the same logical graph
- **Forward and backward are just ops** - Same graph, same execution model
- **Gradient checkpointing is natural** - Just re-execute forward ops in backward section

### Missing Components (To Be Designed)

- Optimizer operations (Adam, SGD as explicit graph ops)
- Loss function operations (CrossEntropy, MSE as graph ops)
- Mixed precision support (cast operations in graph)
- Communication primitives (AllReduce as explicit or implicit?)
- Profiling and debugging tools
- Serialization and checkpoint format

### Benefits of Runtime Abstraction

| Aspect | Benefit |
|--------|---------|
| **Portability** | Same tile/tensor code works with any backend |
| **Future-Proofing** | Easy to add TaskFlow, HPX, or custom backends |
| **Testing** | Serial backend for debugging without StarPU |
| **Benchmarking** | Compare runtime performance with same workload |
| **Maintenance** | Backend-specific code isolated in `src/runtime/<backend>/` |

The phased approach allows parallel development across 6 tracks while maintaining backward compatibility. Track F (Runtime Abstraction) is foundational and should be prioritized as it defines interfaces used by all other tracks.

By implementing these changes, NNTile will evolve from a low-level tiled tensor library into a production-ready distributed deep learning framework competitive with state-of-the-art systems like DeepSpeed and Megatron-LM, while maintaining the flexibility to adopt future runtime systems as they emerge.
