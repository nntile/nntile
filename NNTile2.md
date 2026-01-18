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
