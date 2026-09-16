/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/execution_driver.hh
 * ExecutionDriver: run an already-lowered TileGraph. Does not compile
 * TensorGraph → TileGraph.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

#include <nntile/runtime.hh>
#include <nntile/tile/graph_decl.hh>

namespace nntile
{

//! Run a TileGraph on StarPU. Does **not** lower a TensorGraph.
//! In-process: ``RuntimeExecutionDriver``. Remote daemon:
//! ``RemoteExecutionDriver`` (``NNTILE_DRIVER_SOCKET``).
class ExecutionDriver
{
  public:
    virtual ~ExecutionDriver() = default;

    //! Compile new tile ops (if any) and submit them. Does not wait.
    virtual void submit(const TileGraph &graph) = 0;

    //! Drain StarPU after one or more ``submit`` calls.
    virtual void wait() = 0;
};

//! In-process driver wrapping ``Runtime``. bind/gather stay on ``runtime()``.
class RuntimeExecutionDriver : public ExecutionDriver
{
  public:
    RuntimeExecutionDriver() = default;

    explicit RuntimeExecutionDriver(const TileGraph &graph)
        : graph_(&graph), runtime_(std::make_unique<Runtime>(graph))
    {
    }

    void submit(const TileGraph &graph) override
    {
        attach(graph);
        runtime_->compile();
        const size_t end = runtime_->execution_op_count();
        if (submitted_end_ > end)
        {
            submitted_end_ = 0;
        }
        if (end > submitted_end_)
        {
            runtime_->execute_range(submitted_end_, end);
            submitted_end_ = end;
        }
    }

    void wait() override
    {
        if (runtime_)
        {
            runtime_->wait();
        }
    }

    Runtime &runtime()
    {
        if (!runtime_)
        {
            throw std::runtime_error(
                "RuntimeExecutionDriver: no TileGraph attached; call submit");
        }
        return *runtime_;
    }

    const Runtime &runtime() const
    {
        if (!runtime_)
        {
            throw std::runtime_error(
                "RuntimeExecutionDriver: no TileGraph attached; call submit");
        }
        return *runtime_;
    }

  private:
    void attach(const TileGraph &graph)
    {
        if (!runtime_)
        {
            graph_ = &graph;
            runtime_ = std::make_unique<Runtime>(graph);
            return;
        }
        if (graph_ != &graph)
        {
            throw std::invalid_argument(
                "RuntimeExecutionDriver::submit: TileGraph is not this "
                "session");
        }
    }

    const TileGraph *graph_ = nullptr;
    std::unique_ptr<Runtime> runtime_;
    size_t submitted_end_ = 0;
};

} // namespace nntile
