/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine.
 *
 * @file tests/test_runtime_cycles.cc
 * Sequential init/shutdown cycles with handles that outlive a runtime:
 * the intrusive dead-list links must not survive shutdown, or the next
 * runtime's unlink writes through a freed handle. A host-only init is
 * refused with a catchable error.
 *
 * SPDX-License-Identifier: MIT
 *
 * @version 1.0.0
 * */

#include <nnhaul/nnhaul.hh>

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>

namespace
{

int failures = 0;

// A minimal device body so a task can be pending at handle destruction.
void cuda_read(void **buffers, void *cl_args) noexcept
{
    auto *out = *static_cast<int **>(cl_args);
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    cudaMemcpyAsync(out, buf->get_ptr<int>(), sizeof(int),
        cudaMemcpyDeviceToHost, nnhaul::cuda_stream());
}

void cuda_fill(void **buffers, void *cl_args) noexcept
{
    auto const pattern =
        static_cast<unsigned char>(*static_cast<int *>(cl_args));
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    for (std::size_t i = 0; i < buf->chunk_count(); ++i)
    {
        cudaMemsetAsync(buf->get_chunk_ptr<void>(i), pattern,
            buf->chunk_bytes(i), nnhaul::cuda_stream());
    }
}

void check(bool ok, char const *what)
{
    if (!ok)
    {
        std::fprintf(stderr, "runtime_cycles: FAIL: %s\n", what);
        ++failures;
    }
}

} // namespace

int main()
{
    std::size_t const cap = 4 * nnhaul::chunk_class_bytes();
    nnhaul::init(0, 1, std::size_t{1} << 20, cap);
    auto *a = new nnhaul::Handle(1024);
    auto *b = new nnhaul::Handle(1024);
    nnhaul::shutdown();

    // 'a' dies while no runtime exists; 'b' survives into the next one.
    delete a;
    nnhaul::init(0, 1, std::size_t{1} << 20, cap);
    delete b;
    // A later push through the dead list would dereference a stale tail
    // if shutdown had left b's link pointing at the first runtime.
    auto *c = new nnhaul::Handle(1024);
    delete c;
    nnhaul::shutdown();

    // A handle destroyed with a pending task must join that work, not
    // leave the task with a dangling Handle pointer.
    nnhaul::init(0, 1, std::size_t{1} << 20, cap);
    {
        nnhaul::Handle h(1024);
        nnhaul::Codelet fill("fill", nullptr, cuda_fill, nullptr);
        int pattern = 0x5a;
        nnhaul::insert(fill, {{&h, nnhaul::Access::Write}}, &pattern,
            sizeof(pattern), nnhaul::cuda_worker);
    } // h dies here with the task still pending
    nnhaul::wait();
    nnhaul::shutdown();

    // Dropping a temporary handle without invalidate() must still retire
    // its bindings; a one-chunk pool exposes a dropped marker at once.
    nnhaul::init(0, 1, std::size_t{1} << 20, nnhaul::chunk_class_bytes());
    {
        nnhaul::Codelet fill("fill", nullptr, cuda_fill, nullptr);
        for(int i = 0; i < 24; ++i)
        {
            nnhaul::Handle h(1024);
            int pattern = i;
            nnhaul::insert(fill, {{&h, nnhaul::Access::Write}}, &pattern,
                sizeof(pattern), nnhaul::cuda_worker);
            nnhaul::signal_graph_end();
            nnhaul::wait();
        } // h dies here, without invalidate()
    }
    nnhaul::shutdown();

    // A Write stacked over a Read acquire must publish the host bytes:
    // a later device read has to see them, not the stale replica.
    nnhaul::init(0, 1, std::size_t{1} << 20, cap);
    {
        nnhaul::Handle h(4);
        nnhaul::Codelet fill("fill", nullptr, cuda_fill, nullptr);
        nnhaul::Codelet read("read", nullptr, cuda_read, nullptr);
        int pattern = 0x11;
        int seen = -1;
        int *seen_ptr = &seen;
        nnhaul::insert(fill, {{&h, nnhaul::Access::Write}}, &pattern,
            sizeof(pattern), nnhaul::cuda_worker);
        nnhaul::signal_graph_end();
        nnhaul::wait();
        {
            auto rd = h.acquire(nnhaul::Access::Read);
            {
                auto wr = h.acquire(nnhaul::Access::Write);
                *static_cast<int *>(wr.get_ptr<int>()) = 0x22222222;
            }
        }
        nnhaul::insert(read, {{&h, nnhaul::Access::Read}}, &seen_ptr,
            sizeof(seen_ptr), nnhaul::cuda_worker);
        nnhaul::signal_graph_end();
        nnhaul::wait();
        check(seen == 0x22222222, "stacked-acquire publication");
    }
    nnhaul::shutdown();

    // Two un-retired batches must not sweep the same dropped handle
    // twice (its impl is engine-owned and freed at the first join).
    nnhaul::init(0, 1, std::size_t{1} << 20, cap);
    {
        nnhaul::Codelet fill("fill", nullptr, cuda_fill, nullptr);
        int pattern = 0x33;
        {
            nnhaul::Handle h(1024);
            nnhaul::insert(fill, {{&h, nnhaul::Access::Write}}, &pattern,
                sizeof(pattern), nnhaul::cuda_worker);
            nnhaul::signal_graph_end();
            nnhaul::wait();
        } // engine-owned: retires at the next sweep
        nnhaul::Handle g(1024);
        nnhaul::insert(fill, {{&g, nnhaul::Access::Write}}, &pattern,
            sizeof(pattern), nnhaul::cuda_worker);
        nnhaul::signal_graph_end(); // batch 1, un-retired
        nnhaul::insert(fill, {{&g, nnhaul::Access::Write}}, &pattern,
            sizeof(pattern), nnhaul::cuda_worker);
        nnhaul::signal_graph_end(); // batch 2, un-retired
        nnhaul::wait();
    }
    nnhaul::shutdown();

    // ReadWrite must see the current version, not an outdated host run.
    nnhaul::init(0, 1, std::size_t{1} << 20, cap);
    {
        nnhaul::Handle h(4);
        nnhaul::Codelet fill("fill", nullptr, cuda_fill, nullptr);
        int p1 = 0x11, p2 = 0x22;
        nnhaul::insert(fill, {{&h, nnhaul::Access::Write}}, &p1,
            sizeof(p1), nnhaul::cuda_worker);
        nnhaul::signal_graph_end();
        nnhaul::wait();
        { auto rd = h.acquire(nnhaul::Access::Read); (void)rd; }
        nnhaul::insert(fill, {{&h, nnhaul::Access::Write}}, &p2,
            sizeof(p2), nnhaul::cuda_worker);
        nnhaul::signal_graph_end();
        nnhaul::wait();
        {
            auto rw = h.acquire(nnhaul::Access::ReadWrite);
            check(*static_cast<int *>(rw.get_ptr<int>()) == 0x22222222,
                "ReadWrite sees the current version");
        }
    }
    nnhaul::shutdown();

    // An Acquired outliving shutdown must not pin a handle in the next
    // runtime nor abort at its destruction.
    nnhaul::init(0, 1, std::size_t{1} << 20, cap);
    auto *survivor = new nnhaul::Handle(1024);
    {
        nnhaul::Codelet fill("fill", nullptr, cuda_fill, nullptr);
        int pattern = 0x44;
        nnhaul::insert(fill, {{survivor, nnhaul::Access::Write}},
            &pattern, sizeof(pattern), nnhaul::cuda_worker);
        nnhaul::signal_graph_end();
        nnhaul::wait();
    }
    {
        auto late = survivor->acquire(nnhaul::Access::Read);
        nnhaul::shutdown();
    } // Acquired dies after the runtime
    nnhaul::init(0, 1, std::size_t{1} << 20, cap);
    delete survivor;
    nnhaul::shutdown();

    // A host-only init is a loud, catchable error, not a crash.
    bool refused = false;
    try
    {
        nnhaul::init(1, 0, std::size_t{1} << 20, 0);
    }
    catch (std::exception const &)
    {
        refused = true;
    }
    check(refused, "host-only init is refused");
    nnhaul::shutdown(); // idempotent after the failed init

    if (failures != 0)
    {
        return 1;
    }
    std::printf("runtime_cycles: PASS\n");
    return 0;
}
