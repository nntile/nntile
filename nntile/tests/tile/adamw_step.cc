/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/adamw_step.cc
 * AdamW step for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/adamw_step.hh"
#include "nntile/starpu/adamw_step.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> grad({2, 3, 4}), m({2, 3, 4}), v({2, 3, 4}), p({2, 3, 4});
    Tile<T> m_ref({2, 3, 4}), v_ref({2, 3, 4}), p_ref({2, 3, 4});

    auto grad_local = grad.acquire(STARPU_W);
    auto m_local = m.acquire(STARPU_W);
    auto v_local = v.acquire(STARPU_W);
    auto p_local = p.acquire(STARPU_W);
    auto m_ref_local = m_ref.acquire(STARPU_W);
    auto v_ref_local = v_ref.acquire(STARPU_W);
    auto p_ref_local = p_ref.acquire(STARPU_W);
    for(Index i = 0; i < grad.nelems; ++i)
    {
        grad_local[i] = Y(0.01 * (i+1));
        m_local[i] = Y(0.1 * (i+1));
        v_local[i] = Y(0.2 * (i+1));
        p_local[i] = Y(1.0 + 0.05 * i);
        m_ref_local[i] = m_local[i];
        v_ref_local[i] = v_local[i];
        p_ref_local[i] = p_local[i];
    }
    grad_local.release();
    m_local.release();
    v_local.release();
    p_local.release();
    m_ref_local.release();
    v_ref_local.release();
    p_ref_local.release();

    Index num_iter = 3;
    Scalar beta1 = 0.9, beta2 = 0.95, eps = 1e-8, lr = 1e-3, wd = 1e-2;
    starpu::adamw_step.submit<std::tuple<T>>(num_iter, p.nelems, beta1, beta2,
            eps, lr, wd, grad, m, v, p);
    adamw_step<T>(num_iter, beta1, beta2, eps, lr, wd, grad, m_ref, v_ref,
            p_ref);

    m_local.acquire(STARPU_R);
    v_local.acquire(STARPU_R);
    p_local.acquire(STARPU_R);
    m_ref_local.acquire(STARPU_R);
    v_ref_local.acquire(STARPU_R);
    p_ref_local.acquire(STARPU_R);
    for(Index i = 0; i < grad.nelems; ++i)
    {
        TEST_ASSERT(Y(m_local[i]) == Y(m_ref_local[i]));
        TEST_ASSERT(Y(v_local[i]) == Y(v_ref_local[i]));
        TEST_ASSERT(Y(p_local[i]) == Y(p_ref_local[i]));
    }
    m_local.release();
    v_local.release();
    p_local.release();
    m_ref_local.release();
    v_ref_local.release();
    p_ref_local.release();
}

int main(int argc, char **argv)
{
    int ncpu=1, ncuda=0, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
