/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/grad_mode.hh
 * Gradient recording mode - PyTorch-like no_grad().
 *
 * When grad is disabled, autograd ops add to the logical graph but do NOT
 * create OpNode or set producer on outputs. This allows modules with custom
 * backward to run forward without building an autograd chain, then wrap the
 * output in a single ModuleOp.
 *
 * @version 1.1.0
 * */

#pragma once

namespace nntile::graph
{

//! Gradient recording mode - controls whether autograd ops register producer.
//! Similar to torch.no_grad() / torch.is_grad_enabled().
class GradMode
{
public:
    //! Check if gradient recording is enabled
    static bool is_enabled();

    //! Set gradient recording (for testing; prefer Guard)
    static void set_enabled(bool enabled);

    //! RAII guard to temporarily disable gradient recording.
    //! Use: { GradMode::Guard g; module.build_forward(...); }
    class Guard
    {
    public:
        Guard();
        ~Guard();
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;

    private:
        bool prev_;
    };
};

} // namespace nntile::graph
