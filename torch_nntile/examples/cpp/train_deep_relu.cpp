/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/examples/cpp/train_deep_relu.cpp
 * Tiny DeepReLU training smoke on CPU (LibTorch); nntile when registered.
 */

#include <torch/torch.h>

#include <torch_nntile/models/deep_relu.hh>

#include <iostream>

int main()
{
    torch::manual_seed(42);
    auto model = torch_nntile::models::DeepReLUImpl::tiny();
    model->to(torch::kCPU);
    auto x = torch::randn({8, 128});
    auto y = torch::randn({8, 10});
    torch::optim::SGD optim(model->parameters(), 1e-2);
    for (int step = 0; step < 3; ++step)
    {
        optim.zero_grad();
        auto pred = model->forward(x);
        auto loss = torch::mse_loss(pred, y);
        loss.backward();
        optim.step();
        std::cout << "step " << step << " loss "
                  << loss.item<float>() << "\n";
    }
    return 0;
}
