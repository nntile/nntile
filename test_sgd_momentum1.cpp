
#include <iostream>
#include <vector>
#include <cmath>

// Define common types used by the kernel.
using Index = std::size_t;
using Scalar = float;

// A simple fp32_t type, similar to your framework's definition.
struct fp32_t {
    float value;
    using repr_t = float;
    fp32_t() : value(0.0f) {}
    fp32_t(float v) : value(v) {}
    // Allow implicit conversion to float for arithmetic operations.
    operator float() const { return value; }
};

// Provide a stream operator to print fp32_t values.
std::ostream& operator<<(std::ostream& os, const fp32_t &val) {
    os << val.value;
    return os;
}

namespace nntile {
namespace kernel {
namespace sgd_momentum
{

// Implementation of the fused SGD with Momentum kernel.
template<typename T>
void cpu(Index num_iter, Index num_elems, Scalar momentum_, Scalar lr_,
         Scalar weight_decay_, const T *grad, T *velocity, T *p)
    noexcept
{
    using Y = typename T::repr_t;
    const Y momentum = momentum_, lr = lr_, weight_decay = weight_decay_;

    // Cycle over each element in the buffers.
    for(Index i = 0; i < num_elems; ++i)
    {
        // Read the parameter and gradient.
        Y p_val = static_cast<Y>(p[i]);
        Y grad_val = static_cast<Y>(grad[i]);

        // Apply weight decay if specified.
        if(weight_decay != 0)
        {
            grad_val += weight_decay * p_val;
        }

        // Compute the velocity update.
        Y v_val;
        if(num_iter == 1)
        {
            // For the first iteration, initialize velocity as lr * grad.
            v_val = lr * grad_val;
        }
        else
        {
            // For subsequent iterations, use momentum accumulation.
            Y v_prev = static_cast<Y>(velocity[i]);
            v_val = momentum * v_prev + lr * grad_val;
        }

        // Store the updated velocity.
        velocity[i] = static_cast<T>(v_val);

        // Update the parameter: p = p - v.
        p[i] = static_cast<T>(p_val - v_val);
    }
}

// Explicit instantiation for fp32_t.
template
void cpu<fp32_t>(Index num_iter, Index num_elems, Scalar momentum, Scalar lr,
                 Scalar weight_decay, const fp32_t *grad, fp32_t *velocity, fp32_t *p)
    noexcept;

} // namespace sgd_momentum
} // namespace kernel
} // namespace nntile

// Test the SGD with momentum kernel.
int main() {
    using namespace nntile::kernel::sgd_momentum;

    // Test configuration.
    const Index num_elems = 3;
    const Index num_iter = 1; // First iteration.
    const Scalar momentum = 0.9f;
    const Scalar lr = 0.1f;
    const Scalar weight_decay = 0.0f;

    // Initialize test buffers.
    std::vector<fp32_t> grad = { fp32_t(1.0f), fp32_t(2.0f), fp32_t(3.0f) };
    std::vector<fp32_t> velocity(num_elems, fp32_t(0.0f));
    std::vector<fp32_t> params = { fp32_t(0.5f), fp32_t(-1.5f), fp32_t(2.0f) };

    // Print the parameters before the update.
    std::cout << "Before update:\n";
    for(Index i = 0; i < num_elems; ++i) {
        std::cout << "param[" << i << "] = " << params[i] << "\n";
    }

    // Perform the SGD with momentum update.
    cpu<fp32_t>(num_iter, num_elems, momentum, lr, weight_decay,
                grad.data(), velocity.data(), params.data());

    // Print the parameters after the update.
    std::cout << "\nAfter update:\n";
    for(Index i = 0; i < num_elems; ++i) {
        std::cout << "param[" << i << "] = " << params[i] << "\n";
    }

    // Indicate that the test (and compilation) was successful.
    std::cout << "\nTest compiled and ran successfully.\n";

    return 0;
}

