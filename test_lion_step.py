import nntile
import nntile.utils.constructors as nnt
import numpy as np
import nntile.functions as nnf

# !export CUDA_VISIBLE_DEVICES=""
# !export STARPU_NCPU=2

config = nntile.starpu.Config(2)
nntile.starpu.init()


# Numpy implementation of Lion

def fused_lion_step(p, grad, m, lr, lambda_param, beta1, beta2, weight_decay, num_iter):
    """
    Performs a fused Lion optimizer step update.
    
    The Lion step is defined as:
      c = beta1 * m + (1 - beta1) * grad
      p = p - lr * (sign(c) + lambda_param * p)
      m = beta2 * m + (1 - beta2) * grad
    
    Optionally, weight decay is applied by modifying the gradient:
      grad = grad + weight_decay * p
      
    Parameters:
        p (np.ndarray): Parameters to be updated.
        grad (np.ndarray): Gradient array.
        m (np.ndarray): Momentum (first moment) buffer.
        lr (float): Learning rate.
        lambda_param (float): Penalty coefficient applied to parameters.
        beta1 (float): Coefficient for computing the exponential moving average (EMA) for c.
        beta2 (float): Coefficient for updating the momentum m.
        weight_decay (float): L2 regularization coefficient.
        num_iter (int): Current iteration number (not used in Lion update, maintained for API symmetry).
        
    Returns:
        tuple: Updated parameters p and momentum m.
    """
    # Optionally apply weight decay (L2 regularization) by modifying the gradient.
    if weight_decay != 0:
        grad = grad + weight_decay * p
    
    # Compute the intermediate value c as a moving average of the gradients.
    c = beta1 * m + (1 - beta1) * grad
    
    # Compute the update direction: sign(c) gives the discrete direction,
    # then add the penalty term (lambda_param * p).
    update_direction = np.sign(c) + lambda_param * p
    
    # Update the parameters in-place.
    p[:] = p - lr * update_direction
    
    # Update the momentum (first moment) buffer.
    m[:] = beta2 * m + (1 - beta2) * grad
    
    return p, m


np.random.seed(42)

# Create dummy parameters, gradients, and initialize momentum to zero.
p = np.random.randn(64, 64).astype(np.float32)
grad = np.random.randn(64, 64).astype(np.float32)
m = np.zeros_like(p)

# Set Lion optimizer hyperparameters:
lr = 0.001         # Learning rate.
lambda_param = 0.01  # Penalty coefficient.
beta1 = 0.9        # EMA coefficient for computing c.
beta2 = 0.99       # Momentum update coefficient.
weight_decay = 0.0 # No weight decay in this example.
num_iter = 1       # First iteration (kept for API symmetry).

print("Before update:")
print("Parameters p:\n", p)
print("Momentum m:\n", m)

# Perform the fused Lion step update.
p_final, m_final = fused_lion_step(p, grad, m, lr, lambda_param, beta1, beta2, weight_decay, num_iter)

print("\nAfter update:")
print("Updated parameters p:\n", p_final)
print("Updated momentum m:\n", m_final)


block_size = [4, 4]

p_nnt = nnt.from_array(p, block_size)
grad_nnt = nnt.from_array(grad, block_size)
m_nnt = nnt.from_array(m, block_size)


nntile.functions.fused_lion_step(
    p_nnt,
    grad_nnt,
    m_nnt,
    lr,
    lambda_param
    beta1,
    beta2,
    weight_decay,
    num_iter
)

p_nnt_np = nnt.to_numpy(p_nnt)
grad_nnt_np = nnt.to_numpy(grad_nnt)
m_nnt_np = nnt.to_numpy(m_nnt)



print(np.linalg.norm(p - p_nnt_np, "fro"))
print(np.linalg.norm(grad - grad_nnt_np, "fro"))
print(np.linalg.norm(m - m_nnt_np, "fro"))


p_nnt.unregister()
grad_nnt.unregister()
m_nnt.unregister()
nntile.starpu.shutdown()
