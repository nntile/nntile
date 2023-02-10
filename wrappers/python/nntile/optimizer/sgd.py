import nntile
import numpy as np

class SGD:
    def __init__(self, params, grads, lr,
                 momentum=0., nesterov=False,
                 weight_decay=0., damping=0., dtype=np.float32):
        self.params = params
        self.grads = grads
        self.nesterov = nesterov
        self.num_iter = 0
        self.dtype=dtype
        if dtype == np.float32:
            self.lr = np.float32(lr)
            self.momentum = np.float32(momentum)
            self.weight_decay = np.float32(weight_decay)
            self.damping = np.float32(damping)
        elif dtype == np.float64:
            self.lr = np.float64(lr)
            self.momentum = np.float64(momentum)
            self.weight_decay = np.float64(weight_decay)
            self.damping = np.float64(damping)
        if momentum > 0:
            if dtype == np.float32:
                self.states = [nntile.tensor.Tensor_fp32() for p in params]
            elif dtype == np.float64:
                self.states = [nntile.tensor.Tensor_fp64() for p in params]

    def step_fp32_(self):
        for i in range(len(self.params)):
            if self.weight_decay != 0.:
                nntile.tensor.axpy2_async_fp32(self.weight_decay, self.params[i], self.grads[i])

            if self.momentum > 0:
                if self.num_iter == 0:
                    nntile.tensor.copy_fp32(self.grads[i], self.states[i])
                else:
                    nntile.tensor.axpy2_async_fp32(self.momentum - 1, self.states[i], self.states[i])
                    nntile.tensor.axpy2_async_fp32(1 - self.damping, self.grads[i], self.states[i])
                if self.nesterov:
                    nntile.tensor.axpy2_async_fp32(self.momentum, self.states[i], self.grads[i])
                else:
                    nntile.tensor.copy_async_fp32(self.states[i], self.grads[i])
            nntile.tensor.axpy2_async_fp32(self.lr, self.grads[i], self.params[i])
        self.num_iter += 1

    def step_fp64_(self):
        for i in range(len(self.params)):
            if self.weight_decay != 0.:
                nntile.tensor.axpy2_async_fp64(self.weight_decay, self.params[i], self.grads[i])

            if self.momentum > 0:
                if self.num_iter == 0:
                    nntile.tensor.copy_fp64(self.grads[i], self.states[i])
                else:
                    nntile.tensor.axpy2_async_fp64(self.momentum - 1, self.states[i], self.states[i])
                    nntile.tensor.axpy2_async_fp64(1 - self.damping, self.grads[i], self.states[i])
                if self.nesterov:
                    nntile.tensor.axpy2_async_fp64(self.momentum, self.states[i], self.grads[i])
                else:
                    nntile.tensor.copy_async_fp64(self.states[i], self.grads[i])
            nntile.tensor.axpy2_async_fp64(self.lr, self.grads[i], self.params[i])
        self.num_iter += 1

    def step(self):
        if self.dtype == np.float32:
            self.step_fp32_()
        elif self.dtype == np.float64:
            self.step_fp64_()

if __name__ == "__main__":
    pass