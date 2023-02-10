import nntile
import numpy as np

class Adam:
    def __init__(self, params, grads, lr,
                 beta1=0.9, beta2=0.999, amsgrad=False,
                 weight_decay=0., dtype=np.float32):
        self.params = params
        self.grads = grads
        self.amsgrad = amsgrad
        self.num_iter = 0
        self.dtype=dtype
        
        if dtype == np.float32:
            self.lr = np.float32(lr)
            self.beta1 = np.float32(beta1)
            self.beta2 = np.float32(beta2)
            self.weight_decay = np.float32(weight_decay)
            self.first_moments = [nntile.tensor.Tensor_fp32() for p in params]
            self.second_moments = [nntile.tensor.Tensor_fp32() for p in params]
            self.denoms = [nntile.tensor.Tensor_fp32() for p in params]
            if amsgrad:
                self.max_second_moments = [nntile.tensor.Tensor_fp32() for p in params]
        elif dtype == np.float64:
            self.lr = np.float64(lr)
            self.beta1 = np.float64(beta1)
            self.beta2 = np.float64(beta2)
            self.weight_decay = np.float64(weight_decay)
            self.first_moments = [nntile.tensor.Tensor_fp64() for p in params]
            self.second_moments = [nntile.tensor.Tensor_fp64() for p in params]
            self.denoms = [nntile.tensor.Tensor_fp64() for p in params]
            if amsgrad:
                self.max_second_moments = [nntile.tensor.Tensor_fp64() for p in params]

    def step_fp32_(self):
        for i in range(len(self.params)):
            if self.weight_decay != 0.:
                nntile.tensor.axpy2_async_fp32(self.weight_decay, self.params[i], self.grads[i])
            

            # Update first moments
            nntile.tensor.axpy2_async_fp32(self.beta1 - 1, self.first_moments[i], self.first_moments[i])
            nntile.tensor.axpy2_async_fp32(1 - self.beta1, self.grads[i], self.first_moments[i])

            # Update second moments
            nntile.tensor.prod_async_fp32(self.grads[i], self.grads[i])
            nntile.tensor.axpy2_async_fp32(self.beta2 - 1, self.second_moments[i], self.second_moments[i])
            nntile.tensor.axpy2_async_fp32(1 - self.beta2, self.grads[i], self.second_moments[i])

            # Scale 
            
            # Mult tensor by scalar
            
            step_size = -self.lr / (1 - self.beta1**self.num_iter)
            if self.amsgrad:
                nntile.tensor.max_async_fp32(self.second_moments[i], self.max_second_moments[i])
                nntile.tensor.copy_async_fp32(self.max_second_moments[i], self.denoms[i])
            else:
                nntile.tensor.copy_async_fp32(self.second_moments[i], self.denoms[i])
            
            nntile.tensor.sqrt_async_fp32(self.denoms[i])
            scale_factor = np.float32(1.) / np.sqrt(1 - self.beta2**self.num_iter,dtype=np.float32)
            nntile.tensor.axpy2_async_fp32(scale_factor, self.denoms[i], self.denoms[i])
            nntile.tensor.add_scalar_async_fp32(self.eps, self.denoms[i])

            nntile.tensor.addcdiv_async_fp32(step_size, 
                                            self.first_moments[i], self.denoms[i], self.params[i])
        self.num_iter += 1

    def step_fp64_(self):
        # for i in range(len(self.params)):
        #     if self.weight_decay != 0.:
        #         nntile.tensor.axpy2_async_fp64(self.weight_decay, self.params[i], self.grads[i])

        #     if self.momentum > 0:
        #         if self.num_iter == 0:
        #             nntile.tensor.copy_fp64(self.grads[i], self.states[i])
        #         else:
        #             nntile.tensor.axpy2_async_fp64(self.momentum - 1, self.states[i], self.states[i])
        #             nntile.tensor.axpy2_async_fp64(1 - self.damping, self.grads[i], self.states[i])
        #         if self.nesterov:
        #             nntile.tensor.axpy2_async_fp64(self.momentum, self.states[i], self.grads[i])
        #         else:
        #             nntile.tensor.copy_async_fp64(self.states[i], self.grads[i])
        #     nntile.tensor.axpy2_async_fp64(self.lr, self.grads[i], self.params[i])
        self.num_iter += 1

    def step(self):
        if self.dtype == np.float32:
            self.step_fp32_()
        elif self.dtype == np.float64:
            self.step_fp64_()

if __name__ == "__main__":
    pass