from os import environ
from pathlib import Path
from timeit import Timer

import jax
import jax.numpy as jnp


@jax.jit
def maxsumexp(xs: jax.Array) -> jax.Array:
    assert xs.ndim == 3, xs.shape
    batch_size, _, seq_len = xs.shape
    max_ = xs.max(axis=1, keepdims=True)
    sum_ = jnp.exp(xs - max_).sum(axis=1)
    out = jnp.stack([max_[:, 0, :], sum_])
    assert out.shape == (2, batch_size, seq_len), out.shape
    return out


if __name__ == '__main__':
    xla_dir = Path('xla-bytecode').absolute()
    environ['XLA_FLAGS'] = f'--xla_dump_to={xla_dir}'
    xs = jnp.ones((4, 256, 256))
    ys = maxsumexp(xs)
    print('input shape: ', xs.shape)
    print('output shape:', ys.shape)
    timer = Timer('maxsumexp(xs)', globals={'maxsumexp': maxsumexp, 'xs': xs})
    mean = timer.timeit(50) / 50
    print(f'timeit: {mean * 1e6:.3f} us')
