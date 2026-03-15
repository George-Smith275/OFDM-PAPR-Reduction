"""这里是生成DFT矩阵"""
"""测试过N=8，跟MATLAB里DFTmtx = dftmtx(N) / sqrt(N);IDFTmtx = conj(DFTmtx);的效果一样"""
"""生成IDFT矩阵只需要找到get_dft_matrix(N)的共轭矩阵"""
import jax.numpy as jnp

def get_dft_matrix(N):
  """
  Constructs the N x N Discrete Fourier Transform (DFT) matrix.
  """
  # Create the indices for q and k
  q = jnp.arange(N)
  k = jnp.arange(N)

  # Compute the outer product of q and k, then the -2 * pi * i / N factor
  exponent = -2j * jnp.pi * jnp.outer(q, k) / N

  # Compute the DFT matrix elements
  F = jnp.exp(exponent) / jnp.sqrt(N)

  return F

# The result of the DFT on a vector 'x' is F_matrix @ x
# Note: jax.numpy.fft.fft(x) might use a different normalization factor.

"""测试"""
"""
dftmtx = get_dft_matrix(6)
print(dftmtx)
print(jnp.conj(dftmtx))
print(dftmtx @ jnp.conj(dftmtx))
"""