# 这里生成用于将长度为N的字符列向量变换成长度为2N+2的字符列向量的矩阵
# 长度为2N+2的字符列向量满足hermitian symmetry，使得用反向傅里叶变换生成的时域信号为实信号
import jax.numpy as jnp
from jax import random
from dft import get_dft_matrix

# 这个版本里，最后N行是exchange matrix
# 输入变量为列的数量
def generate_matrix_1(N,L):
    # 计算总行数和列数
    rows = 2 * L * N + 2
    cols = N
    # 初始化全零矩阵
    M = jnp.zeros((rows, cols))
    # 设置单位矩阵部分（第二行到第N+1行）
    identity_block = jnp.eye(N)
    M = M.at[1:N+1, :].set(identity_block)
    # 生成交换矩阵（通过翻转单位矩阵的列）
    exchange_matrix = jnp.flip(jnp.eye(N), axis=1)
    # 设置交换矩阵部分（第2LN-N+3行到最后一行）
    M = M.at[(2*L*N-N+2):rows, :].set(exchange_matrix)
    return M

# 这个版本里，最后N行是负的exchange matrix
# 输入变量为列的数量
def generate_matrix_2(N,L):
    # 计算总行数和列数
    rows = 2 * L * N + 2
    cols = N
    # 初始化全零矩阵
    M = jnp.zeros((rows, cols))
    # 设置单位矩阵部分（第二行到第N+1行）
    identity_block = jnp.eye(N)
    M = M.at[1:N+1, :].set(identity_block)
    # 生成交换矩阵（通过翻转单位矩阵的列）
    exchange_matrix = jnp.flip(-jnp.eye(N), axis=1)
    # 设置交换矩阵部分（第2LN-N+3行到最后一行）
    M = M.at[(2*L*N-N+2):rows, :].set(exchange_matrix)
    return M

"""
# 示例：生成N=2时的矩阵
N = 3
L = 3
matrix1 = generate_matrix_1(N,L)
matrix2 = generate_matrix_2(N,L)
print(matrix1)
print(matrix2)

# 生成随机复数字符向量并试验结果
key = random.PRNGKey(42)
shape = (N, 1)
real_part = random.normal(key, shape)
key, subkey = random.split(key)
imag_part = random.normal(subkey, shape)
symbol = real_part + 1j * imag_part
print("symbol:")
print(symbol)

real_symbol = jnp.real(symbol)
imag_symbol = jnp.imag(symbol)
long_symbol = matrix1 @ jnp.real(symbol) + 1j * matrix2 @ jnp.imag(symbol)
print("long_symbol:")
print(long_symbol)

IDFT_mtx = jnp.conj( get_dft_matrix(2*L*N+2) )
time_domain_samples = IDFT_mtx @ long_symbol
print("time_domain_samples:")
print(time_domain_samples)

zeros_to_add = jnp.zeros(((L - 1) * N,1), dtype=symbol.dtype)
print(zeros_to_add.shape)
print(jnp.concatenate([symbol, zeros_to_add]))
symbol = jnp.concatenate([symbol, zeros_to_add])
print(symbol)
"""