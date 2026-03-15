"""这段代码让单调对齐可微，可输出优化过的硬对齐"""
"""具体原理没学"""
import jax
import jax.numpy as jnp
import os
from functools import partial
from jax.scipy.special import logsumexp
from hard_alignment import find_optimal_path_jit, backtrack_jit
from log_partition import log_partition_function

# --- Helper functions (assumed to be defined as in the previous answer) ---
# find_optimal_path_jit(...)
# backtrack_jit(...)
# log_partition_function(...)


@partial(jax.jit, static_argnames=("m", "n","hard"))
def differentiable_monotonic_alignment_simple(W, m, n,hard=True):
    """
    Performs monotonic alignment using the stop_gradient trick.

    Forward pass returns the one-hot hard alignment matrix.
    Backward pass gradient is the soft marginals matrix.
    """
    # 1. Compute the "soft" alignment (the differentiable proxy)
    A_soft = jax.grad(log_partition_function)(W, m, n)
    if not hard: return A_soft

    # 2. Compute the "hard" alignment (non-differentiable)
    D, B = find_optimal_path_jit(W, m, n)
    path = backtrack_jit(D, B, m)
    A_hard = jnp.zeros_like(W).at[jnp.arange(m), path].set(1.0) # 生成一个与W维度相同的全零矩阵，对该矩阵的每一行i（其中i从0到m-1），在列optimal_path[i]的位置设置为1。

    # 3. Combine them with the stop_gradient trick
    return jax.lax.stop_gradient(A_hard - A_soft) + A_soft

# 下面这个def跟上面那个一样，纯粹是为了输出D表和B表
def differentiable_monotonic_alignment_simple_D_and_B(W, m, n,hard=True):
    """
    Performs monotonic alignment using the stop_gradient trick.

    Forward pass returns the one-hot hard alignment matrix.
    Backward pass gradient is the soft marginals matrix.
    """
    # 1. Compute the "soft" alignment (the differentiable proxy)
    A_soft = jax.grad(log_partition_function)(W, m, n)
    if not hard: return A_soft

    # 2. Compute the "hard" alignment (non-differentiable)
    D, B = find_optimal_path_jit(W, m, n)
    path = backtrack_jit(D, B, m)
    A_hard = jnp.zeros_like(W).at[jnp.arange(m), path].set(1.0)

    # 3. Combine them with the stop_gradient trick
    return D,B

'''
# --- Example Usage ---


seed = int.from_bytes(os.urandom(4), "big")  # 4字节随机数
key = jax.random.key(seed)
key, subkey = jax.random.split(key)
# 生成标准正态分布并取绝对值，确保所有值非负
#W = jnp.abs(jax.random.normal(subkey, shape=(5, 6)))
W = jax.random.normal(subkey, shape=(5, 8))
W = jnp.round(W, decimals=2)


m, n = W.shape
jnp.set_printoptions(precision=4, suppress=True, linewidth=150)
"""
start_index = n - m + 1  # 后 m-1 个元素的起始索引
W = W.at[0, start_index:].set(-100) # 将后(m-1)个元素设为-100
W = W.at[0, 0].set(1)
W = W.at[0, 1].set(1.5)
W = W.at[1, 0].set(2)
W = W.at[1, 1].set(1)
"""

# --- 1. Test the Forward Pass ---
print("A random score matrix W:")
print(W)
D,B = differentiable_monotonic_alignment_simple_D_and_B(W, m, n)
print("The DP table D:")
print(D)
print("The backtracking table B:")
print(B)
print("--- FORWARD PASS (Hard Alignment) ---")
A_hard_output = differentiable_monotonic_alignment_simple(W, m, n)
print("Output of the function (A_hard):")
print(A_hard_output)

# --- 2. Test the Backward Pass ---
print("\n--- BACKWARD PASS (Soft Gradient) ---")
def dummy_loss(W, m, n):
    A = differentiable_monotonic_alignment_simple(W, m, n)
    return jnp.sum(A*A)

grad_A_soft = jax.grad(dummy_loss)(W, m, n)
print("Gradient of the function (A_soft):")
print(grad_A_soft)
'''