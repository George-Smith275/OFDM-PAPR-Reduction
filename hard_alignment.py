"""这段代码负责输出硬的对齐（当给定一个score matrix时，这个分数矩阵似乎对应着软对齐）"""
import jax
import jax.numpy as jnp
import os
from functools import partial

@partial(jax.jit, static_argnames=("m", "n"))
def find_optimal_path_jit(W, m, n):
    """
    Finds the optimal monotonic alignment path using dynamic programming.

    Args:
        W (jax.Array): The score matrix of shape (m, n).
        m (int): Number of rows.
        n (int): Number of columns.

    Returns:
        tuple: A tuple containing:
            - D (jax.Array): The dynamic programming table.
            - B (jax.Array): The backtracking table.
    """
    # Initialize DP and backtracking tables
    D = jnp.zeros_like(W).at[0, :].set(W[0, :])
    B = jnp.zeros_like(W, dtype=jnp.int32)

    # Outer loop over rows (i from 1 to m-1)
    def outer_loop_body(i, state):
        D, B = state
        d_prev_row = D[i-1, :]
        #shifted_d_prev_row = jnp.concatenate ([-jnp.inf*jnp.ones((1)), d_prev_row[:-1]])

        pad_value = jnp.where(i <= m-n, 0.0, -jnp.inf) # 第一行排头插零，其余行排头插负无穷。至于第一排为什么插零暂时不明
        shifted_d_prev_row = jnp.concatenate ([jnp.array([pad_value]), d_prev_row[:-1]]) # 这行是为了让硬对齐一直往东南方向走，不准往南走的

        w_curr_row = W[i, :]

        # Inner loop over columns (j from 0 to n-1) to find cumulative max
        def inner_loop_body(j, carry):
            max_val, max_idx, d_curr_row, b_curr_row = carry

            # Check if current element in previous row is the new max
            is_greater = shifted_d_prev_row[j] > max_val                        # 这行是为了让硬对齐一直往东南方向走，不准往南走的
            new_max_val = jnp.where(is_greater, shifted_d_prev_row[j], max_val) # 这行是为了让硬对齐一直往东南方向走，不准往南走的
            new_max_idx = jnp.where(is_greater, j-1, max_idx)                   # 这行是为了让硬对齐一直往东南方向走，不准往南走的

            """
            is_greater = d_prev_row[j] > max_val                                # 如果用这个，则硬对齐可以往南走
            new_max_val = jnp.where(is_greater, d_prev_row[j], max_val)         # 如果用这个，则硬对齐可以往南走
            new_max_idx = jnp.where(is_greater, j, max_idx)                     # 如果用这个，则硬对齐可以往南走
            """

            # Update current row of D and B
            d_curr_row = d_curr_row.at[j].set(w_curr_row[j] + new_max_val)
            b_curr_row = b_curr_row.at[j].set(new_max_idx)

            return (new_max_val, new_max_idx, d_curr_row, b_curr_row)

        # Run the inner loop  D[1,2] = max ( D[0,1]+ W[1,2])
        init_carry = (-jnp.inf, 0, jnp.zeros(n), jnp.zeros(n, dtype=jnp.int32))
        _, _, d_curr_row, b_curr_row = jax.lax.fori_loop(0, n, inner_loop_body, init_carry)

        D = D.at[i, :].set(d_curr_row)
        B = B.at[i, :].set(b_curr_row)
        return (D, B)

    # Run the outer loop
    D, B = jax.lax.fori_loop(1, m, outer_loop_body, (D, B))
    return D, B

@partial(jax.jit, static_argnames=("m",))
def backtrack_jit(D, B, m):
    """
    Performs backtracking to find the optimal path.

    Args:
        D (jax.Array): The DP table.
        B (jax.Array): The backtracking table.
        m (int): Number of rows.

    Returns:
        jax.Array: The optimal monotonic path (indices).

    这里的逻辑好像是这样的：先从D表的最后一行中找到最大值的位置，把该位置赋给next_path_j，
    并把该位置放在path的最后一个位置，然后从B表的最后一行开始，看该行的第next_path_j列是
    多少，把该值放在path的倒数第二个位置，然后把next_path_j的值更新为path里倒数第二个值，
    看B表倒数第二行的next_path_j列的值为多少，放到path的倒数第三个位置，如此下去。

    因为在计算D表的时候，D(i,j)的值是来自D表中第i-1行、第j-1列或更往前的一列，所以当我们
    用B表回溯时得到path时（注意path里的值对应着列的索引），path里的取值必然是单调递增的。
    """
    path = jnp.zeros(m, dtype=jnp.int32)

    # Find the end of the best path in the last row
    last_j = jnp.argmax(D[-1, :])
    path = path.at[-1].set(last_j)

    # Loop for backtracking (from m-2 down to 0)
    def backtrack_loop_body(i, path):
        row_idx = m - 2 - i
        # Get pointer from the next position in the path
        next_path_j = path[row_idx + 1]
        path = path.at[row_idx].set(B[row_idx + 1, next_path_j])
        return path

    path = jax.lax.fori_loop(0, m - 1, backtrack_loop_body, path)
    return path

"""
# Below are testing codes

    # Sample score matrix W (m=5, n=6)
W = jnp.array([
    [1.0, 1.5, 0.8, 0.3, 0.1, 0.0],
    [2.0, 1, 0.5, 0.9, 0.4, 0.1],
    [0.1, 0.2, 0.3, 0.4, 0.8, 0.5],
    [0.0, 0.1, 0.2, 0.3, 0.5, 0.9],
    [0.3, 0.2, 0.1, 0.4, 0.6, 0.8]
])


seed = int.from_bytes(os.urandom(4), "big")  # 4字节随机数
key = jax.random.key(seed)
key, subkey = jax.random.split(key)
W = jax.random.normal(subkey, shape=(4, 6))
W = jnp.round(W, decimals=2)


m, n = W.shape

# 1. Run the forward DP pass
D, B = find_optimal_path_jit(W, m, n)

# 2. Run backtracking to find the path
optimal_path = backtrack_jit(D, B, m)

# 3. Calculate max score and construct alignment matrix A
max_score = D[-1, optimal_path[-1]]
A = jnp.zeros_like(W).at[jnp.arange(m), optimal_path].set(1) # 生成一个与W维度相同的全零矩阵，对该矩阵的每一行i（其中i从0到m-1），在列optimal_path[i]的位置设置为1。

# --- Print Results ---
print("Score Matrix W:")
print(W.round(2))
print("\nDynamic Programming Table D:")
print(D.round(2))
print("\nBacktracking Table B:")
print(B.round(2))
print("\nOptimal Monotonic Path (indices):")
print(optimal_path)
print(f"\nMaximum Score: {max_score:.2f}")
print("\nOptimal Alignment Matrix A:")
print(A)
"""