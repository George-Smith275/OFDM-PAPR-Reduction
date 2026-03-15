"""这段代码定义了log partition function，为后面的可微分对齐函数服务"""
"""具体原理还没学"""
import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import logsumexp

@partial(jax.jit, static_argnames=("m", "n"))
def log_partition_function(W, m, n):
    """
    Computes the log partition function for all monotonic paths.


  limit_beta->infty Soft(W*beta) = Hard(W)

 limit_t->0 \nabla_w E_eps L(Soft((W+eps)/t)) = \nabla_w E L(Hard(W+eps))

    Args:
        W (jax.Array): The score matrix of shape (m, n).
        m (int): Number of rows.
        n (int): Number of columns.

    Returns:
        float: The log partition function, log Z.
    """
    # Initialize the DP table (alpha)
    alpha = jnp.zeros_like(W).at[0, :].set(W[0, :])

    # Outer loop over rows (i from 1 to m-1)
    def outer_loop_body(i, alpha):
        alpha_prev_row = alpha[i-1, :]
        shifted_alpha_prev_row = jnp.concatenate ([-1e6*jnp.ones((1)), alpha_prev_row[:-1]])
        w_curr_row = W[i, :]

        # Inner loop to compute cumulative logsumexp semiring logsumexp max
        def inner_loop_body(j, lse_val_and_row):
            lse_val, alpha_curr_row = lse_val_and_row
            # Update cumulative logsumexp
            new_lse_val = jnp.logaddexp(lse_val, shifted_alpha_prev_row[j])
            # Update current row of alpha
            alpha_curr_row = alpha_curr_row.at[j].set(w_curr_row[j] + new_lse_val)
            return new_lse_val, alpha_curr_row

        # Run the inner loop
        init_carry = (-jnp.inf, jnp.zeros(n))
        lse_final, alpha_curr_row = jax.lax.fori_loop(0, n, inner_loop_body, init_carry)

        alpha = alpha.at[i, :].set(alpha_curr_row)
        return alpha

    # Run the outer loop
    alpha = jax.lax.fori_loop(1, m, outer_loop_body, alpha)

    # Final log partition function is the logsumexp of the last row
    return logsumexp(alpha[-1, :])

# Create a function that computes the gradient, which are the marginals
@partial(jax.jit, static_argnames=("m", "n"))
def compute_marginals(W, m, n):
    """
    Computes the soft alignment matrix (marginals) by taking the gradient of
    the log partition function.
    """
    # The value_and_grad function returns both the output and the gradient
    log_z, grads = jax.value_and_grad(log_partition_function)(W, m, n)
    return log_z, grads


''' 
# Below are test codes
    # Sample score matrix W (m=5, n=6) - same as before
W = jnp.array([
    [ 0.62,       -1.2099999,  -0.59,       -1.36,       -1.78,       -2.86,      ],
    [-1.04,       -0.02,       -0.22,        0.78999996, -0.21,       -0.26,      ],
    [-0.62,       -0.96,       -1.55,       -0.05,       -1.37,        1.27,      ],
    [ 0.09,        1.5999999,  -0.5,        -0.45,        0.85999995, -0.39,      ]]
)

m, n = W.shape

# Compute the log partition function and the soft alignment matrix A
log_z_val, A_soft = compute_marginals(W, m, n)

# --- Print Results ---
print("Score Matrix W:")
print(W)
print(f"\nLog Partition Function (log Z): {log_z_val:.2f}")
print("\nSoft Alignment Matrix A (Marginals):")
print(A_soft.round(3))
print("\nRow sums of A (should be 1.0):")
print(A_soft.sum(axis=1))
'''