import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import os
from functools import partial
from gen_pam_qam import generate_random_qam, generate_qam_constellation
from gen_matrices import get_dft_matrix, generate_matrix_1, generate_matrix_2
from diff_mono_align import differentiable_monotonic_alignment_simple
from jaxopt import BFGS, OSQP
import optax
from datetime import datetime
import time
import scipy.io
from glob import glob
from jax.scipy.special import logsumexp
from jax import lax
from scipy.io import savemat
#from save_JAX_results import save_results_to_mat, robust_save
from save_JAX_results import (
    get_checkpoint_filename,
    save_checkpoint,
    load_checkpoint,
    save_final_mat,
)

from jax import grad, jit
from jax import debug

read_from_MATLAB = True
input_mat_filename = 'pattern_and_data_36c6_4QAM_100000problems.mat'

which_solver = 1 # 1 for BFGS, 2 for OSQP
if which_solver not in (1, 2):
    raise ValueError(f"which_solver 必须为 1 或 2，但传入的是 {which_solver}")

# Number of parallel restarts
if read_from_MATLAB:
    mat_data = scipy.io.loadmat(input_mat_filename)
    mat_pattern = mat_data['random_PRT_pattern'].astype(bool)
    #num_restarts = mat_pattern.shape[0]
    data_symbol = mat_data['data_symbol']
    N = mat_data['N'].item()
    nk = mat_data['nk'].item()
    K = N-nk
    mod_order = mat_data['mod_order'].item()
    #Test_Num = data_symbol.shape[0]  # number of problems instances(different data symbols X)
    Test_Num = 100  # test
else:
    Test_Num = 500  # number of problems instances(different data symbols X)
    # Define initial parameters and hyperparameters
    N = 24 # Number of subcarriers in the multicarrier system
    K = 20 # Number of data symbols that should not be bigger than N
    nk = N - K
    mod_order = 4 # Modulation order

L = 4 # Oversampling factor，must be a positive integer
Nus = 2*L*N+2
num_restarts = 20
num_steps = 50
learning_rate = 0.0075

##### 重要修正：以下矩阵只依赖于超参数(N,L),而不依赖于可优化参数(a_l,P), 因此完全可以提前生成一次，在 get_loss_b 外部缓存，避免每次 jit 编译或每次 step 时重复计算
F_inv_matrix = jnp.conj(get_dft_matrix(2 * L * N + 2))
matrix1 = generate_matrix_1(N, L)
matrix2 = generate_matrix_2(N, L)
FinvM1 = jnp.real(F_inv_matrix @ matrix1)
FinvM2 = jnp.imag(F_inv_matrix @ matrix2)

def objective(P_inner, s, FinvM1_useful, FinvM2_useful):
    linear_vals = s + FinvM1_useful @ P_inner[:N-K] - FinvM2_useful @ P_inner[K-N:]
    abs_vals = jnp.abs(linear_vals)
    return jnp.max(abs_vals)

# 只创建一次（放在 get_loss_b 外面，全局/超参数确定后）
if which_solver == 1:
    solver = BFGS(objective, maxiter=100, tol=1e-5, jit=True)

elif which_solver == 2:
    solver = OSQP(maxiter=100)

# 关键修复：完全分离内层优化器
def inner_optimization_wrapper(s, FinvM1_useful, FinvM2_useful, P_init, which_solver):
    if which_solver == 1:
        result = solver.run(P_init, s, FinvM1_useful, FinvM2_useful)
        P_optimized = result.params
        inner_loss = result.state.value

    elif which_solver == 2:
        epsilon = 0
        Q = jnp.eye(2*nk+1) * epsilon
        c1 = 1
        c2 = jnp.zeros((2*nk,1))
        cc = jnp.vstack([c1, c2])
        c = cc.reshape(-1)
        o = jnp.ones((Nus, 1))
        row1 = jnp.hstack([-o, FinvM1_useful, -FinvM2_useful])
        row2 = jnp.hstack([-o, -FinvM1_useful, FinvM2_useful])
        G = jnp.vstack([row1, row2])
        h = jnp.concatenate([-s, s])
        result = solver.run(params_obj=(Q, c), params_eq=None, params_ineq=(G, h))
        P_optimized = result.params.primal[-2*(N-K):]
        inner_loss = result.params.primal[0]
    return P_optimized, inner_loss

def get_loss_b(X, A_fixed=None):
    """
    通用损失函数：
    - 若传入 A，则不使用 a_l，A 固定；
    - 若不传入，则从 a_l 生成 A。
    """
    def loss_b(params):
        # 从 params 取出 P
        P = params['P']

        # 判断是否需要 a_l
        if A_fixed is None:
            a_l = params['a_l']
            A = differentiable_monotonic_alignment_simple(a_l, K, N)
        else:
            A = A_fixed

        X_hat = A.transpose() @ X
        X_s = matrix1 @ jnp.real(X_hat) + 1j * matrix2 @ jnp.imag(X_hat)
        s = jnp.real(F_inv_matrix @ X_s).reshape(-1)

        v = jnp.ones((1, K)) @ A
        zero_indices = jnp.where(v[0] == 0, size=nk)[0]
        FinvM1_useful = FinvM1[:, zero_indices]
        FinvM2_useful = FinvM2[:, zero_indices]

        # 关键修复：使用分离的内层优化器
        P_optimized, inner_loss = inner_optimization_wrapper(
            jax.lax.stop_gradient(s),
            jax.lax.stop_gradient(FinvM1_useful),
            jax.lax.stop_gradient(FinvM2_useful),
            jax.lax.stop_gradient(P),
            which_solver
        )

        # 计算损失值（仅用于外层优化）
        linear_vals = s + FinvM1_useful @ P_optimized[:N-K] - FinvM2_useful @ P_optimized[K-N:]
        abs_vals = jnp.abs(linear_vals)
        b = jnp.max(abs_vals)

        return b, (P_optimized, A)
    return loss_b

""" 生成多个初始化的值，然后分别去优化，最后找一个Loss最小的"""
import time
from datetime import datetime
import numpy as np
@jax.vmap
def init_params(key):
    key, subkey = jax.random.split(key)
    a_l = jax.random.normal(subkey, shape=(K, N))
    #a_l = jnp.abs(jax.random.normal(subkey, shape=(K, N)))
    #print(a_l)
    key, subkey1 = jax.random.split(key)
    key, subkey2 = jax.random.split(key)
    #P = jax.random.normal(subkey1, shape=(N,)) + 1j * jax.random.normal(subkey2, shape=(N,))
    P = jax.random.normal(subkey1, shape=(2*(N-K),))
    return {'a_l': a_l, 'P': P}

@jax.vmap
def init_params_fromMATLAB(key, path_hard_external):
    # key: 随机密钥
    # path_hard_external: 从外部传入的矩阵，形状 (K, N)，即 (12, 16)

    # 1. 分割 Key：只需要 key_noise 和 subkey1 了，因为路径不再随机生成
    key_noise, subkey1 = jax.random.split(key, 2)

    # 2. 生成背景噪声 (保持不变)
    noise = jax.random.normal(key_noise, shape=(K, N)) * 2

    # --- 删除原有的步骤 3, 4, 5 (采样、排序、One-Hot) ---
    # 3. 直接使用传入的矩阵
    # 确保它是 float64 (虽然我们在外部转换了，加一层保险)
    path_hard = path_hard_external.astype(jnp.float64)

    # 6. 路径平滑 (保持不变)
    # 使用传入的 path_hard 进行平滑
    #path_smooth = path_hard + 0.5 * (jnp.roll(path_hard, 1, axis=-1) + jnp.roll(path_hard, -1, axis=-1))
    path_smooth = path_hard

    # 7. 组合得到最终的软对齐矩阵 (保持不变)
    bias_strength = 1.0
    a_l = noise + path_smooth * bias_strength

    P = jax.random.normal(subkey1, shape=(2*(N-K),)) * 1.5

    return {'a_l': a_l, 'P': P}

@jax.jit
def single_step(params, opt_state,X):
    #def loss_wrapper(params):
    #    return get_loss_b(X)
    # 直接使用get_loss_b(X)创建损失函数
    loss_fn = get_loss_b(X)

    # 计算损失和梯度（使用has_aux=True处理辅助输出）
    (loss_val, (P_optimized, A_used)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # 更新参数
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_val, P_optimized, A_used

# Vmap the step function to operate over the leading batch dimension
vmapped_step = jax.vmap(single_step, in_axes=(0, 0, None))  # None 表示 X 在 batch 维度上不变（即所有restart共享同一个X）

# 从MATLAB读取PRT patterns和data symbols
if read_from_MATLAB:
    #mat_data = scipy.io.loadmat('pattern_and_symbol.mat')
    #data_symbol = mat_data['data_symbol']
    data_symbol_jax = jnp.array(data_symbol, dtype=jnp.complex128)

    mat_pattern = mat_data['random_PRT_pattern'].astype(bool)
    flipped_mat_pattern = ~mat_pattern
    selected_cols_np = jnp.zeros((num_restarts, K))
    for i in range(num_restarts):
        # 找到当前样本中为 True 的列索引
        indices = jnp.where(flipped_mat_pattern[i])[0]

        # 排序 (对应原代码中的 sorted_cols = jnp.sort(...))
        selected_cols_np = selected_cols_np.at[i].set(jnp.sort(indices))

    # 3. 转换为 JAX 数组并生成 One-Hot 矩阵 (对应 path_hard)
    # selected_cols_jax Shape: (50, 12)
    selected_cols_jax = jnp.array(selected_cols_np)

    # 生成 path_hard_batch, Shape: (50, 12, 16)
    # 注意：我们要在这里直接生成好 float64 的矩阵，传进去直接用
    path_hard_batch = jax.nn.one_hot(selected_cols_jax, num_classes=N, dtype=jnp.float64)

peak_fixedPRT_list = []
peak_opt_list = []

A_opt_results_list = []
P_opt_results_list = []
X_list_list = []
best_step_list = []
time_per_problem_list = []

save_interval = 2
resume_enabled = True

checkpoint_filename = get_checkpoint_filename(
    base_filename="results",
    N=N,
    K=K,
    num_restarts=num_restarts,
    num_steps=num_steps,
    learning_rate=learning_rate,
)

expected_meta = {
    "N": int(N),
    "K": int(K),
    "num_restarts": int(num_restarts),
    "num_steps": int(num_steps),
    "learning_rate": float(learning_rate),
    "which_solver": int(which_solver),
    "read_from_MATLAB": 1 if read_from_MATLAB else 0,
    "input_mat_filename": os.path.basename(str(input_mat_filename)) if read_from_MATLAB else "",
}

start_idx = 0
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if resume_enabled:
    found, state = load_checkpoint(checkpoint_filename, K, N, expected_meta=expected_meta)
    if found:
        start_idx = state["start_idx"]
        timestamp = state["timestamp"] if state["timestamp"] else timestamp
        A_opt_results_list = state["A_opt_results_list"]
        P_opt_results_list = state["P_opt_results_list"]
        X_list_list = state["X_list_list"]
        peak_opt_list = state["peak_opt_list"]
        best_step_list = state["best_step_list"]
        time_per_problem_list = state["time_per_problem_list"]

        print(f"**Resume enabled. Will continue from problem {start_idx + 1}/{Test_Num}")
    else:
        print("**No checkpoint found. Start from scratch.")
else:
    print("**Resume disabled. Start from scratch.")

fix_rand_key = True
if fix_rand_key:
    master_key = jax.random.key(2025)   # 主随机key（保证每次Test独立）
else:
    master_key = jax.random.key(int(time.time() * 1e9))
for i in range(start_idx, Test_Num):        ###起点改动
  print(f"\n===== Problem {i+1}/{Test_Num} =====")
  if which_solver == 1:
      print("Solver is BFGS")
  elif which_solver == 2:
      print("Solver is OSQP")

  start = time.time()

  # 生成随机数据符号
  if read_from_MATLAB:
    X = data_symbol_jax[i, :].reshape(-1, 1)
  else:
    master_key, subkey_data = jax.random.split(master_key)
    X = generate_random_qam(subkey_data, mod_order, K)

  #(peak_rand, P_rand) = get_loss_b(X)(initial_params)
  # 随机选择 K 个位置作为数据子载波（PRT 是剩下的）
  '''
  master_key, subkey_rand = jax.random.split(master_key)
  perm = jax.random.permutation(subkey_rand, N)
  data_idx = jnp.sort(perm[:K])
  A_rand = jnp.zeros((K, N))
  A_rand = A_rand.at[jnp.arange(K), data_idx].set(1.0)
  loss_fn_rand = get_loss_b(X, A_rand)
  master_key, subkey_init_rand = jax.random.split(master_key)
  peak_rand, P_rand = loss_fn_rand({'P': jax.random.normal(subkey_init_rand, shape=(2*(N-K),))})
  '''

  # 固定 PRT：最高频 nk = N-K 个子载波作为 PRT, 即 A 固定为选择前 K 个子载波的数据矩阵
  A_fixed = jnp.hstack([jnp.eye(K), jnp.zeros((K, N - K))])  # 固定对齐矩阵 A_fixed (K, N)
  loss_fn_fixed = get_loss_b(X, A_fixed)
  master_key, subkey_init_fix = jax.random.split(master_key)
  peak_fixed, P_fixed = loss_fn_fixed({'P': jax.random.normal(subkey_init_fix, shape=(2*(N-K),))})

  # Use JAX's PRNG to generate a batch of keys for reproducibility
  master_key, subkey_restart = jax.random.split(master_key)
  restart_keys = jax.random.split(subkey_restart, num_restarts)
  if read_from_MATLAB:
      initial_params = init_params_fromMATLAB(restart_keys, path_hard_batch)
  else:
    initial_params = init_params(restart_keys)

  # Initialize the optimizer state for each restart
  optimizer = optax.adam(learning_rate)
  opt_state = jax.vmap(optimizer.init)(initial_params)

  # 初始化最佳记录
  best_loss_so_far = jnp.full(num_restarts, jnp.inf)
  best_P_optimized_so_far = None  # 记录每次最优的内层优化结果
  best_step_so_far = jnp.full(num_restarts, -1)  # 记录每个restart取得最优解时的步数

  best_A_so_far = jnp.zeros((num_restarts, K, N), dtype=jnp.float32)

  ##### 重要修正：之前的代码最后打印的结果其实不是全程最优解，而只是最后一步迭代中每个重启（restart）得到的结果，然后选了其中 loss 最小的那个
  for step_num in range(num_steps):
      # 执行并行优化步骤
      initial_params, opt_state, loss_val, P_optimized, A_used = vmapped_step(initial_params, opt_state, X)

      # 判断哪些 restart 取得了改进
      improved = loss_val < best_loss_so_far

      # 更新最优 loss（每一个restart搜到的最优值都分别记录）
      best_loss_so_far = jnp.where(improved, loss_val, best_loss_so_far)

      # 更新 best_A_so_far
      improved_expanded_A = improved.reshape((improved.shape[0],) + (1,) * (A_used.ndim - 1))  # (R,1,1)
      best_A_so_far = jnp.where(improved_expanded_A, A_used, best_A_so_far)

      # 更新最优 P_optimized
      if best_P_optimized_so_far is None:
          best_P_optimized_so_far = P_optimized
      else:
          improved_expanded = improved.reshape((improved.shape[0],) + (1,) * (P_optimized.ndim - 1))
          best_P_optimized_so_far = jnp.where(improved_expanded, P_optimized, best_P_optimized_so_far)

      # 记录最优步数（只有在改进时才更新）
      best_step_so_far = jnp.where(improved, step_num, best_step_so_far)

  end = time.time()
  print(f"执行时间: {end - start:.4f}秒")

  # 结束后提取最优解
  best_idx = jnp.argmin(best_loss_so_far)
  best_step = best_step_so_far[best_idx]
  A_best = best_A_so_far[best_idx]
  P_best = best_P_optimized_so_far[best_idx]
  peak_best = best_loss_so_far[best_idx]

  A_opt_results_list.append(np.array(A_best, dtype=np.float32))
  P_opt_results_list.append(np.array(P_best, dtype=np.float32))
  X_list_list.append(np.array(X.reshape(-1), dtype=np.complex64))

  # 保存结果
  peak_fixedPRT_list.append(float(peak_fixed))
  peak_opt_list.append(float(peak_best))
  best_step_list.append(int(best_step))

  #print(f"Random PRT Peak: {float(peak_rand):.4f}")
  print(f"Fixed  PRT Peak: {float(peak_fixed):.4f}")
  print(
    f"Optimized Peak: {float(peak_best):.4f} "
    f"(restart {int(best_idx)}, step {best_step})"
  )

  end = time.time()
  time_per_problem_list.append(end-start)
  if (i + 1) % save_interval == 0 or (i + 1) == Test_Num:
    print("**正在保存 checkpoint...")      ###保存断点状态
    save_checkpoint(
        checkpoint_filename=checkpoint_filename,
        A_opt_list=A_opt_results_list,
        P_opt_list=P_opt_results_list,
        peak_opt_list=peak_opt_list,
        X_list=X_list_list,
        best_step_list=best_step_list,
        mod_order=mod_order,
        N=N,
        K=K,
        NumTrial=i + 1,
        L=L,
        time_per_problem_list=time_per_problem_list,
        num_restarts=num_restarts,
        num_steps=num_steps,
        learning_rate=learning_rate,
        timestamp=timestamp,
        fix_rand_key=fix_rand_key,
        which_solver=which_solver,
        read_from_MATLAB=read_from_MATLAB,
        input_mat_filename=(input_mat_filename if read_from_MATLAB else ""),
    )
  print(f"Execution time for this problem: {end - start:.4f} seconds")


  NumDone = min(
    len(peak_opt_list),
    len(A_opt_results_list),
    len(P_opt_results_list),
    len(X_list_list),
    len(best_step_list),
)

if which_solver == 1:
    out_name = "BFGS_results_N{}_K{}_nr{}_ns{}_lr{}_tn{}_{}.mat".format(
        int(N), int(K), num_restarts, num_steps,
        str(learning_rate).replace(".", "p"),
        NumDone,
        timestamp
    )
elif which_solver == 2:
    out_name = "OSQP_results_N{}_K{}_nr{}_ns{}_lr{}_tn{}_{}.mat".format(
        int(N), int(K), num_restarts, num_steps,
        str(learning_rate).replace(".", "p"),
        NumDone,
        timestamp
    )

save_final_mat(
    out_name=out_name,
    A_opt_results_list=A_opt_results_list,
    P_opt_results_list=P_opt_results_list,
    X_list_list=X_list_list,
    peak_opt_list=peak_opt_list,
    best_step_list=best_step_list,
    mod_order=mod_order,
    N=N,
    K=K,
    L=L,
)