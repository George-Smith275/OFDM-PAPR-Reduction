import os
import tempfile
import numpy as np
import scipy.io
import jax
from gen_pam_qam import generate_qam_constellation


def _to_numpy_list(x_list, dtype=None):
    """list[JAX/NumPy] -> list[np.ndarray]"""
    if x_list is None or len(x_list) == 0:
        return []
    out = []
    for x in x_list:
        arr = jax.device_get(x)
        arr = np.array(arr)
        if dtype is not None:
            arr = arr.astype(dtype)
        out.append(arr)
    return out


def _stack_or_empty(x_list, empty_shape, dtype):
    """
    把 list of arrays 堆成一个 ndarray。
    如果为空，返回一个合法的空数组，避免 savemat 出错。
    """
    if x_list is None or len(x_list) == 0:
        return np.empty(empty_shape, dtype=dtype)
    return np.stack(x_list, axis=0).astype(dtype)


def get_checkpoint_filename(base_filename, N, K, num_restarts, num_steps, learning_rate):
    lr_str = f"{learning_rate:.6g}".replace(".", "p")
    return f"{base_filename}_N{N}_K{K}_nr{num_restarts}_ns{num_steps}_lr{lr_str}_checkpoint.mat"


def atomic_savemat(filename, save_dict, do_compression=True):
    """
    原子保存：
    先写临时文件，再替换正式文件。
    防止保存过程中断电/崩溃导致旧文件也没了。
    """
    directory = os.path.dirname(os.path.abspath(filename)) or "."
    fd, tmp_path = tempfile.mkstemp(suffix=".mat", dir=directory)
    os.close(fd)
    try:
        scipy.io.savemat(tmp_path, save_dict, do_compression=do_compression)
        os.replace(tmp_path, filename)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def save_checkpoint(
    checkpoint_filename,
    A_opt_list,
    P_opt_list,
    peak_opt_list,
    X_list,
    best_step_list,
    mod_order,
    N,
    K,
    NumTrial,
    L,
    time_per_problem_list,
    num_restarts,
    num_steps,
    learning_rate,
    timestamp,
    fix_rand_key,
    which_solver,
    read_from_MATLAB,
    input_mat_filename,
):
    """
    保存可恢复的 checkpoint。
    这里保存的是“已完成 problem 的聚合结果”，用于下次继续跑。
    """
    qam_constellation = np.array(generate_qam_constellation(mod_order))

    A_numpy = _to_numpy_list(A_opt_list, dtype=np.float64)
    P_numpy = _to_numpy_list(P_opt_list, dtype=np.float64)
    peak_numpy = [float(jax.device_get(x)) for x in peak_opt_list] if len(peak_opt_list) > 0 else []
    X_numpy = _to_numpy_list(X_list, dtype=np.complex128)
    best_step_numpy = [int(x) for x in best_step_list] if len(best_step_list) > 0 else []
    time_numpy = [float(x) for x in time_per_problem_list] if len(time_per_problem_list) > 0 else []

    A_arr = _stack_or_empty(A_numpy, empty_shape=(0, K, N), dtype=np.float64)
    P_arr = _stack_or_empty(P_numpy, empty_shape=(0, 2 * (N - K)), dtype=np.float64)
    X_arr = _stack_or_empty(X_numpy, empty_shape=(0, K), dtype=np.complex128)

    peak_arr = np.array(peak_numpy, dtype=np.float64).reshape(-1, 1)
    best_step_arr = np.array(best_step_numpy, dtype=np.int32).reshape(-1, 1)
    time_arr = np.array(time_numpy, dtype=np.float64).reshape(-1, 1)

    save_dict = {
        "A_opt_results": A_arr,
        "P_opt_results": P_arr,
        "peak_opt_list": peak_arr,
        "X_list": X_arr,
        "best_step_list": best_step_arr,
        "time_per_problem_list": time_arr,
        "qam_constellation": qam_constellation.reshape(-1, 1),

        "mod_order": np.array([[int(mod_order)]], dtype=np.int32),
        "N": np.array([[int(N)]], dtype=np.int32),
        "K": np.array([[int(K)]], dtype=np.int32),
        "NumTrial": np.array([[int(NumTrial)]], dtype=np.int32),
        "L": np.array([[int(L)]], dtype=np.int32),
        "num_restarts": np.array([[int(num_restarts)]], dtype=np.int32),
        "num_steps": np.array([[int(num_steps)]], dtype=np.int32),
        "learning_rate": np.array([[float(learning_rate)]], dtype=np.float64),

        "starting_timestamp": np.array([[str(timestamp)]], dtype=object),
        "fix_rand_key": np.array([[1 if fix_rand_key else 0]], dtype=np.int32),
        "which_solver": np.array([[int(which_solver)]], dtype=np.int32),
        "read_from_MATLAB": np.array([[1 if read_from_MATLAB else 0]], dtype=np.int32),
        "input_mat_filename": np.array([[os.path.basename(str(input_mat_filename))]], dtype=object),
    }

    atomic_savemat(checkpoint_filename, save_dict, do_compression=True)
    print(f"**Checkpoint saved: {checkpoint_filename} (NumTrial={NumTrial})")


def load_checkpoint(
    checkpoint_filename,
    K,
    N,
    expected_meta=None,
):
    """
    读取 checkpoint。
    返回：
        found, state_dict
    """
    if not os.path.exists(checkpoint_filename):
        return False, None

    data = scipy.io.loadmat(checkpoint_filename)

    def _scalar(name, default=None):
        if name not in data:
            return default
        arr = data[name]
        try:
            arr = np.asarray(arr).squeeze()
            if arr.shape == ():
                return arr.item()
            return arr.flat[0].item() if hasattr(arr.flat[0], "item") else arr.flat[0]
        except Exception:
            return default

    def _string(name, default=""):
        if name not in data:
            return default

        val = data[name]
        try:
            while isinstance(val, np.ndarray):
                if val.size == 0:
                    return default
                val = val.squeeze()
                if isinstance(val, np.ndarray) and val.shape == ():
                    val = val.item()
                elif isinstance(val, np.ndarray):
                    val = val.flat[0]
            return str(val)
        except Exception:
            return default

    meta = {
        "N": int(_scalar("N", -1)),
        "K": int(_scalar("K", -1)),
        "num_restarts": int(_scalar("num_restarts", -1)),
        "num_steps": int(_scalar("num_steps", -1)),
        "learning_rate": float(_scalar("learning_rate", np.nan)),
        "which_solver": int(_scalar("which_solver", -1)),
        "read_from_MATLAB": int(_scalar("read_from_MATLAB", -1)),
        "input_mat_filename": _string("input_mat_filename", ""),
    }

    if expected_meta is not None:
        for k, v in expected_meta.items():
            if k not in meta:
                continue

            if k == "learning_rate":
                if not np.isclose(float(meta[k]), float(v), rtol=0, atol=1e-12):
                    raise ValueError(f"Checkpoint 与当前配置不一致: {k}={meta[k]} != {v}")
            elif k == "input_mat_filename":
                meta_val = os.path.basename(str(meta[k]).strip())
                expected_val = os.path.basename(str(v).strip())
                if meta_val != expected_val:
                    raise ValueError(f"Checkpoint 与当前配置不一致: {k}={meta_val} != {expected_val}")
            else:
                if str(meta[k]).strip() != str(v).strip():
                    raise ValueError(f"Checkpoint 与当前配置不一致: {k}={meta[k]} != {v}")

    NumTrial = int(_scalar("NumTrial", 0))
    timestamp = _string("starting_timestamp", "")

    A_arr = data.get("A_opt_results", np.empty((0, K, N)))
    P_arr = data.get("P_opt_results", np.empty((0, 2 * (K if False else 1))))
    X_arr = data.get("X_list", np.empty((0,)))
    peak_arr = data.get("peak_opt_list", np.empty((0, 1)))
    best_step_arr = data.get("best_step_list", np.empty((0, 1)))
    time_arr = data.get("time_per_problem_list", np.empty((0, 1)))

    A_opt_results_list = [A_arr[i] for i in range(A_arr.shape[0])] if isinstance(A_arr, np.ndarray) and A_arr.ndim >= 3 else []
    P_opt_results_list = [P_arr[i] for i in range(P_arr.shape[0])] if isinstance(P_arr, np.ndarray) and P_arr.ndim >= 2 else []
    X_list_list = [X_arr[i] for i in range(X_arr.shape[0])] if isinstance(X_arr, np.ndarray) and X_arr.ndim >= 2 else []

    peak_opt_list = [float(x) for x in np.array(peak_arr).reshape(-1)]
    best_step_list = [int(x) for x in np.array(best_step_arr).reshape(-1)]
    time_per_problem_list = [float(x) for x in np.array(time_arr).reshape(-1)]

    state = {
        "start_idx": NumTrial,
        "timestamp": timestamp,
        "A_opt_results_list": A_opt_results_list,
        "P_opt_results_list": P_opt_results_list,
        "X_list_list": X_list_list,
        "peak_opt_list": peak_opt_list,
        "best_step_list": best_step_list,
        "time_per_problem_list": time_per_problem_list,
    }

    print(f"**Loaded checkpoint: {checkpoint_filename} (NumTrial={NumTrial})")
    return True, state


def save_final_mat(
    out_name,
    A_opt_results_list,
    P_opt_results_list,
    X_list_list,
    peak_opt_list,
    best_step_list,
    mod_order,
    N,
    K,
    L,
):
    NumDone = min(
        len(peak_opt_list),
        len(A_opt_results_list),
        len(P_opt_results_list),
        len(X_list_list),
        len(best_step_list),
    )

    if NumDone == 0:
        raise ValueError("No results to save in final MAT.")

    A_opt_results = np.stack(A_opt_results_list[:NumDone], axis=0).astype(np.float64)
    P_opt_results = np.stack(P_opt_results_list[:NumDone], axis=0).astype(np.float64)
    X_list = np.stack(X_list_list[:NumDone], axis=0).astype(np.complex128)
    peak_opt_arr = np.array(peak_opt_list[:NumDone], dtype=np.float64).reshape(-1, 1)
    best_step_arr = np.array(best_step_list[:NumDone], dtype=np.int32).reshape(-1, 1)
    qam_constellation = np.array(generate_qam_constellation(mod_order)).reshape(-1, 1).astype(np.complex128)

    mdict = {
        "L": np.array([[float(L)]]),
        "N": np.array([[int(N)]]),
        "K": np.array([[int(K)]]),
        "NumTrial": np.array([[int(NumDone)]]),
        "A_opt_results": A_opt_results,
        "P_opt_results": P_opt_results,
        "X_list": X_list,
        "peak_opt_list": peak_opt_arr,
        "best_step_list": best_step_arr,
        "qam_constellation": qam_constellation,
    }

    scipy.io.savemat(out_name, mdict, do_compression=True)
    print("**Final MAT saved:", out_name)