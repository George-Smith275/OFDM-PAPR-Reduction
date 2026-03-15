# 这里生成PAM constellation和QAM constellation，且定义随机选取PAM/QAM字符的函数
import jax
import jax.numpy as jnp
import os
from jax import random

def generate_pam_constellation(modulation_order):
    """
    生成 PAM 星座点

    参数:
    modulation_order: 调制阶数 (必须是偶数)

    返回:
    包含 PAM 星座点的数组
    """
    if modulation_order % 2 != 0:
        raise ValueError("调制阶数必须是偶数")

    # 生成对称的 PAM 星座点
    # 例如，对于 4-PAM: [-3, -1, 1, 3]
    # 对于 6-PAM: [-5, -3, -1, 1, 3, 5]
    pam_symbols = jnp.arange(-(modulation_order - 1), modulation_order, 2)
    return pam_symbols

# 这里的key应该要随机
def generate_random_pam(key, modulation_order, num_symbols):
    """
    生成随机 PAM 字符

    参数:
    key: JAX 随机密钥
    modulation_order: 调制阶数 (必须是偶数)
    num_symbols: 需要生成的符号数量

    返回:
    包含随机 PAM 字符的数组
    """
    # 生成 PAM 星座点
    pam_constellation = generate_pam_constellation(modulation_order)

    # 生成随机索引
    indices = random.randint(key, (num_symbols,), 0, modulation_order)

    # 通过索引获取 PAM 符号
    random_pam = pam_constellation[indices]
    random_pam = random_pam.reshape(-1, 1)

    return random_pam


def generate_qam_constellation(modulation_order):
    """
    生成 QAM 星座点

    参数:
    modulation_order: 调制阶数 (必须是平方数，如 4, 16, 64, 256)

    返回:
    包含 QAM 星座点的复数数组
    """
    # 检查调制阶数是否为平方数
    sqrt_order = jnp.sqrt(modulation_order)
    if sqrt_order % 1 != 0:
        raise ValueError("QAM 调制阶数必须是平方数 (如 4, 16, 64, 256)")

    sqrt_order = int(sqrt_order)

    # 生成 PAM 星座点 (I 和 Q 分量)
    pam_constellation = jnp.arange(-(sqrt_order - 1), sqrt_order, 2)

    # 创建网格以生成所有可能的组合
    i_components, q_components = jnp.meshgrid(pam_constellation, pam_constellation)

    # 将 I 和 Q 分量组合成复数
    qam_constellation = i_components + 1j * q_components

    # 展平并返回星座点
    return qam_constellation.flatten()

def generate_random_qam(key, modulation_order, num_symbols):
    """
    生成随机 QAM 字符

    参数:
    key: JAX 随机密钥
    modulation_order: 调制阶数 (必须是平方数)
    num_symbols: 需要生成的符号数量

    返回:
    包含随机 QAM 字符的复数数组
    """
    # 生成 QAM 星座点
    qam_constellation = generate_qam_constellation(modulation_order)

    # 生成随机索引
    indices = random.randint(key, (num_symbols,), 0, modulation_order)

    # 通过索引获取 QAM 符号
    random_qam = qam_constellation[indices]
    random_qam = random_qam.reshape(-1, 1)

    return random_qam

"""
# Below are test codes
mod_order = 4 # modulation order
K = 8
seed = int.from_bytes(os.urandom(4), "big")  # 4字节随机数
key = jax.random.key(seed)
key, subkey = random.split(key)
random_pam = generate_random_pam(subkey, mod_order, K)
print(f"随机 {mod_order}-PAM 字符: {random_pam}")

key = jax.random.key(1)
key, subkey = random.split(key)
mod_order = 4 # modulation order
K = 8
random_qam = generate_random_qam(key, mod_order, K)
print(f"随机 {mod_order}-QAM 字符: {random_qam}")
"""