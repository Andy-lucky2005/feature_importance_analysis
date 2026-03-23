import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon

# =========================================================
# === Step 0: 读取数据（保持特征为行，方法为列） ===
# =========================================================
file_path = "all_sort.xlsx"
df = pd.read_excel(file_path, header=0)
df = df.set_index(df.columns[0])  # 行=特征, 列=方法

print("数据形状：", df.shape)
print("行 = 特征数，列 = 方法数\n")


# =========================================================
# === H1: 跨方法一致性检验 (Kendall's W + 置换 + CI)
# =========================================================
def kendalls_w(rank_matrix):
    m, n = rank_matrix.shape  # m 方法数, n 特征数
    Rj = rank_matrix.sum(axis=0)
    # R_bar = np.mean(Rj)
    # S = np.sum((Rj - R_bar)**2)
    i = np.sum(Rj ** 2)
    a = 1/n * ((np.sum(Rj)) ** 2)
    W = 12 * (i - a) / (m**2 * (n**3 - n))
    return W

rank_matrix = df.values.T  # 行=方法, 列=特征
W_obs = kendalls_w(rank_matrix)

def permutation_test(rank_matrix, B=50000, seed=1412):
    rng = np.random.default_rng(seed)
    m, n = rank_matrix.shape
    W_perm = np.zeros(B)
    for b in range(B):
        perm = rank_matrix.copy()
        for i in range(m):
            rng.shuffle(perm[i])
        W_perm[b] = kendalls_w(perm)
    p = (np.sum(W_perm >= W_obs) + 1) / (B + 1)
    return p

p_H1 = permutation_test(rank_matrix)

def bootstrap_ci(rank_matrix, B=50000, seed=1412):
    rng = np.random.default_rng(seed)
    m = rank_matrix.shape[0]
    boot = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, m, m)
        boot[b] = kendalls_w(rank_matrix[idx])
    return np.percentile(boot, [2.5, 97.5])

CI_H1 = bootstrap_ci(rank_matrix)

print("=== H1 结果 ===")
print(f"Kendall's W = {W_obs:.4f}")
print(f"Permutation p-value = {p_H1:.5f}")
print(f"95% CI = [{CI_H1[0]:.4f}, {CI_H1[1]:.4f}]\n")