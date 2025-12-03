import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon

# =========================================================
# === Step 0: 读取数据（保持特征为行，方法为列） ===
# =========================================================
file_path = "method_sort.xlsx"
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


# =========================================================
# === H2: 线性 vs 非线性模型与公式 SHAP 一致性检验 ===
# =========================================================
# linear_methods = ['LR-KernelSHAP','LR-permutation','LR-coef']
# nonlinear_methods = [
#     'RF-TreeSHAP','GBRT-TreeSHAP','XGBoost-TreeSHAP',
#     'RF-KernelSHAP','GBRT-KernelSHAP','XGBoost-KernelSHAP',
#     'MLP-KernelSHAP','MLP-permutation',
#     'RF-permutation','GBRT-permutation','XGBoost-permutation',
#     'SVR-KernelSHAP','SVR-PFI',
#     'RF', 'GBRT', 'XGBoost'
# ]
#
# formula = df['Formula-SHAP'].values
#
# def compute_spearman(method_list):
#     return np.array([spearmanr(df[m].values, formula)[0] for m in method_list])
#
# rho_linear = compute_spearman(linear_methods)
# rho_nonlinear = compute_spearman(nonlinear_methods)
#
# stat, p_H2 = mannwhitneyu(rho_nonlinear, rho_linear, alternative='greater')
# effect = np.mean(rho_nonlinear) - np.mean(rho_linear)
#
# print("=== H2 结果 ===")
# print(f"平均 Spearman 差值 (非线性-线性) = {effect:.4f}")
# print(f"Mann-Whitney U 单侧检验 p-value = {p_H2:.5f}\n")


# =========================================================
# === H3: γM 排名稳健性检验 ===
# =========================================================
# gamma_row = df.loc['γ_M (J/m^2)'].values
# other_rows = df.drop(index='γ_M (J/m^2)').values
# second_best = np.min(other_rows, axis=0)
# diff = second_best - gamma_row  # 差值 > 0 表示 γM 更重要
# print(second_best)
# # print(diff)
#
# # Bootstrap 置信区间
# B = 5000
# rng = np.random.default_rng(1412)
# boot = np.zeros(B)
# for b in range(B):
#     idx = rng.integers(0, len(diff), len(diff))
#     boot[b] = np.mean(diff[idx])
#
# CI_H3 = np.percentile(boot, [2.5, 97.5])
#
# # 单侧有符号秩检验
# stat, p_H3 = wilcoxon(diff, alternative='greater')
#
# print("=== H3 结果 ===")
# print(f"平均差值 = {np.mean(diff):.4f}")
# print(f"95% CI = [{CI_H3[0]:.4f}, {CI_H3[1]:.4f}]")
# print(f"Wilcoxon 单侧检验 p-value = {p_H3:.8f}")

