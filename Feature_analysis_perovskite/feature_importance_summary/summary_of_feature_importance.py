import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

# ------------------------
# 1. 特征名
# ------------------------
feature_names = [
    r"$r_A^{\mathrm{atom}}$",
    r"$r_B^{\mathrm{atom}}$",
    r"$r_X^{\mathrm{atom}}$",
    r"$r_A^{\mathrm{ion}}$",
    r"$r_B^{\mathrm{ion}}$",
    r"$r_X^{\mathrm{ion}}$",
    r"$IE_A$",
    r"$IE_B$",
    r"$IE_X$",
    r"$EA_A$",
    r"$EA_B$",
    r"$EA_X$",
    r"$\chi_A$",
    r"$\chi_B$",
    r"$\chi_X$",
    r"$N_A$",
    r"$N_B$",
    r"$N_X$",
    r"$M_A$",
    r"$M_B$",
    r"$M_X$",
    r"$E_{\mathrm{HOMO}}$",
    r"$E_{\mathrm{LUMO}}$",
    r"$\Delta E_{\mathrm{AO}}$",
    r"$\mu_{\mathrm{bond}}$",
    r"$\sigma^2_{\mathrm{bond}}$",
    r"$CV_{\mathrm{bond}}$",
    r"$\Delta \mu_{X-B-X,\mathrm{bond}}$",
    r"$\Delta \sigma^2_{X-B-X,\mathrm{bond}}$",
    r"$\rho_{X-B-X,\mathrm{bond}}$",
    r"$\sigma^2_{\rho,X-B-X,\mathrm{bond}}$",
    r"$\Delta \mu_{EA-B-X,\mathrm{bond}}$",
    r"$\Delta \sigma^2_{EA-B-X,\mathrm{bond}}$",
    r"$\rho_{EA-B-X,\mathrm{bond}}$",
    r"$\sigma^2_{\rho,EA-B-X,\mathrm{bond}}$",
    r"$\Delta CV_{X-B-X,\mathrm{bond}}$",
    r"$CV_{\rho,X-B-X,\mathrm{bond}}$",
    r"$\Delta CV_{EA-B-X,\mathrm{bond}}$",
    r"$CV_{\rho,EA-B-X,\mathrm{bond}}$",
    r"$(\chi_B - \chi_X)$",
    r"$\chi_B / \chi_X$",
    r"$\frac{\chi_B - \chi_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    r"$\frac{\chi_B / \chi_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    r"$\chi_B / r_B^{\mathrm{ion}}$",
    r"$\chi_X / r_X^{\mathrm{ion}}$",
    r"$\frac{\chi_B}{r_B^{\mathrm{ion}}} + \frac{\chi_X}{r_X^{\mathrm{ion}}}$",
    r"$\frac{\chi_B}{r_B^{\mathrm{ion}}} - \frac{\chi_X}{r_X^{\mathrm{ion}}}$",
    r"$\frac{\chi_B}{r_B^{\mathrm{ion}}} \cdot \frac{\chi_X}{r_X^{\mathrm{ion}}}$",
    r"$\frac{\chi_B / r_B^{\mathrm{ion}}}{\chi_X / r_X^{\mathrm{ion}}}$",
    r"$(EA_B - EA_X)$",
    r"$EA_B / EA_X$",
    r"$\frac{EA_B - EA_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    r"$\frac{EA_B / EA_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    r"$EA_B / r_B^{\mathrm{ion}}$",
    r"$EA_X / r_X^{\mathrm{ion}}$",
    r"$\frac{EA_B}{r_B^{\mathrm{ion}}} + \frac{EA_X}{r_X^{\mathrm{ion}}}$",
    r"$\frac{EA_B}{r_B^{\mathrm{ion}}} - \frac{EA_X}{r_X^{\mathrm{ion}}}$",
    r"$\frac{EA_B}{r_B^{\mathrm{ion}}} \cdot \frac{EA_X}{r_X^{\mathrm{ion}}}$",
    r"$\frac{EA_B / r_B^{\mathrm{ion}}}{EA_X / r_X^{\mathrm{ion}}}$",
    r"$r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}$",
    r"$r_B^{\mathrm{ion}} / r_X^{\mathrm{ion}}$",
    r"$\frac{r_A^{\mathrm{ion}} + r_X^{\mathrm{ion}}}{1.414(r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}})}$",
    r"$\log(\sigma^2_{\mathrm{bond}})$",
    r"$\log(CV_{\mathrm{bond}})$",
    r"$\log(\Delta \sigma^2_{X-B-X,\mathrm{bond}})$",
    r"$\log(\Delta \sigma^2_{EA-B-X,\mathrm{bond}})$",
    r"$\log(\sigma^2_{\rho,X-B-X,\mathrm{bond}})$",
    r"$\log(\sigma^2_{\rho,EA-B-X,\mathrm{bond}})$",
    r"$\log(\Delta CV_{X-B-X,\mathrm{bond}})$",
    r"$\log(CV_{\rho,X-B-X,\mathrm{bond}})$",
    r"$\log(\Delta CV_{EA-B-X,\mathrm{bond}})$",
    r"$\log(CV_{\rho,EA-B-X,\mathrm{bond}})$"
]

# ------------------------
# 2. 不同方法的特征排名 (示例数据)
# ------------------------
importance_data = {
    'Pearson correlation': [32, 12, 6, 0, 3, 18, 15, 9, 64, 37, 24, 63, 62, 65, 55, 68, 70, 25, 26, 28, 69, 66, 71, 35, 51, 11, 67, 45, 38, 36, 8, 14, 5, 2, 54, 20, 17, 44, 21, 50, 59, 10, 34, 19, 58, 16, 22, 30, 31, 52, 49, 53, 57, 23, 56, 61, 47, 4, 1, 7, 60, 41, 46, 27, 33, 48, 13, 43, 39, 42, 40, 29],
    'Spearman correlation': [12, 32, 65, 24, 9, 15, 18, 6, 3, 0, 64, 28, 63, 26, 25, 62, 55, 70, 37, 35, 68, 8, 14, 51, 36, 69, 2, 45, 58, 11, 53, 50, 21, 38, 71, 44, 10, 19, 16, 17, 20, 5, 54, 52, 30, 66, 57, 59, 49, 31, 67, 34, 23, 61, 56, 22, 7, 47, 1, 4, 60, 41, 27, 46, 13, 33, 42, 29, 43, 39, 40, 48],
    'MI': [0, 3, 6, 9, 12, 15, 18, 26, 25, 32, 37, 21, 36, 62, 63, 38, 65, 70, 30, 69, 71, 66, 34, 61, 64, 68, 28, 67, 35, 54, 11, 51, 2, 24, 14, 44, 5, 17, 8, 60, 20, 45, 59, 55, 1, 22, 46, 56, 43, 4, 7, 13, 53, 19, 16, 10, 23, 48, 27, 41, 50, 47, 49, 42, 31, 57, 33, 39, 29, 52, 58, 40],
    'RF-TreeSHAP': [37, 70, 62, 32, 25, 65, 26, 63, 12, 15, 6, 9, 3, 18, 0, 21, 71, 38, 68, 35, 14, 34, 10, 8, 67, 64, 28, 69, 30, 66, 36, 2, 11, 53, 61, 16, 19, 45, 23, 24, 47, 57, 59, 54, 17, 1, 44, 5, 51, 7, 20, 49, 50, 58, 52, 27, 22, 4, 60, 41, 31, 46, 56, 55, 33, 39, 48, 42, 43, 29, 13, 40],
    'RF-KernelSHAP': [21, 62, 19, 25, 9, 6, 28, 11, 34, 32, 71, 8, 35, 10, 15, 66, 36, 16, 63, 45, 61, 38, 14, 59, 64, 37, 2, 30, 67, 24, 69, 68, 12, 23, 53, 70, 3, 1, 0, 17, 18, 51, 20, 47, 26, 65, 5, 44, 7, 54, 57, 49, 52, 4, 60, 50, 22, 46, 56, 27, 58, 31, 55, 41, 39, 33, 48, 42, 43, 29, 13, 40],
    'RF-MDI': [12, 18, 6, 15, 9, 3, 0, 37, 32, 25, 70, 26, 65, 63, 62, 21, 38, 71, 35, 68, 28, 34, 36, 69, 30, 64, 14, 8, 66, 11, 67, 10, 61, 2, 19, 16, 45, 59, 53, 54, 24, 17, 44, 1, 49, 51, 5, 57, 23, 60, 20, 47, 52, 56, 31, 7, 46, 58, 55, 27, 50, 41, 4, 22, 33, 48, 39, 42, 43, 29, 13, 40],
    'RF-PFI': [37, 25, 26, 32, 70, 63, 21, 65, 34, 62, 14, 68, 8, 35, 71, 67, 15, 53, 12, 2, 38, 9, 18, 3, 6, 10, 11, 0, 64, 45, 28, 44, 57, 16, 54, 66, 19, 69, 30, 17, 36, 23, 5, 7, 20, 1, 24, 59, 47, 22, 52, 51, 61, 50, 49, 4, 46, 60, 27, 56, 58, 31, 41, 55, 39, 48, 33, 42, 43, 29, 13, 40],
    'GBRT-TreeSHAP': [10, 53, 21, 4, 43, 2, 22, 8, 14, 12, 5, 54, 35, 0, 70, 3, 17, 37, 18, 6, 15, 63, 20, 65, 68, 13, 19, 25, 57, 9, 32, 62, 67, 28, 16, 26, 71, 38, 34, 23, 50, 45, 44, 64, 59, 30, 66, 52, 69, 7, 36, 47, 1, 11, 39, 27, 51, 46, 24, 31, 58, 29, 60, 56, 49, 42, 61, 41, 48, 55, 33, 40],
    'GBRT-KernelSHAP': [0, 2, 3, 4, 5, 14, 10, 8, 15, 21, 20, 53, 68, 12, 17, 32, 6, 35, 25, 37, 19, 22, 67, 18, 54, 43, 70, 65, 63, 34, 9, 62, 28, 57, 13, 38, 64, 16, 26, 71, 45, 59, 23, 66, 44, 50, 30, 52, 69, 47, 36, 7, 11, 1, 39, 27, 24, 51, 31, 46, 58, 29, 56, 60, 49, 61, 42, 41, 55, 48, 33, 40],
    'GBRT-MDI': [10, 53, 4, 21, 2, 12, 22, 8, 18, 14, 37, 54, 0, 32, 15, 70, 43, 25, 63, 6, 3, 65, 9, 68, 35, 17, 5, 20, 13, 57, 19, 62, 26, 30, 67, 38, 28, 36, 64, 71, 34, 16, 7, 23, 66, 44, 69, 45, 50, 11, 1, 39, 52, 59, 46, 58, 60, 51, 42, 47, 27, 31, 29, 24, 49, 56, 61, 48, 33, 41, 55, 40],
    'GBRT-PFI': [10, 14, 53, 62, 21, 20, 63, 43, 2, 17, 8, 65, 37, 54, 4, 68, 19, 28, 26, 71, 32, 25, 16, 3, 5, 44, 0, 35, 70, 57, 67, 6, 64, 9, 38, 12, 22, 34, 15, 13, 18, 45, 59, 50, 11, 30, 39, 66, 52, 23, 69, 36, 1, 47, 7, 46, 51, 31, 27, 56, 42, 24, 58, 60, 49, 29, 61, 41, 48, 55, 33, 40],
    'XGBoost-TreeSHAP': [6, 14, 15, 9, 12, 22, 18, 19, 63, 54, 53, 65, 68, 70, 43, 69, 3, 62, 67, 71, 10, 21, 66, 20, 4, 8, 37, 57, 35, 64, 26, 59, 45, 16, 32, 44, 38, 17, 23, 34, 50, 28, 2, 52, 5, 25, 11, 0, 13, 47, 51, 36, 56, 24, 30, 27, 60, 39, 58, 1, 7, 31, 46, 61, 42, 29, 49, 41, 55, 48, 33, 40],
    'XGBoost-KernelSHAP': [3, 4, 6, 15, 12, 14, 10, 9, 18, 22, 19, 62, 54, 53, 43, 37, 63, 67, 69, 70, 68, 65, 66, 26, 8, 35, 71, 20, 16, 45, 59, 32, 57, 38, 64, 21, 44, 17, 34, 50, 25, 2, 28, 5, 23, 11, 52, 0, 47, 36, 51, 13, 24, 56, 30, 7, 39, 27, 60, 58, 1, 31, 46, 61, 42, 29, 49, 41, 55, 48, 33, 40],
    'XGBoost-MDI': [6, 14, 15, 9, 12, 18, 22, 53, 54, 70, 43, 19, 68, 69, 3, 37, 63, 66, 67, 65, 32, 57, 0, 8, 10, 21, 36, 25, 71, 26, 11, 30, 2, 34, 62, 38, 61, 23, 4, 17, 45, 28, 35, 50, 39, 20, 52, 42, 27, 24, 16, 5, 7, 59, 1, 31, 29, 64, 58, 46, 56, 13, 49, 33, 51, 60, 47, 44, 48, 41, 55, 40],
    'XGBoost-PFI': [59, 4, 37, 67, 43, 12, 15, 6, 22, 14, 63, 53, 54, 19, 18, 9, 68, 70, 65, 3, 69, 64, 62, 66, 10, 71, 20, 8, 35, 26, 57, 44, 45, 21, 32, 17, 38, 16, 5, 11, 34, 39, 52, 23, 50, 28, 51, 25, 56, 36, 2, 13, 60, 30, 47, 46, 1, 31, 58, 0, 42, 27, 7, 29, 49, 24, 61, 41, 55, 48, 33, 40],
    'LR-KernelSHAP': [38, 34, 32, 37, 48, 25, 30, 36, 26, 28, 35, 24, 60, 61, 27, 51, 33, 57, 29, 31, 47, 58, 40, 42, 41, 52, 50, 71, 67, 0, 64, 68, 12, 9, 18, 15, 6, 69, 66, 3, 62, 63, 70, 65, 5, 59, 13, 21, 4, 19, 16, 23, 10, 1, 39, 49, 22, 14, 2, 8, 46, 11, 43, 7, 55, 54, 45, 44, 53, 56, 17, 20],
    'LR-PFI': [38, 34, 32, 37, 48, 25, 30, 28, 36, 35, 26, 24, 60, 61, 27, 33, 51, 29, 57, 31, 47, 58, 40, 42, 41, 52, 50, 64, 68, 71, 0, 67, 12, 9, 18, 15, 3, 69, 66, 6, 62, 63, 65, 70, 59, 5, 13, 4, 21, 19, 16, 23, 1, 10, 39, 49, 22, 14, 8, 2, 11, 46, 7, 43, 54, 55, 45, 44, 53, 56, 17, 20],
    'LR-coefficient': [38, 34, 32, 37, 48, 25, 30, 36, 28, 35, 26, 24, 60, 61, 27, 51, 33, 29, 57, 31, 47, 58, 40, 42, 41, 52, 50, 0, 71, 67, 64, 68, 12, 9, 18, 15, 6, 3, 69, 66, 62, 63, 65, 70, 59, 5, 13, 4, 21, 19, 16, 23, 1, 10, 39, 49, 22, 14, 8, 2, 46, 11, 43, 7, 55, 54, 45, 44, 53, 56, 17, 20],
    'SVR-KernelSHAP': [3, 46, 18, 64, 41, 54, 45, 69, 60, 55, 0, 52, 58, 15, 71, 66, 40, 67, 68, 25, 43, 20, 36, 56, 12, 9, 42, 5, 6, 17, 39, 53, 26, 32, 57, 37, 21, 59, 50, 44, 11, 34, 61, 51, 63, 4, 49, 62, 13, 14, 10, 65, 2, 70, 8, 48, 38, 24, 47, 30, 28, 27, 7, 23, 35, 29, 19, 16, 31, 22, 1, 33],
    'SVR-PFI': [0, 6, 3, 18, 15, 9, 44, 46, 12, 45, 42, 55, 54, 5, 58, 53, 64, 41, 43, 20, 37, 56, 68, 17, 52, 40, 32, 11, 50, 39, 26, 57, 60, 61, 36, 59, 2, 25, 66, 69, 14, 71, 8, 34, 51, 21, 48, 62, 49, 67, 63, 4, 10, 65, 70, 13, 27, 30, 47, 38, 7, 28, 23, 24, 35, 29, 19, 16, 31, 22, 1, 33],
    'MLP-KernelSHAP': [9, 12, 45, 18, 0, 6, 48, 37, 21, 11, 44, 66, 8, 60, 2, 55, 20, 71, 56, 3, 54, 10, 68, 58, 26, 64, 57, 53, 43, 25, 70, 5, 67, 13, 15, 65, 62, 42, 59, 63, 17, 69, 38, 46, 32, 40, 52, 28, 51, 30, 39, 61, 34, 35, 50, 4, 41, 36, 29, 47, 27, 49, 14, 7, 31, 23, 24, 19, 16, 33, 1, 22],
    'MLP-PFI': [9, 12, 6, 18, 0, 66, 20, 2, 11, 70, 37, 10, 45, 3, 53, 55, 64, 44, 56, 8, 62, 15, 5, 54, 25, 38, 65, 68, 71, 48, 43, 21, 17, 52, 69, 67, 63, 32, 57, 39, 59, 13, 60, 51, 26, 40, 42, 41, 58, 28, 4, 34, 61, 30, 50, 35, 46, 47, 36, 49, 14, 27, 7, 29, 23, 31, 1, 16, 19, 33, 24, 22]
}

methods = list(importance_data.keys())
n_methods = len(methods)
n_features = len(feature_names)

# ------------------------
# 3. 构造方法×特征的排名矩阵
# ------------------------
importance_matrix = np.full((n_methods, n_features), np.nan)

Formula_methods = ['Formula-MVPD', 'Formula-SGR', 'Formula-AGM','Formula-SHAP']
for i, (method, ranking) in enumerate(importance_data.items()):
    # print('methods:',method,'ranking:',ranking)
    if method in Formula_methods:
        for rank, feature_index in enumerate(ranking):
            # 特殊规则：13,4,5,12,3 → 最后一位（n_features-1）
            if feature_index in [13, 4, 5, 12, 3]:
                importance_matrix[i, feature_index] = 0
            # Feature MVPD 特殊：8,10 → 第7位
            elif method == 'Formula-MVPD' and feature_index in [8, 10]:
                importance_matrix[i, feature_index] = 7
            # Rank average 特殊：10,8 → 第6位
            elif method == 'Formula-SGR' and feature_index in [10, 8]:
                importance_matrix[i, feature_index] = 8
            # Data average 特殊：10,8 → 第8位
            elif method == 'Formula-AGM' and feature_index in [10, 8]:
                importance_matrix[i, feature_index] = 6
            else:
                importance_matrix[i, feature_index] = rank
    else:
        for rank, feature_index in enumerate(ranking):
            importance_matrix[i, feature_index] = rank
            # print('importance_matrix:',importance_matrix)


# ------------------------
# 4. 计算平均排名（越小越重要）并转换为相对排名（0~1）
# ------------------------
rel_importance = 1 - (importance_matrix / (n_features - 1))
print('rel:',rel_importance)
mean_rel_importance = np.mean(rel_importance, axis=0)

# x轴特征排序：最重要放左边
feature_order = np.argsort(-mean_rel_importance)[::-1]  # ← 这里反转排序顺序
sorted_feature_names = [feature_names[i] for i in feature_order]
sorted_matrix = importance_matrix[:, feature_order]

# ------------------------
# 5. 转换为 DataFrame
# ------------------------
df = pd.DataFrame(importance_matrix.T, index=feature_names, columns=methods)
df["Mean Relative Importance"] = mean_rel_importance
df_sorted = df.loc[sorted_feature_names]
print(df_sorted.round(2))

# ------------------------
# 6. 自定义颜色 (蓝色最重要 -> 红色最不重要)
# ------------------------
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# 原始颜色（你给的）
custom_colors = [
    "#1C5AA6", "#198CB9", "#35B79C", "#82B850", "#A3CA67", "#DBE466",
    "#FAE93B", "#FEEF90", "#FDD047", "#F8BB4B", "#F79356", "#F47D5A",
    "#F1695F", "#EA2C42"
]

# 1️⃣ 构造连续 colormap
base_cmap = LinearSegmentedColormap.from_list(
    "custom_smooth",
    custom_colors
)

# 2️⃣ 根据特征数量自动生成颜色
colors = base_cmap(np.linspace(0, 1, n_features))

# 3️⃣ 转为离散 colormap
cmap_disc = ListedColormap(colors)

# 4️⃣ 分段控制（关键）
bounds = np.arange(-0.5, n_features + 0.5, 1)
norm = BoundaryNorm(bounds, cmap_disc.N)
# custom_colors = [
#     "#1C5AA6", "#198CB9", "#35B79C", "#82B850", "#A3CA67", "#DBE466",
#     "#FAE93B", "#FEEF90", "#FDD047", "#F8BB4B", "#F79356", "#F47D5A",
#     "#F1695F", "#EA2C42"
# ]
# # 按平均相对重要性排序颜色
# sorted_colors = [custom_colors[i] for i in np.linspace(0, len(custom_colors)-1, n_features, dtype=int)]
# cmap_disc = ListedColormap(sorted_colors)
# bounds = np.arange(-0.5, n_features + 0.5, 1)
# norm = BoundaryNorm(bounds, cmap_disc.N)

# ------------------------
# 7. 绘图
# ------------------------
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(70, 60))

mesh = sns.heatmap(
    sorted_matrix,
    cmap=cmap_disc,
    norm=norm,
    cbar=False,
    xticklabels=sorted_feature_names,
    yticklabels=methods,
    ax=ax,
    linewidths=1,
    linecolor='white',
    annot=False
    # annot=True, fmt=".0f"
)

# 调整方块纵横比
ax.set_aspect(0.75)

# 美化字体
ax.set_xticklabels(sorted_feature_names, rotation=90, ha='center', fontsize=7, weight='bold')
ax.set_yticklabels(methods, rotation=0, ha='right', fontsize=8)  # 防止遮挡

# 自定义 colorbar
cbar = fig.colorbar(mesh.get_children()[0], ax=ax, boundaries=bounds, spacing='proportional', fraction=0.03, pad=0.02)
cbar.set_ticks([])
cbar.ax.text(0.5, 1.01, "high", ha='center', va='bottom', fontsize=12, weight='bold', transform=cbar.ax.transAxes)
cbar.ax.text(0.5, -0.02, "low", ha='center', va='top', fontsize=12, weight='bold', transform=cbar.ax.transAxes)
if hasattr(cbar, 'outline'):
    cbar.outline.set_visible(False)
for spine in cbar.ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()
fig.savefig("Feature_Importance_Heatmap.pdf", format='pdf', bbox_inches='tight')