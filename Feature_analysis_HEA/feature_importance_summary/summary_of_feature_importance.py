import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

# ------------------------
# 1. 特征名
# ------------------------
feature_names = [
    # r"$\Delta H_{\mathrm{sss}}$",
    r"$E$",
    r"$K$",
    r"$G$",
    r"$\nu$",
    r"$R_m$",
    r"$R_i$",
    r"$R_c$",
    r"$V_m$",
    r"$H_s$",
    r"$H_c$",
    r"$\mathrm{VEC}$",
    r"$e/a$",
    r"$E_w$",
    r"$\chi_p$",
    r"$\chi_a$",
    r"$\chi_m$",
    r"$\chi_r$",
    r"$E_c$",
    r"$\delta E$",
    r"$\delta K$",
    r"$\delta G$",
    r"$\delta \nu$",
    r"$\delta R_m$",
    r"$\delta R_i$",
    r"$\delta R_c$",
    r"$\delta V_m$",
    r"$\delta H_s$",
    r"$\delta H_c$",
    r"$\delta \mathrm{VEC}$",
    r"$\delta (e/a)$",
    r"$\delta E_w$",
    r"$\delta \chi_p$",
    r"$\delta \chi_a$",
    r"$\delta \chi_m$",
    r"$\delta \chi_r$",
    r"$\delta E_c$",
    r"$\Delta E$",
    r"$\Delta K$",
    r"$\Delta G$",
    r"$\Delta \nu$",
    r"$\Delta R_m$",
    r"$\Delta R_i$",
    r"$\Delta R_c$",
    r"$\Delta V_m$",
    r"$\Delta H_s$",
    r"$\Delta H_c$",
    r"$\Delta \mathrm{VEC}$",
    r"$\Delta (e/a)$",
    r"$\Delta E_w$",
    r"$\Delta \chi_p$",
    r"$\Delta \chi_a$",
    r"$\Delta \chi_m$",
    r"$\Delta \chi_r$",
    r"$\Delta E_c$"
]

# ------------------------
# 2. 不同方法的特征排名 (示例数据)
# ------------------------
importance_data = {
    'Pearson correlation': [49, 44, 11, 23, 39, 26, 21, 1, 41, 33, 53, 51, 9, 31, 47, 5, 48, 50, 30, 45, 0, 35, 28, 46, 2, 15, 17, 6, 7, 40, 3, 13, 4, 14, 12, 29, 16, 8, 36, 27, 38, 52, 10, 32, 37, 22, 34, 42, 19, 43, 18, 24, 20, 25],
    'Spearman correlation': [39, 26, 9, 47, 49, 51, 11, 33, 21, 53, 28, 46, 44, 31, 50, 48, 5, 41, 23, 15, 30, 35, 40, 1, 17, 52, 0, 2, 29, 12, 3, 13, 34, 32, 8, 10, 22, 16, 45, 14, 43, 27, 4, 7, 6, 19, 24, 37, 42, 38, 25, 36, 20, 18],
    'MI': [47, 50, 11, 33, 49, 31, 32, 52, 29, 44, 35, 51, 40, 15, 22, 26, 39, 1, 48, 43, 5, 46, 21, 34, 53, 17, 41, 24, 9, 25, 42, 2, 45, 28, 3, 38, 30, 23, 12, 37, 19, 20, 14, 0, 18, 8, 6, 16, 4, 36, 27, 10, 7, 13],
    'RF-TreeSHAP': [49, 26, 37, 51, 15, 44, 33, 11, 5, 50, 31, 47, 48, 41, 40, 30, 27, 1, 17, 35, 53, 21, 52, 19, 43, 45, 23, 29, 32, 46, 22, 2, 42, 9, 25, 3, 8, 39, 24, 28, 12, 0, 34, 20, 6, 38, 13, 36, 18, 7, 16, 10, 14, 4],
    'RF-KernelSHAP': [5, 26, 51, 49, 41, 44, 47, 33, 48, 11, 40, 50, 17, 31, 27, 23, 37, 15, 1, 46, 45, 52, 21, 30, 32, 43, 2, 29, 22, 35, 19, 25, 53, 9, 42, 3, 24, 39, 0, 12, 8, 28, 34, 20, 6, 38, 13, 36, 18, 7, 16, 10, 4, 14],
    'RF-MDI': [33, 37, 26, 11, 49, 17, 41, 35, 15, 51, 50, 40, 48, 5, 31, 27, 47, 44, 52, 1, 30, 32, 53, 23, 43, 45, 42, 19, 21, 29, 22, 2, 25, 46, 9, 3, 8, 39, 34, 28, 12, 24, 0, 20, 13, 6, 38, 36, 16, 18, 14, 7, 10, 4],
    'RF-PFI': [19, 51, 22, 40, 53, 15, 43, 33, 41, 5, 37, 50, 11, 35, 49, 44, 26, 27, 46, 17, 45, 31, 48, 30, 25, 42, 32, 29, 1, 21, 52, 9, 47, 23, 24, 2, 28, 3, 20, 0, 39, 6, 38, 34, 36, 8, 12, 13, 18, 7, 10, 16, 4, 14],
    'GBRT-TreeSHAP': [15, 48, 11, 41, 24, 51, 5, 50, 33, 44, 19, 31, 30, 6, 49, 52, 2, 26, 37, 29, 3, 46, 27, 22, 17, 23, 9, 35, 43, 47, 40, 8, 53, 21, 45, 42, 25, 1, 12, 32, 28, 0, 7, 18, 10, 39, 38, 34, 36, 13, 4, 20, 16, 14],
    'GBRT-KernelSHAP': [11, 31, 30, 52, 48, 49, 41, 33, 51, 17, 5, 24, 26, 27, 15, 29, 6, 3, 50, 37, 23, 19, 44, 46, 53, 22, 43, 2, 45, 47, 9, 8, 35, 40, 25, 21, 42, 1, 12, 0, 7, 32, 28, 18, 10, 39, 36, 38, 34, 13, 4, 20, 16, 14],
    'GBRT-MDI': [15, 33, 51, 48, 19, 41, 11, 50, 24, 49, 5, 52, 31, 22, 3, 29, 6, 37, 17, 35, 46, 30, 44, 27, 23, 26, 40, 2, 9, 53, 25, 45, 21, 47, 8, 42, 43, 1, 32, 12, 18, 34, 28, 0, 39, 7, 38, 10, 13, 36, 4, 16, 20, 14],
    'GBRT-PFI': [6, 37, 31, 19, 24, 48, 50, 41, 17, 53, 3, 11, 35, 9, 49, 26, 40, 27, 43, 15, 44, 2, 33, 29, 46, 5, 51, 22, 23, 30, 52, 21, 42, 0, 25, 45, 28, 47, 1, 8, 12, 10, 18, 7, 32, 38, 36, 39, 34, 20, 4, 13, 16, 14],
    'XGBoost-TreeSHAP': [15, 41, 11, 44, 2, 50, 26, 29, 51, 33, 30, 24, 37, 19, 49, 46, 5, 31, 48, 52, 6, 43, 9, 40, 47, 53, 27, 21, 45, 3, 35, 42, 22, 17, 23, 1, 25, 7, 28, 36, 10, 32, 38, 8, 12, 0, 39, 34, 18, 20, 13, 16, 4, 14],
    'XGBoost-KernelSHAP': [2, 11, 15, 26, 24, 51, 41, 37, 5, 30, 6, 50, 33, 29, 31, 46, 49, 19, 52, 43, 44, 40, 47, 9, 3, 53, 45, 27, 48, 17, 22, 21, 42, 1, 35, 23, 7, 25, 10, 28, 36, 38, 32, 12, 8, 0, 34, 39, 18, 20, 13, 16, 4, 14],
    'XGBoost-MDI': [2, 1, 15, 33, 41, 49, 37, 52, 19, 11, 51, 5, 24, 9, 17, 26, 3, 30, 40, 47, 43, 0, 6, 22, 31, 48, 44, 50, 35, 29, 46, 27, 45, 53, 21, 8, 23, 42, 25, 34, 18, 32, 39, 7, 38, 28, 36, 10, 12, 20, 4, 13, 16, 14],
    'XGBoost-PFI': [9, 2, 41, 28, 48, 27, 19, 50, 44, 21, 49, 29, 26, 53, 15, 6, 52, 51, 33, 30, 31, 43, 5, 3, 35, 11, 47, 40, 37, 17, 25, 24, 42, 45, 38, 22, 46, 36, 10, 7, 23, 1, 34, 32, 8, 12, 18, 0, 39, 20, 13, 16, 4, 14],
    'LR-KernelSHAP': [47, 49, 45, 50, 31, 30, 29, 37, 32, 53, 38, 36, 46, 40, 35, 52, 48, 39, 27, 18, 19, 20, 51, 33, 34, 42, 44, 21, 22, 23, 43, 41, 26, 28, 24, 25, 11, 5, 12, 2, 9, 1, 15, 17, 8, 3, 0, 14, 16, 10, 6, 4, 13, 7],
    'LR-PFI': [49, 45, 47, 50, 30, 31, 32, 37, 53, 29, 40, 48, 36, 52, 46, 35, 38, 39, 19, 27, 20, 18, 34, 42, 51, 33, 21, 22, 44, 23, 43, 41, 26, 28, 24, 25, 11, 5, 12, 2, 9, 1, 17, 15, 3, 8, 0, 14, 16, 4, 6, 10, 13, 7],
    'LR-coefficient': [49, 45, 50, 47, 31, 30, 37, 32, 38, 29, 36, 53, 40, 48, 46, 52, 35, 39, 27, 18, 19, 20, 51, 33, 34, 42, 21, 44, 22, 23, 43, 41, 26, 28, 24, 25, 11, 5, 12, 2, 9, 15, 17, 1, 8, 3, 14, 0, 16, 10, 6, 4, 13, 7],
    'SVR-KernelSHAP': [13, 15, 9, 16, 49, 39, 5, 6, 29, 31, 25, 27, 48, 47, 51, 2, 23, 14, 7, 4, 12, 35, 36, 40, 50, 32, 0, 3, 53, 38, 30, 24, 20, 19, 18, 37, 45, 22, 34, 52, 21, 10, 42, 1, 41, 44, 11, 33,17, 26, 28, 8, 46, 43],
    'SVR-PFI': [47, 5, 39, 31, 9, 49, 13, 48, 23, 6, 25, 15, 35, 7, 16, 29, 12, 51, 14, 4, 27, 32, 3, 40, 2, 36, 24, 18, 20, 19, 37, 50, 53, 30, 0, 17, 8, 28, 38, 34, 26, 46, 22,  42, 41, 1, 11, 44, 33,43, 45, 52, 10, 21,],
    'MLP-KernelSHAP': [7, 6, 17, 4, 14, 5, 36, 3, 22, 33, 26, 53, 45, 23, 47, 16, 44, 51, 49, 11, 9, 41, 50, 48, 40, 37, 24, 1, 31, 46, 13, 38, 39, 35, 27, 2, 0, 34, 42, 32, 8, 19, 52, 28, 15, 18, 30, 25, 29, 10, 43, 21, 12, 20],
    'MLP-PFI': [46, 3, 23, 26, 22, 5, 11, 6, 48, 47, 41, 31, 21, 0, 28, 7, 4, 51, 2, 43, 17, 24, 53, 44, 42, 14, 38, 16, 36, 8, 39, 13, 50, 37, 49, 25, 33, 19, 10, 32, 52, 9, 35, 18, 34, 29, 40, 15, 20, 1, 27, 45, 12, 30],
    # 在四个公式中13 4 5 12 3排名均设置为最后一位，formula-Feature mean的8 10均设置为排名第7位 formula-Rank average的10,8均设置为第6  formula-Data average的10 8均设置为第8位
    # 'Formula-MVPD': [13, 4, 5, 12, 3, 7, 8, 10, 2, 6, 9, 0, 1, 11],
    # 'Formula-SGR': [13,4,5,12,3,7,2,10,8,6,9,0,1,11],
    # 'Formula-AGM': [13,4,5,12,3,10,8,2,7,6,9,0,1,11],
    # 'Formula-SHAP': [13,4,5,12,3,8,2,10,0,1,9,7,6,11],
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
fig, ax = plt.subplots(figsize=(20,12))

mesh = sns.heatmap(
    sorted_matrix,
    cmap=cmap_disc,
    norm=norm,
    cbar=False,
    xticklabels=sorted_feature_names,
    yticklabels=methods,
    ax=ax,
    linewidths=2,
    linecolor='white',
    annot=False
    # annot=True, fmt=".0f"
)

# 调整方块纵横比
ax.set_aspect(1)

# 美化字体
ax.set_xticklabels(sorted_feature_names, rotation=70, ha='center', fontsize=12, weight='bold')
ax.set_yticklabels(methods, rotation=0, ha='right', fontsize=13)  # 防止遮挡

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