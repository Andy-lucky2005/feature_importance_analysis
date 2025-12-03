import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

# ------------------------
# 1. 特征名
# ------------------------
feature_names = [
    r"$\chi_p^M$", r"$\chi_p^{M'}$", r"$IE^M$", r"$IE^{M'}$", r"$r^M$", r"$r^{M'}$",
    r"$\Delta H_f^{MO}$", r"$\Delta H_f^{M'O}$", r"$\Delta H_f^{M'M}$",
    r"$\Delta H_{sub}^M$", r"$\Delta H_{sub}^{M'}$", r"$\gamma^M$", r"$n_{ws}^M$", r"$E_g^{M'O}$"
]

# ------------------------
# 2. 不同方法的特征排名 (示例数据)
# ------------------------
importance_data = {
    'Pearson correlation': [13, 2, 10, 8, 7, 3, 5, 0, 6, 1, 4, 9, 12, 11],
    'Spearman correlation': [2, 13, 8, 10, 7, 5, 3, 1, 0, 6, 12, 4, 9, 11],
    'MI': [10, 3, 7, 5, 13, 1, 8, 2, 0, 4, 12, 6, 9, 11],
    'RF-TreeSHAP': [3, 10, 5, 2, 8, 0, 1, 4, 12, 7, 13, 6, 9, 11],
    'RF-KernelSHAP': [3, 10, 5, 2, 8, 0, 1, 4, 12, 7, 13, 6, 9, 11],
    'RF-MDI': [3, 10, 5, 13, 2, 1, 8, 0, 4, 6, 7, 12, 9, 11],
    'RF-PFI': [3, 5, 10, 8, 2, 4, 1, 12, 0, 13, 7, 6, 9, 11],
    'GBRT-TreeSHAP': [4, 3, 12, 10, 2, 5, 8, 0, 1, 13, 7, 6, 9, 11],
    'GBRT-KernelSHAP': [4, 3, 12, 10, 2, 5, 8, 0, 1, 13, 7, 6, 9, 11],
    'GBRT-MDI': [3, 4, 10, 12, 5, 13, 8, 2, 0, 1, 6, 7, 9, 11],
    'GBRT-PFI': [4,8,3,10,5,12,2,1,0,13,7,6,9,11],
    'XGBoost-TreeSHAP': [12, 3, 2, 10, 5, 8, 4, 0, 1, 7, 13, 6, 9, 11],
    'XGBoost-KernelSHAP': [4, 3, 10, 2, 12, 5, 8, 0, 1, 13, 7, 6, 9, 11],
    'XGBoost-MDI': [12, 13, 5, 3, 8, 2, 6, 1, 10, 7, 4, 0, 11, 9],
    'XGBoost-PFI': [3,12,10,5,2,8,4,1,13,7,0,6,9,11],
    'LR-KernelSHAP': [0, 3, 1, 13, 9, 2, 8, 10, 4, 5, 7, 6, 12, 11],
    'LR-PFI': [1,3,0,9,13,10,2,5,8,4,7,6,12,11],
    'LR-coefficient': [3,1,0,13,9,2,10,8,4,5,6,7,12,11],
    'SVR-KernelSHAP': [3, 2, 12, 10, 8, 1, 13, 5, 0, 7, 9, 4, 6, 11],
    'SVR-PFI': [2, 3, 12, 8, 13, 5, 10, 1, 0, 9, 7, 4, 6, 11],
    'MLP-KernelSHAP': [3, 8, 2, 10, 5, 4, 13, 12, 7, 0, 9, 1, 6, 11],
    'MLP-PFI': [3, 5, 8, 2, 10, 12, 13, 4, 0, 9, 1, 7, 6, 11],
    # 在四个公式中13 4 5 12 3排名均设置为最后一位，formula-Feature mean的8 10均设置为排名第7位 formula-Rank average的10,8均设置为第6  formula-Data average的10 8均设置为第8位
    'Formula-MVPD': [13, 4, 5, 12, 3, 7, 8, 10, 2, 6, 9, 0, 1, 11],
    'Formula-SGR': [13,4,5,12,3,7,2,10,8,6,9,0,1,11],
    'Formula-AGM': [13,4,5,12,3,10,8,2,7,6,9,0,1,11],
    'Formula-SHAP': [13,4,5,12,3,8,2,10,0,1,9,7,6,11],
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
custom_colors = [
    "#1C5AA6", "#198CB9", "#35B79C", "#82B850", "#A3CA67", "#DBE466",
    "#FAE93B", "#FEEF90", "#FDD047", "#F8BB4B", "#F79356", "#F47D5A",
    "#F1695F", "#EA2C42"
]
# 按平均相对重要性排序颜色
sorted_colors = [custom_colors[i] for i in np.linspace(0, len(custom_colors)-1, n_features, dtype=int)]
cmap_disc = ListedColormap(sorted_colors)
bounds = np.arange(-0.5, n_features + 0.5, 1)
norm = BoundaryNorm(bounds, cmap_disc.N)

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
ax.set_aspect(0.6)

# 美化字体
ax.set_xticklabels(sorted_feature_names, rotation=45, ha='center', fontsize=15, weight='bold')
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