import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# 1. 读取Excel数据
file_path = "../perovskite_dataset/feature_data.xlsx"
df = pd.read_excel(file_path, header=0)

# 2. 提取14个重要特征
all_columns = df.columns.tolist()
cut_index = 74
selected_features = all_columns[1:cut_index]

# 3. 数据清洗
df_selected = df[selected_features].apply(pd.to_numeric, errors='coerce').dropna()

# 4. 计算Spearman相关系数矩阵
corr_matrix = df_selected.corr(method='spearman').abs()

# 5. 特征简化名称
feature_labels = {
    "bandgap": r"$E_g$",
    "A_r_atom": r"$r_A^{\mathrm{atom}}$",
    "B_r_atom": r"$r_B^{\mathrm{atom}}$",
    "X_r_atom": r"$r_X^{\mathrm{atom}}$",
    "A_r_ion": r"$r_A^{\mathrm{ion}}$",
    "B_r_ion": r"$r_B^{\mathrm{ion}}$",
    "X_r_ion": r"$r_X^{\mathrm{ion}}$",
    "A_ie": r"$IE_A$",
    "B_ie": r"$IE_B$",
    "X_ie": r"$IE_X$",
    "A_ea": r"$EA_A$",
    "B_ea": r"$EA_B$",
    "X_ea": r"$EA_X$",
    "A_x": r"$\chi_A$",
    "B_x": r"$\chi_B$",
    "X_x": r"$\chi_X$",
    "A_N": r"$N_A$",
    "B_N": r"$N_B$",
    "X_N": r"$N_X$",
    "A_M": r"$M_A$",
    "B_M": r"$M_B$",
    "X_M": r"$M_X$",
    "HOMO": r"$E_{\mathrm{HOMO}}$",
    "LUMO": r"$E_{\mathrm{LUMO}}$",
    "gap_AO": r"$\Delta E_{\mathrm{AO}}$",
    "bond_mean": r"$\mu_{\mathrm{bond}}$",
    "bond_var": r"$\sigma^2_{\mathrm{bond}}$",
    "bond_cv": r"$CV_{\mathrm{bond}}$",
    "delta_X_bx_bond_mean": r"$\Delta \mu_{X-B-X,\mathrm{bond}}$",
    "delta_X_bx_bond_var": r"$\Delta \sigma^2_{X-B-X,\mathrm{bond}}$",
    "ratio_X_bx_bond_mean": r"$\rho_{X-B-X,\mathrm{bond}}$",
    "ratio_X_bx_bond_var": r"$\sigma^2_{\rho,X-B-X,\mathrm{bond}}$",
    "delta_ea_bx_bond_mean": r"$\Delta \mu_{EA-B-X,\mathrm{bond}}$",
    "delta_ea_bx_bond_var": r"$\Delta \sigma^2_{EA-B-X,\mathrm{bond}}$",
    "ratio_ea_bx_bond_mean": r"$\rho_{EA-B-X,\mathrm{bond}}$",
    "ratio_ea_bx_bond_var": r"$\sigma^2_{\rho,EA-B-X,\mathrm{bond}}$",
    "delta_X_bx_bond_cv": r"$\Delta CV_{X-B-X,\mathrm{bond}}$",
    "ratio_X_bx_bond_cv": r"$CV_{\rho,X-B-X,\mathrm{bond}}$",
    "delta_ea_bx_bond_cv": r"$\Delta CV_{EA-B-X,\mathrm{bond}}$",
    "ratio_ea_bx_bond_cv": r"$CV_{\rho,EA-B-X,\mathrm{bond}}$",
    "(B_x - X_x)": r"$(\chi_B - \chi_X)$",
    "(B_x / X_x)": r"$\chi_B / \chi_X$",
    "(B_x - X_x)/(B_r_ion + X_r_ion)": r"$\frac{\chi_B - \chi_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    "(B_x / X_x)/(B_r_ion + X_r_ion)": r"$\frac{\chi_B / \chi_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    "(B_x/B_r_ion)": r"$\chi_B / r_B^{\mathrm{ion}}$",
    "(X_x/X_r_ion)": r"$\chi_X / r_X^{\mathrm{ion}}$",
    "(B_x/B_r_ion)+(X_x/X_r_ion)": r"$\frac{\chi_B}{r_B^{\mathrm{ion}}} + \frac{\chi_X}{r_X^{\mathrm{ion}}}$",
    "(B_x/B_r_ion)-(X_x/X_r_ion)": r"$\frac{\chi_B}{r_B^{\mathrm{ion}}} - \frac{\chi_X}{r_X^{\mathrm{ion}}}$",
    "(B_x/B_r_ion)*(X_x/X_r_ion)": r"$\frac{\chi_B}{r_B^{\mathrm{ion}}} \cdot \frac{\chi_X}{r_X^{\mathrm{ion}}}$",
    "(B_x/B_r_ion)/(X_x/X_r_ion)": r"$\frac{\chi_B / r_B^{\mathrm{ion}}}{\chi_X / r_X^{\mathrm{ion}}}$",
    "(B_ea - X_ea)": r"$(EA_B - EA_X)$",
    "(B_ea / X_ea)": r"$EA_B / EA_X$",
    "(B_ea - X_ea)/(B_r_ion + X_r_ion)": r"$\frac{EA_B - EA_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    "(B_ea / X_ea)/(B_r_ion + X_r_ion)": r"$\frac{EA_B / EA_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    "(B_ea/B_r_ion)": r"$EA_B / r_B^{\mathrm{ion}}$",
    "(X_ea/X_r_ion)": r"$EA_X / r_X^{\mathrm{ion}}$",
    "(B_ea/B_r_ion)+(X_ea/X_r_ion)": r"$\frac{EA_B}{r_B^{\mathrm{ion}}} + \frac{EA_X}{r_X^{\mathrm{ion}}}$",
    "(B_ea/B_r_ion)-(X_ea/X_r_ion)": r"$\frac{EA_B}{r_B^{\mathrm{ion}}} - \frac{EA_X}{r_X^{\mathrm{ion}}}$",
    "(B_ea/B_r_ion)*(X_ea/X_r_ion)": r"$\frac{EA_B}{r_B^{\mathrm{ion}}} \cdot \frac{EA_X}{r_X^{\mathrm{ion}}}$",
    "(B_ea/B_r_ion)/(X_ea/X_r_ion)": r"$\frac{EA_B / r_B^{\mathrm{ion}}}{EA_X / r_X^{\mathrm{ion}}}$",
    "(B_r_ion + X_r_ion)": r"$r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}$",
    "(B_r_ion / X_r_ion)": r"$r_B^{\mathrm{ion}} / r_X^{\mathrm{ion}}$",
    "(A_r_ion + X_r_ion)/(1.414*(B_r_ion + X_r_ion))": r"$\frac{r_A^{\mathrm{ion}} + r_X^{\mathrm{ion}}}{1.414(r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}})}$",
    "log(bond_var)": r"$\log(\sigma^2_{\mathrm{bond}})$",
    "log(bond_cv)": r"$\log(CV_{\mathrm{bond}})$",
    "log(delta_X_bx_bond_var)": r"$\log(\Delta \sigma^2_{X-B-X,\mathrm{bond}})$",
    "log(delta_ea_bx_bond_var)": r"$\log(\Delta \sigma^2_{EA-B-X,\mathrm{bond}})$",
    "log(ratio_X_bx_bond_var)": r"$\log(\sigma^2_{\rho,X-B-X,\mathrm{bond}})$",
    "log(ratio_ea_bx_bond_var)": r"$\log(\sigma^2_{\rho,EA-B-X,\mathrm{bond}})$",
    "log(delta_X_bx_bond_cv)": r"$\log(\Delta CV_{X-B-X,\mathrm{bond}})$",
    "log(ratio_X_bx_bond_cv)": r"$\log(CV_{\rho,X-B-X,\mathrm{bond}})$",
    "log(delta_ea_bx_bond_cv)": r"$\log(\Delta CV_{EA-B-X,\mathrm{bond}})$",
    "log(ratio_ea_bx_bond_cv)": r"$\log(CV_{\rho,EA-B-X,\mathrm{bond}})$"
}
corr_matrix = corr_matrix.rename(columns=feature_labels, index=feature_labels)

# --- 按照与 Eadh 的相关性大小排序 ---
target_col = r"$E_g$"
corr_with_target = corr_matrix[target_col].sort_values(ascending=False)
ordered_labels = corr_with_target.index.tolist()
corr_matrix = corr_matrix.loc[ordered_labels, ordered_labels]

# 转换为 numpy 数组用于 imshow
corr_values = corr_matrix.values
labels = corr_matrix.columns.tolist()

# 6. 自定义 colormap + norm
bounds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
colors = ["#3654A5", "#0070B3", "#1F9FD4", "#35B79C", "#C0BEC0",
          "#E4E0E4", "#F7D635", "#F4A153", "#F48E98", "#EB3A4B"]
cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, cmap.N)

# 7. 绘图（无色条，最大化铺满）
fig, ax = plt.subplots(figsize=(10, 9))  # 图像尺寸增大
im = ax.imshow(corr_values, cmap=cmap, norm=norm)

# 设置坐标轴标签
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=26)
ax.set_yticklabels(labels, fontsize=26)
ax.invert_yaxis()

# 添加网格线
for edge, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_color("black")
ax.set_xticks(np.arange(len(labels)+1)-0.5, minor=True)
ax.set_yticks(np.arange(len(labels)+1)-0.5, minor=True)
ax.grid(which="minor", color="white", linewidth=1.0)
ax.tick_params(which="both", bottom=False, left=False)

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("spearman.png", dpi=300, bbox_inches="tight")
# plt.show()

corr_with_target = corr_matrix[target_col].sort_values(ascending=False)

# 去掉STY自身
corr_with_target = corr_with_target.drop(target_col)

# 按相关性排序后的特征
sorted_features = corr_with_target.index.tolist()

custom_order = [
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

# GA排序（按照STY相关性，但输出顺序按custom_order）
rank_array = np.empty(len(custom_order), dtype=int)

for i, feat in enumerate(custom_order):
    rank_array[i] = sorted_features.index(feat) + 1

print("\nGA排序数组：")
print(rank_array.tolist())

ga_df = pd.DataFrame({
    "Feature": custom_order,
    "GA_rank": rank_array
})

ga_df.to_excel("GA_ranking_Spearman.xlsx", index=False)

print("GA排序已保存到 GA_ranking_Spearman.xlsx")

from scipy.stats import rankdata
# -------- 提取 X 和 y --------
X = df_selected.drop(columns=["bandgap"]).values
y = df_selected["bandgap"].values
feature_names = list(df_selected.drop(columns=["bandgap"]).columns)
# 对每一列特征做秩转换
X_ranked = np.apply_along_axis(rankdata, 0, X)
y_ranked = rankdata(y)

# 计算 Spearman 绝对相关性
pearson_scores = np.abs(np.corrcoef(X_ranked.T, y_ranked, rowvar=True)[-1, :-1])

# -------- 升序排序（最不重要 → 最重要） --------
sorted_idx_asc = np.argsort(pearson_scores)
ranking_array = sorted_idx_asc.tolist()
print("\nSpearman 特征重要性排序数组（最不重要 → 最重要）:")
print(ranking_array)

# -------- GA 排序（1 最重要） --------
rank_array = np.empty(len(pearson_scores), dtype=int)
rank_array[np.argsort(pearson_scores)[::-1]] = np.arange(1, len(pearson_scores)+1)
print("\nSpearman GA排序（1 最重要）:")
print(rank_array.tolist())

# -------- 对照特征名称 --------
feature_ordered = [feature_names[i] for i in sorted_idx_asc]
print("\n对应特征名称（最不重要 → 最重要）:")
print(feature_ordered)
