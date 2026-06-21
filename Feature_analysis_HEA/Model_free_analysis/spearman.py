import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# 1. 读取Excel数据
file_path = "../HEA_dataset/feature_data.xlsx"
df = pd.read_excel(file_path, header=0)

# 2. 提取14个重要特征
all_columns = df.columns.tolist()
cut_index = 56
selected_features = all_columns[1:cut_index]

# 3. 数据清洗
df_selected = df[selected_features].apply(pd.to_numeric, errors='coerce').dropna()

# 4. 计算Spearman相关系数矩阵
corr_matrix = df_selected.corr(method='spearman').abs()

# 5. 特征简化名称
feature_labels = {
    "ΔHsss": r"$\Delta H_{\mathrm{sss}}$",
    "E": r"$E$",
    "K": r"$K$",
    "G": r"$G$",
    "ν": r"$\nu$",
    "Rm": r"$R_m$",
    "Ri": r"$R_i$",
    "Rc": r"$R_c$",
    "Vm": r"$V_m$",
    "Hs": r"$H_s$",
    "Hc": r"$H_c$",
    "VEC": r"$\mathrm{VEC}$",
    "e/a": r"$e/a$",
    "Ew": r"$E_w$",
    "Xp": r"$\chi_p$",
    "Xa": r"$\chi_a$",
    "Xm": r"$\chi_m$",
    "Xr": r"$\chi_r$",
    "Ec": r"$E_c$",
    "δE": r"$\delta E$",
    "δK": r"$\delta K$",
    "δG": r"$\delta G$",
    "δν": r"$\delta \nu$",
    "δRm": r"$\delta R_m$",
    "δRi": r"$\delta R_i$",
    "δRc": r"$\delta R_c$",
    "δVm": r"$\delta V_m$",
    "δHs": r"$\delta H_s$",
    "δHc": r"$\delta H_c$",
    "δVEC": r"$\delta \mathrm{VEC}$",
    "δe/a": r"$\delta (e/a)$",
    "δEw": r"$\delta E_w$",
    "δXp": r"$\delta \chi_p$",
    "δXa": r"$\delta \chi_a$",
    "δXm": r"$\delta \chi_m$",
    "δXr": r"$\delta \chi_r$",
    "δEc": r"$\delta E_c$",
    "ΔE": r"$\Delta E$",
    "ΔK": r"$\Delta K$",
    "ΔG": r"$\Delta G$",
    "Δν": r"$\Delta \nu$",
    "ΔRm": r"$\Delta R_m$",
    "ΔRi": r"$\Delta R_i$",
    "ΔRc": r"$\Delta R_c$",
    "ΔVm": r"$\Delta V_m$",
    "ΔHs": r"$\Delta H_s$",
    "ΔHc": r"$\Delta H_c$",
    "ΔVEC": r"$\Delta \mathrm{VEC}$",
    "Δe/a": r"$\Delta (e/a)$",
    "Δew": r"$\Delta E_w$",
    "ΔXp": r"$\Delta \chi_p$",
    "ΔXa": r"$\Delta \chi_a$",
    "ΔXm": r"$\Delta \chi_m$",
    "ΔXr": r"$\Delta \chi_r$",
    "ΔEc": r"$\Delta E_c$"
}

corr_matrix = corr_matrix.rename(columns=feature_labels, index=feature_labels)

# --- 按照与 Eadh 的相关性大小排序 ---
target_col = r"$\Delta H_{\mathrm{sss}}$"
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
X = df_selected.drop(columns=["ΔHsss"]).values
y = df_selected["ΔHsss"].values
feature_names = list(df_selected.drop(columns=["ΔHsss"]).columns)
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