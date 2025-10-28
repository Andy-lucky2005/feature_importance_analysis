import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# 1. 读取Excel数据
file_path = "../Feature_Value/Science_feature_data.xlsx"
df = pd.read_excel(file_path, header=0)

# 2. 提取14个重要特征
all_columns = df.columns.tolist()
cut_index = all_columns.index("Secondary features that have been screened out")
selected_features = all_columns[1:cut_index]

# 3. 数据清洗
df_selected = df[selected_features].apply(pd.to_numeric, errors='coerce').dropna()

# 4. 计算相关系数矩阵并取绝对值（Pearson）
corr_matrix = df_selected.corr(method='pearson').abs()

# 5. 特征简化名称
feature_labels = {
    "Eadh_Experimental_Data (J/m2)": r"$E_{\mathrm{adh}}$",
    "Xp_M": r"$\chi_p^M$",
    "Xp_M'": r"$\chi_p^{M'}$",
    "IE_M (eV)": r"$IE^M$",
    "IE_M' (eV)": r"$IE^{M'}$",
    "r_M (Å)": r"$r^M$",
    "r_M' (Å)": r"$r^{M'}$",
    "Hf_MO (eV M)": r"$\Delta H_f^{MO}$",
    "Hf_M'O (eV M')": r"$\Delta H_f^{M'O}$",
    "Hf_M'(M) (eV)": r"$\Delta H_f^{M'M}$",
    "Hsub_M (eV)": r"$\Delta H_{sub}^M$",
    "Hsub_M' (eV)": r"$\Delta H_{sub}^{M'}$",
    "γ_M (J/m^2)": r"$\gamma^M$",
    "Nws_M (d.u.)": r"$n_{ws}^M$",
    "Eg_M'O (eV)": r"$E_g^{M'O}$"
}
corr_matrix = corr_matrix.rename(columns=feature_labels, index=feature_labels)

# --- 按照与 Eadh 的相关性大小排序 ---
target_col = r"$E_{\mathrm{adh}}$"
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
fig, ax = plt.subplots(figsize=(10, 9))  # 扩展图像比例
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

# 去掉四周空白
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("pearson.png", dpi=300, bbox_inches="tight")
# plt.show()
