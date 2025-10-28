import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

# 1. 读取Excel数据
file_path = "Feature mean.xlsx"
df = pd.read_excel(file_path, header=0)

# 2. 提取14个重要特征
all_columns = df.columns.tolist()
selected_features = all_columns[1:]

# 3. 数据清洗
df_selected = df[selected_features].apply(pd.to_numeric, errors='coerce').dropna()

# 4. 计算Spearman相关系数矩阵
corr_matrix = df_selected.corr(method='spearman')
corr_matrix = corr_matrix.abs()

# 5. 特征简化名称
feature_labels = {
    "RF-TreeSHAP": r"RF-TreeSHAP",
    "GBRT-TreeSHAP": r"GBRT-TreeSHAP",
    "XGBoost-TreeSHAP": r"XGBoost-TreeSHAP",

    "RF-KernelSHAP": r"RF-KernelSHAP",
    "GBRT-KernelSHAP": r"GBRT-KernelSHAP",
    "XGBoost-KernelSHAP": r"XGBoost-KernelSHAP",

    "RF-permutation": r"RF-PFI",
    "GBRT-permutation": r"GBRT-PFI",
    "XGBoost-permutation": r"XGBoost-PFI",

    "LR-KernelSHAP": r"LR-KernelSHAP",
    "MLP-KernelSHAP": r"MLP-KernelSHAP",

    "LR-permutation": r"LR-PFI",
    "MLP-permutation": r"MLP-PFI",

    "LR-coef": r"LR-Coefficient",

    "RF": r"RF-MDI",
    "GBRT": r"GBRT-MDI",
    "XGBoost": r"XGBoost-MDI",

    "MI": r"MI",

    "pearson correlation": r"Pearson",
    "spearman correlation": r"Spearman",

    "Eadh formula(Feature mean)": r"Formula-MVPD",
    # "Eadh formula(Average of all data)": r"$E_{\mathrm{adh}}$ Formula(Data mean)",
    # "Eadh formula(Average of all data)": r"Formula(Data mean)",
}

corr_matrix = corr_matrix.rename(columns=feature_labels, index=feature_labels)

# 6. 计算与 "Eadh formula" 的相关性
adh_corr = corr_matrix["Formula-MVPD"].sort_values(ascending=False)

# 7. 根据与 Eadh formula 相关性的大小对相关矩阵进行排序
sorted_corr_matrix = corr_matrix.loc[adh_corr.index, adh_corr.index]

# 转换为 numpy 数组用于 imshow
corr_values = sorted_corr_matrix.values
labels = sorted_corr_matrix.columns.tolist()

# 8. 自定义 colormap + norm
bounds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,  0.8, 0.9, 1.0]
colors = ["#3654A5", "#0070B3", "#1F9FD4", "#35B79C", "#C0BEC0","#E4E0E4",
          "#F7D635", "#F4A153", "#F48E98", "#EB3A4B"]
cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, cmap.N)

# 9. 绘图
fig = plt.figure(figsize=(11, 8))  # Ensure figure size is adequate for the image
# gs = gridspec.GridSpec(1, 2, width_ratios=[30, 1.2], wspace=0.00001)
gs = gridspec.GridSpec(1, 2, width_ratios=[30, 0.8], wspace=0.01)  # 色条更窄

# 主图区域
ax = fig.add_subplot(gs[0])
im = ax.imshow(corr_values, cmap=cmap, norm=norm)

# 设置坐标轴标签
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
ax.invert_yaxis()

# 添加黑色网格
for edge, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_color("black")
# 设置白色边距
ax.set_xticks(np.arange(len(labels)+1)-0.5, minor=True)
ax.set_yticks(np.arange(len(labels)+1)-0.5, minor=True)
ax.grid(which="minor", color="white", linewidth=1.0)
# 去除横纵坐标轴的刻度线
ax.tick_params(which="both", bottom=False, left=False)

# 数值注释
# for i in range(len(labels)):
#     for j in range(len(labels)):
#         val = corr_values[i, j]
#         ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5)

# 添加 colorbar
cbar_ax = fig.add_subplot(gs[1])
cb = ColorbarBase(
    cbar_ax, cmap=cmap, norm=norm,
    ticks=[0, 0.2,0.4,0.6, 0.8,1.0],
    spacing='proportional',
    orientation='vertical'
)

# Remove excess margins and make image fill the figure

fig.subplots_adjust(left=0.18, right=0.82, top=0.95, bottom=0.2)
plt.tight_layout(pad=0.5)  # Adjust padding to ensure that labels don't get cut off

# 6. 计算与 "Eadh formula" 的相关性
adh_corr = corr_matrix["Formula-MVPD"].sort_values(ascending=False)
# 输出按照 Spearman 相关系数从大到小排序的结果
print("=== 与 Eadh formula(Feature Mean) 的 Spearman 相关系数（从大到小） ===")
print(adh_corr)
# 保存为 PDF
png_file = "Spearman_Correlation_with_Formula_mean.png"
output_file = "Spearman_Correlation_with_Formula_mean.pdf"
plt.savefig(output_file, format="pdf", bbox_inches="tight")
plt.savefig(png_file, format="png", dpi=300, bbox_inches="tight")  # 300 dpi 适合论文和PPT
plt.show()

print(f"\n已将相关性结果保存为 PDF 文件：{output_file}")

output_excel = "Spearman_Correlation_Results.xlsx"
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    corr_matrix.to_excel(writer, sheet_name="Full Correlation Matrix")
    adh_corr.to_excel(writer, sheet_name="Formula(Feature Mean) Ranking")

print(f"Spearman 相关矩阵与排序结果已保存至：{output_excel}")
