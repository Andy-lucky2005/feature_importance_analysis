import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ---------------- 基本设置 ----------------
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
random_seed = 1412
np.random.seed(random_seed)
start_time = time.time()

out_pdf = "LinearRegression_coef.pdf"

# ---------------- 数据读取 ----------------
file_path = "../perovskite_dataset/feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

# y：第2列（目标变量），X：第3~16列（14个特征）
y = data.iloc[:, 1]
X = data.iloc[:, 2:74]

# 特征名（Latex 格式）
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

# ---------------- 十折交叉验证 ----------------
kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)
coef_importances = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 训练线性回归
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    coef_importances.append(np.abs(model.coef_))  # 绝对值

# ---------------- 计算平均重要性 ----------------
lr_mean_importance = np.mean(coef_importances, axis=0)

# ---------------- 自定义颜色列表（按顺序对应每个柱子） ----------------
custom_colors = [
    "#3B4CC0", "#516DDB",
    "#6B8DF0", "#86A9FC",
    "#A1C0FF", "#BBD1F8",
    "#D3DBE7", "#E6D7CF",
    "#F3C7B1", "#F7AF91",
    "#F29274", "#E46E56",
    "#CF453C", "#B40426"
]

# ---------------- 对特征按重要性排序 ----------------
sorted_idx = np.argsort(lr_mean_importance)[::-1]  # 降序索引
sorted_importance = lr_mean_importance[sorted_idx]  # 排序后的数值
sorted_feature_names = [feature_names[i] for i in sorted_idx]  # 排序后的特征名
# sorted_colors = [custom_colors[i] for i in sorted_idx]  # 排序后的颜色

# ---------------- 绘制柱状图 ----------------
x = np.arange(len(sorted_feature_names))
width = 0.75  # 柱宽

fig, ax = plt.subplots(figsize=(8, 7))
ax.bar(x, sorted_importance, color=custom_colors, width=width)

# 坐标轴与排版设置
plt.xticks(x, sorted_feature_names, rotation=45, ha="center", fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('Coefficient Importance', fontsize=16)

plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.25)

# 保存 PDF
plt.savefig(out_pdf, dpi=300, bbox_inches='tight', edgecolor='white')
plt.close(fig)

print("保存完成：", out_pdf)

# 输出排序结果
print("\n线性回归 平均特征重要性（降序）:")
for idx in sorted_idx:
    print(f"{feature_names[idx]}: {lr_mean_importance[idx]:.4f}")

print("Feature_importance_heatmap:")
sorted_idx_asc = np.argsort(lr_mean_importance)   # 升序
ranking_array = sorted_idx_asc.tolist()
print("\n线性回归特征重要性排序数组（最不重要 → 最重要）:")
print(ranking_array)

print("GA排序：")
rank_array = np.empty(len(lr_mean_importance), dtype=int)
rank_array[np.argsort(lr_mean_importance)[::-1]] = np.arange(1, len(lr_mean_importance)+1)
print(rank_array.tolist())

ga_df = pd.DataFrame({
    "GA_rank": rank_array
})

ga_df.to_excel("GA_ranking.xlsx", index=False)

print("GA排序已保存到 GA_ranking.xlsx")