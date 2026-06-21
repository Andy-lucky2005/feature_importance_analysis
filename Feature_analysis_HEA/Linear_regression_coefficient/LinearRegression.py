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
file_path = "../HEA_dataset/feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

# y：第2列（目标变量），X：第3~16列（14个特征）
y = data.iloc[:, 1]
X = data.iloc[:, 2:56]

# 特征名（Latex 格式）
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

print("Feature_importance_heatmap:")
sorted_idx_asc = np.argsort(lr_mean_importance)   # 升序
ranking_array = sorted_idx_asc.tolist()
print("\n线性回归特征重要性排序数组（最不重要 → 最重要）:")
print(ranking_array)