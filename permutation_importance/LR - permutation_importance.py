import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import time

# ---------------- 基础设置 ----------------
random_seed = 1412
np.random.seed(random_seed)
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# ---------------- 读取数据 ----------------
file_path = "../Feature_Value/Science_feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

y = data.iloc[:, 1]   # 目标变量
X = data.iloc[:, 2:16]  # 特征

feature_names = [
    r"$\chi_p^M$", r"$\chi_p^{M'}$", r"$IE^M$", r"$IE^{M'}$",
    r"$r^M$", r"$r^{M'}$", r"$\Delta H_f^{MO}$", r"$\Delta H_f^{M'O}$",
    r"$\Delta H_f^{M'M}$", r"$\Delta H_{sub}^M$", r"$\Delta H_{sub}^{M'}$",
    r"$\gamma^M$", r"$n_{ws}^M$", r"$E_g^{M'O}$"
]

# ---------------- 十折交叉验证 + 标准化 ----------------
kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)

mae_scores, r2_scores = [], []
feature_importance_list = []

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 标准化（仅在训练集上拟合）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 训练 Linear Regression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # 验证预测
    y_pred = model.predict(X_val_scaled)
    mae_scores.append(mean_absolute_error(y_val, y_pred))
    r2_scores.append(r2_score(y_val, y_pred))

    # 置换重要性分析（在标准化后的验证集上做）
    result = permutation_importance(
        model, X_val_scaled, y_val,
        n_repeats=50, random_state=random_seed
    )
    feature_importance_list.append(result.importances_mean)

# ---------------- 结果统计 ----------------
avg_mae = np.mean(mae_scores)
avg_r2 = np.mean(r2_scores)
avg_importance = np.mean(feature_importance_list, axis=0)

print("每折 MAE:", mae_scores)
print("每折 R²:", r2_scores)
print("\n平均 MAE:", avg_mae)
print("平均 R²:", avg_r2)

print("\nPermutation Importance（平均）:")
for i, imp in enumerate(avg_importance):
    print(f"{feature_names[i]}: {imp:.6f}")



# ========== 自定义排序：使用你指定的科学顺序 ==========
custom_order = [
    r"$\gamma^M$", r"$\Delta H_f^{MO}$", r"$\Delta H_f^{M'O}$",
    r"$\Delta H_{sub}^M$", r"$\chi_p^{M'}$", r"$\chi_p^M$",
    r"$\Delta H_{sub}^{M'}$", r"$IE^M$", r"$\Delta H_f^{M'M}$",
    r"$E_g^{M'O}$", r"$n_{ws}^M$", r"$r^M$", r"$r^{M'}$", r"$IE^{M'}$"
]

# 建立 custom_order 在原 feature_names 中的索引
order_idx = [feature_names.index(f) for f in custom_order]
avg_importance_ordered = avg_importance[order_idx]

# custom_cmap = LinearSegmentedColormap.from_list(
#     'custom_blue_gradient',
#     ['#60C3DF', '#9AE859']  # 浅蓝 -> 深蓝，可根据需要调整色值
# )
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_blue_gradient',
    ['#807FFE', '#70D0F6']  # 浅蓝 -> 深蓝，可根据需要调整色值
)
color_list = custom_cmap(np.linspace(0, 1, len(avg_importance_ordered)))
# ========== 绘图：不按大小排，只按 custom_order ==========
fig, ax = plt.subplots(figsize=(15, 23))
plt.barh(
    range(len(avg_importance_ordered)),
    avg_importance_ordered,
    color = color_list,
    height=0.7
)

# 设置Y轴标签为 custom_order
plt.yticks(
    range(len(avg_importance_ordered)),
    custom_order,
    fontsize=53,
    fontweight="bold",
    # fontname="Arial"
)

plt.gca().invert_yaxis()

# 设置X轴刻度
x_max = np.max(avg_importance_ordered)
x_ticks = np.linspace(0, x_max, 4)
plt.xticks(x_ticks, [f'{x:.2f}' for x in x_ticks], fontsize=45)

plt.xlabel("Permutation Importance", fontsize=45)

#边框加粗
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.savefig("LR_PFI_custom_order.png", dpi=300, bbox_inches="tight")
plt.close()

print("LR 特征重要性柱图已按指定顺序绘制完成！")
