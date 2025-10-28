import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.preprocessing import StandardScaler  # 标准化工具

# 设置全局随机种子
random_seed = 1412
np.random.seed(random_seed)
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# 读取数据
file_path = "../../Feature_Value/Science_feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

# y 为第二列
y = data.iloc[:, 1]
# X 为从第三列到第16列的14个特征
X = data.iloc[:, 2:16]

# 特征名称
feature_names = [
    r"$\chi_p^M$",
    r"$\chi_p^{M'}$",
    r"$IE^M$",
    r"$IE^{M'}$",
    r"$r^M$",
    r"$r^{M'}$",
    r"$\Delta H_f^{MO}$",
    r"$\Delta H_f^{M'O}$",
    r"$\Delta H_f^{M'M}$",
    r"$\Delta H_{sub}^M$",
    r"$\Delta H_{sub}^{M'}$",
    r"$\gamma^M$",
    r"$n_{ws}^M$",
    r"$E_g^{M'O}$"
]

#使用最新最佳超参数
best_params = {
    'hidden_layer_sizes': (40, 8),  # 两层：第一层42个神经元，第二层22个神经元
    'activation': 'relu',
    'solver': 'lbfgs',
    'max_iter': 6500,
    'alpha': 0.009885444029062893
}

# 交叉验证设置
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

# 存储指标
mae_scores, r2_scores = [], []
all_shap_values, all_importances = [], []

# 初始化标准化器（在循环外保证所有折一致）
scaler = StandardScaler()

# 交叉验证
for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    #在训练集上拟合标准化器，并应用到训练集和验证集
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 初始化模型
    model = MLPRegressor(**best_params, random_state=random_seed)

    # 训练模型（使用标准化后的 X）
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_val_scaled)

    # 计算指标
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    mae_scores.append(mae)
    r2_scores.append(r2)

    # SHAP 解释器（基于标准化后的数据）
    explainer = shap.KernelExplainer(model.predict, X_train_scaled)
    shap_values = explainer.shap_values(X_val_scaled)

    # 特征重要性
    importance = np.abs(shap_values).mean(axis=0)
    all_importances.append(importance)
    all_shap_values.append(shap_values)

# 平均特征重要性
avg_importance = np.mean(all_importances, axis=0)

# 输出指标
avg_mae = np.mean(mae_scores)
avg_r2 = np.mean(r2_scores)
print("\n所有交叉验证的 MAE:", mae_scores)
print("\n所有交叉验证的 R²:", r2_scores)
print(f"\n平均 MAE: {avg_mae}")
print(f"平均 R²: {avg_r2}")

# 平均特征重要性排序输出
sorted_idx = np.argsort(avg_importance)[::-1]
sorted_features = np.array(feature_names)[sorted_idx]
sorted_importances = avg_importance[sorted_idx]

print("\n=== 特征重要性排序（从大到小） ===")
for feat, val in zip(sorted_features, sorted_importances):
    print(f"{feat}: {val:.6f}")




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
#     ['#9AE859', '#60C3DF']  # 浅蓝 -> 深蓝，可根据需要调整色值
# )

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_blue_gradient',
    ['#EE7956', '#F4A830']  # 浅蓝 -> 深蓝，可根据需要调整色值
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
plt.xticks(x_ticks, [f'{x:.2f}' for x in x_ticks], fontsize=45,)

plt.xlabel("Average SHAP Importance", fontsize=45,)

#边框加粗
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.savefig("MLP_KernelSHAP_custom_order.png", dpi=300, bbox_inches="tight")
plt.close()

print("MLP 特征重要性柱图已按指定顺序绘制完成！")
