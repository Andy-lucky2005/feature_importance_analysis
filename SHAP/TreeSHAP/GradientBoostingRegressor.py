# GBRT（Gradient Boosting Regression Trees
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import GradientBoostingRegressor  # 使用梯度提升树
from sklearn.metrics import mean_absolute_error, r2_score  # 用于计算 MAE 和 R²
from sklearn.model_selection import KFold  # 导入 KFold 进行交叉验证
import time

# 设置全局随机种子
random_seed = 1412
np.random.seed(random_seed)  # 设置 numpy 随机种子

plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


# 读取数据
file_path = "../../Feature_Value/Science_feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

# y 为第二列，目标变量
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

# 交叉验证
n_splits = 10  # 10折交叉验证
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

# 初始化模型
model = GradientBoostingRegressor(
        random_state=random_seed,
        max_depth=3,
        learning_rate= 0.07400928184149287,
        n_estimators=266,
        subsample= 0.8314787464970493,
        min_samples_split=9,
        min_samples_leaf=3
    )

# 存储 MAE 和 R²
mae_scores = []
r2_scores = []

# 存储每次交叉验证的特征重要性
all_importances = []
all_shap_values = []  # 用于存储所有的 SHAP 值
start_time = time.time()

# 交叉验证过程
for fold, (train_index, val_index) in enumerate(kf.split(X), 1):  # 增加fold编号

    # 获取训练集和验证集
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # 训练模型
    model.fit(X_train, y_train)

    # SHAP 特征重要性
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    print(shap_values, "\n...........")

    # 计算每个特征的重要性（使用 SHAP 值的绝对值均值）
    importance = np.abs(shap_values).mean(axis=0)
    all_importances.append(importance)  # 保存本折的重要性

    # 预测
    y_pred = model.predict(X_val)

    # 计算 R² 和 MAE
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    # 存储 MAE 和 R²
    mae_scores.append(mae)
    r2_scores.append(r2)

    # 将每次验证集的 SHAP 值合并
    all_shap_values.append(shap_values)

# 计算平均特征重要性
avg_importance = np.mean(all_importances, axis=0)

# 计算平均 MAE 和 R²
avg_mae = np.mean(mae_scores)
avg_r2 = np.mean(r2_scores)

# 输出平均 MAE 和 R²
print('---------------------------------------')
print('所有MAE:', mae_scores)
print('---------------------------------------'+'\n')
print('所有R²:', r2_scores)
print('---------------------------------------'+'\n')
print('10折交叉验证平均 MAE:', avg_mae)
print('10折交叉验证平均 R²:', avg_r2)
print('---------------------------------------'+'\n')
# 输出每次交叉验证的特征重要性
print('10折交叉验证特征重要性数据:')
print(all_importances)
print('---------------------------------------'+'\n')
# print('平均特征值：')
# print(avg_importance)
# print('---------------------------------------')
sorted_idx_desc = np.argsort(avg_importance)[::-1]

print("【特征重要性排序：从高到低】")
for idx in sorted_idx_desc:
    print(f"{feature_names[idx]:15s}  →  {avg_importance[idx]:.6f}")
print("---------------------------------------")


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
#     ['#EE7956', '#F4A830']  # 浅蓝 -> 深蓝，可根据需要调整色值
# )

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_blue_gradient',
    ['#60C3DF','#9AE859']  # 浅蓝 -> 深蓝，可根据需要调整色值
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

plt.savefig("GBRT_TreeSHAP_custom_order.png", dpi=300, bbox_inches="tight")
plt.close()

print("GBRT 特征重要性柱图已按指定顺序绘制完成！")
