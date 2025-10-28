import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb  # 导入 XGBoost
import time

from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_absolute_error, r2_score  # 导入计算 MAE 和 R² 的函数
from sklearn.model_selection import KFold  # 导入 KFold 进行交叉验证

# 设置全局随机种子
random_seed = 1412
np.random.seed(random_seed)  # 设置 numpy 随机种子
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# 读取数据
file_path = "../../Feature_Value/Science_feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

# y 为第二列，目标变量
y = data.iloc[:, 1]  # 第二列作为目标变量 Eadh
# X 为从第三列到第16列的14个特征
X = data.iloc[:, 2:16]  # 从第三列到第16列（包含第16列）

# 创建特征名称列表
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

# 设置交叉验证的折数
n_splits = 10  # 10折交叉验证
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

# 存储每次交叉验证的 MAE 和 R²
mae_scores = []
r2_scores = []
feature_importance_list = []
all_shap_values = []  # 用于存储所有的 SHAP 值

# 记录整体训练时间
start_time = time.time()

# 使用最佳超参数初始化 XGBoost 模型
best_params = {
    'max_depth': 3,
    'learning_rate': 0.09222929445288928,
    'n_estimators': 199,
    'subsample': 0.5003718198735058,
    'colsample_bytree': 0.8310917506470121,
    'gamma': 0.13566644039562242,
    'reg_alpha': 0.029867618709022974,
    'reg_lambda': 0.0005818694744638248
}

# 交叉验证过程
for fold, (train_index, val_index) in enumerate(kf.split(X), 1):  # 增加fold编号

    # 获取训练集和验证集
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # 初始化 XGBoost 模型
    model = xgb.XGBRegressor(**best_params, random_state=random_seed)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_val)

    # 计算 R² 和 MAE
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    # 存储 MAE 和 R²
    mae_scores.append(mae)
    r2_scores.append(r2)

    # SHAP 特征重要性
    explainer = shap.Explainer(model, X_train)  # 使用训练数据来确保解释器的一致性
    shap_values = explainer.shap_values(X_val)
    print(shap_values)
    # 计算每个特征的重要性（使用 SHAP 值的绝对值均值）
    importance = np.abs(shap_values).mean(axis=0)
    feature_importance_list.append(importance)

    # 将每次验证集的 SHAP 值合并
    all_shap_values.append(shap_values)
# 结束计时
end_time = time.time()
print('---------------------------------------')
# 输出每次交叉验证的 R² 和 MAE 数据
print("\n每次交叉验证的 MAE: ", mae_scores)
print('---------------------------------------'+'\n')
print("每次交叉验证的 R²: ", r2_scores)
print('---------------------------------------'+'\n')
# 计算平均 R² 和 MAE
avg_r2 = np.mean(r2_scores)
avg_mae = np.mean(mae_scores)
print('---------------------------------------'+'\n')
# 输出平均 R² 和 MAE
print(f"平均决定系数 (R²): {avg_r2:.4f}")
print('---------------------------------------'+'\n')
print(f"平均平均绝对误差 (MAE): {avg_mae:.4f}")
print('---------------------------------------'+'\n')
print('特征重要性所有数据：')
print(feature_importance_list)
print('---------------------------------------'+'\n')

# sorted_idx_desc = np.argsort(feature_importance_list)[::-1].tolist()
# for idx in sorted_idx_desc:
#     print(f"{feature_names[idx]}  →  {feature_importance_list[idx]:.6f}")


# 汇总所有交叉验证的特征重要性
avg_feature_importance = np.mean(feature_importance_list, axis=0)

# 输出每次交叉验证的特征重要性
print("\n平均特征重要性:")
for i, importance in enumerate(avg_feature_importance):
    print(f"{feature_names[i]}: {importance}")
print('---------------------------------------'+'\n')



# ========== 自定义排序：使用你指定的科学顺序 ==========
custom_order = [
    r"$\gamma^M$", r"$\Delta H_f^{MO}$", r"$\Delta H_f^{M'O}$",
    r"$\Delta H_{sub}^M$", r"$\chi_p^{M'}$", r"$\chi_p^M$",
    r"$\Delta H_{sub}^{M'}$", r"$IE^M$", r"$\Delta H_f^{M'M}$",
    r"$E_g^{M'O}$", r"$n_{ws}^M$", r"$r^M$", r"$r^{M'}$", r"$IE^{M'}$"
]

# 建立 custom_order 在原 feature_names 中的索引
order_idx = [feature_names.index(f) for f in custom_order]
avg_feature_importance = np.mean(feature_importance_list, axis=0)
avg_importance_ordered = avg_feature_importance[order_idx]  # 按自定义顺序排列


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
    fontweight="bold"
)

plt.gca().invert_yaxis()

# 设置X轴刻度
x_max = np.max(avg_importance_ordered)
x_ticks = np.linspace(0, x_max, 4)
plt.xticks(x_ticks, [f'{x:.2f}' for x in x_ticks], fontsize=45)

plt.xlabel("Average SHAP Importance", fontsize=45)

#边框加粗
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.savefig("XGBoost_TreeSHAP_custom_order.png", dpi=300, bbox_inches="tight")
plt.close()

print("XGBoost 特征重要性柱图已按指定顺序绘制完成！")
