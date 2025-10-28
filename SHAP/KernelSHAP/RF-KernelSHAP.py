import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.model_selection import KFold

# 设置全局随机种子
random_seed = 1412
np.random.seed(random_seed)
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
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

# 初始化模型
model = RandomForestRegressor(
        random_state=random_seed,
        n_estimators=66,
        max_depth=9,
        max_features=7,
        min_samples_split=2,
        min_samples_leaf=1
    )

# 存储 MAE 和 R²
mae_scores = []
r2_scores = []

# 存储每次交叉验证的特征重要性
all_importances = []
all_shap_values = []

start_time = time.time()

for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # 训练模型
    model.fit(X_train, y_train)

    # KernelSHAP (采样背景数据, 避免过大计算开销)
    # background = shap.sample(X_train, 50, random_state=random_seed)  # 从训练集采样50个点作为背景
    explainer = shap.KernelExplainer(model.predict, X_train)

    # 计算验证集 SHAP 值（注意 KernelSHAP 计算量非常大，这里只取部分验证集样本）
    # shap_values = explainer.shap_values(X_val, nsamples=100)  # nsamples 可调，越大越准但越慢
    shap_values = explainer.shap_values(X_val)  # nsamples 可调，越大越准但越慢

    importance = np.abs(shap_values).mean(axis=0)
    all_importances.append(importance)

    # 预测
    y_pred = model.predict(X_val)

    # 误差指标
    mae_scores.append(mean_absolute_error(y_val, y_pred))
    r2_scores.append(r2_score(y_val, y_pred))

    # 保存 shap 值
    all_shap_values.append(shap_values)

# 计算平均特征重要性
avg_importance = np.mean(all_importances, axis=0)
print('10折交叉特征重要性分析数据:')
print(all_importances)
print('---------------------------------------')
print('特征重要性平均数据:')
print(avg_importance)
print('---------------------------------------')
print('10次MAE数据：')
print(mae_scores)
print('---------------------------------------')
print('10次R²数据：')
print(r2_scores)
print('---------------------------------------')
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

plt.savefig("RF_KernelSHAP_custom_order.png", dpi=300, bbox_inches="tight")
plt.close()

print("RF 特征重要性柱图已按指定顺序绘制完成！")
