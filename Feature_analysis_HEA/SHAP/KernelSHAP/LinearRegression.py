import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression  # 导入线性回归模型
from sklearn.metrics import mean_absolute_error, r2_score  # 用于计算 MAE 和 R²
from sklearn.model_selection import KFold  # 导入 KFold 进行交叉验证
from sklearn.preprocessing import StandardScaler  # 用于标准化
import matplotlib.pyplot as plt
import time
import numpy as np

# 设置全局随机种子
random_seed = 1412
np.random.seed(random_seed)  # 设置 numpy 随机种子
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# 读取数据
file_path = "../../HEA_dataset/feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

# y：第2列（目标变量），X：第3~16列（14个特征）
y = data.iloc[:, 1]
X = data.iloc[:, 2:56]

# 特征名
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

# 初始化线性回归模型
model = LinearRegression()

# 交叉验证过程
for fold, (train_index, val_index) in enumerate(kf.split(X), 1):  # 增加 fold 编号
    # 获取训练集和验证集
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # ---------- 标准化处理 ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 用训练集拟合并转换
    X_val_scaled = scaler.transform(X_val)  # 用训练集的参数转换验证集

    # 训练模型
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_val_scaled)

    # 计算 R² 和 MAE
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    # 存储 MAE 和 R²
    mae_scores.append(mae)
    r2_scores.append(r2)

    # SHAP 特征重要性分析
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_val_scaled)

    # 计算特征重要性并存储
    importance = np.abs(shap_values).mean(axis=0)  # 计算每个特征的平均SHAP值
    feature_importance_list.append(importance)
    # 将每次验证集的 SHAP 值合并
    all_shap_values.append(shap_values)

print('---------------------------------------'+'\n')
# 输出所有折次的 MAE 和 R²
print("\n所有交叉验证的 MAE: ", mae_scores)
print('---------------------------------------'+'\n')
print("所有交叉验证的 R²: ", r2_scores)

# 计算平均 MAE 和 R²
avg_mae = np.mean(mae_scores)
avg_r2 = np.mean(r2_scores)

print('---------------------------------------'+'\n')
print(f"\n平均 MAE: {avg_mae}")
print('---------------------------------------'+'\n')
print(f"平均 R²: {avg_r2}")
print('---------------------------------------'+'\n')

# 汇总所有交叉验证的特征重要性
avg_feature_importance = np.mean(feature_importance_list, axis=0)
print('---------------------------------------'+'\n')
print("\n10次交叉验证的特征重要性: ")
for i, importance in enumerate(avg_feature_importance):
    print(f"{feature_names[i]}: {importance}")
print('---------------------------------------'+'\n')



# ========== 自定义排序：使用你指定的科学顺序 ==========
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

# 建立 custom_order 在原 feature_names 中的索引
order_idx = [feature_names.index(f) for f in custom_order]
avg_importance_ordered = avg_feature_importance[order_idx]
importance_sorted = sorted(
    zip(feature_names, avg_importance_ordered),
    key=lambda x: x[1],
    reverse=True
)

print("\n（按从大到小降序排列）:")
for name, imp in importance_sorted:
    print(f"{name}: {imp:.6f}")
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

plt.savefig("LR_KernelSHAP_custom_order.png", dpi=300, bbox_inches="tight")
plt.close()

print("LR 特征重要性柱图已按指定顺序绘制完成！")
# ---------------- PFI排序输出 ----------------
# ---------------- LR-SHAP 排序输出 ----------------

sorted_idx = np.argsort(avg_feature_importance)[::-1]

print("\nLR-SHAP 平均特征重要性（降序）:")
for idx in sorted_idx:
    print(f"{feature_names[idx]}: {avg_feature_importance[idx]:.6f}")

# -------- Feature_importance_heatmap --------
print("Feature_importance_heatmap:")

sorted_idx_asc = np.argsort(avg_feature_importance)
ranking_array = sorted_idx_asc.tolist()

print("\nLR-SHAP特征重要性排序数组（最不重要 → 最重要）:")
print(ranking_array)

# -------- GA排序 --------
print("GA排序：")

rank_array = np.empty(len(avg_feature_importance), dtype=int)
rank_array[np.argsort(avg_feature_importance)[::-1]] = np.arange(1, len(avg_feature_importance) + 1)

print(rank_array.tolist())

ga_df = pd.DataFrame({
    "GA_rank": rank_array
})

ga_df.to_excel("GA_ranking_LR_KernelSHAP.xlsx", index=False)

print("GA排序已保存到 GA_ranking.xlsx")