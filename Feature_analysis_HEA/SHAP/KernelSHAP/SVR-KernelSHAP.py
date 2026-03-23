import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler  # 标准化工具

# ---------------- 全局设置 ----------------
random_seed = 1412
np.random.seed(random_seed)
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# ---------------- 读取数据 ----------------
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

# ---------------- 使用最佳 SVR 超参数 ----------------
best_params = {
    'kernel': 'linear',
    'C': 582.1033339684861,
    'epsilon': 0.19634276708263904,
    # 'gamma': 0.3578200860232905
}

# best_params = {
#     'kernel': 'rbf',
#     'C': 1.0370559264947947,
#     'epsilon': 0.02715970423042251,
#     'gamma': 0.11282947064237635
# }


# ---------------- 交叉验证设置 ----------------
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

# ---------------- 存储指标 ----------------
mae_scores, r2_scores = [], []
all_shap_values, all_importances = [], []

# ---------------- 初始化标准化器（在每折内独立 fit） ----------------
scaler = StandardScaler()

# ---------------- 交叉验证 + KernelSHAP ----------------
for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
    print(f"\n=== 第 {fold} 折 ===")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # 标准化
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 初始化 SVR 模型
    model = SVR(**best_params)

    # 训练
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_val_scaled)

    # 计算性能指标
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    mae_scores.append(mae)
    r2_scores.append(r2)
    print(f"MAE: {mae:.4f} | R²: {r2:.4f}")

    # KernelSHAP 解释器
    explainer = shap.KernelExplainer(model.predict, X_train_scaled)

    shap_values = explainer.shap_values(X_val_scaled)
    all_shap_values.append(shap_values)

    # 计算特征重要性（取绝对值的平均值）
    importance = np.abs(shap_values).mean(axis=0)
    all_importances.append(importance)

# ---------------- 平均特征重要性 ----------------
avg_importance = np.mean(all_importances, axis=0)

# ---------------- 输出指标 ----------------
avg_mae = np.mean(mae_scores)
avg_r2 = np.mean(r2_scores)
print("\n所有交叉验证的 MAE:", mae_scores)
print("所有交叉验证的 R²:", r2_scores)
print(f"\n平均 MAE: {avg_mae:.4f}")
print(f"平均 R²: {avg_r2:.4f}")

# ---------------- 特征重要性排序 ----------------
sorted_idx = np.argsort(avg_importance)[::-1]
sorted_features = np.array(feature_names)[sorted_idx]
sorted_importances = avg_importance[sorted_idx]

print("\n=== 特征重要性排序（从大到小） ===")
for feat, val in zip(sorted_features, sorted_importances):
    print(f"{feat}: {val:.6f}")




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

plt.savefig("SVR_KernelSHAP_custom_order.png", dpi=300, bbox_inches="tight")
plt.close()

print("SVR 特征重要性柱图已按指定顺序绘制完成！")
# ---------------- PFI排序输出 ----------------

sorted_idx = np.argsort(avg_importance)[::-1]

print("\nPFI 平均特征重要性（降序）:")
for idx in sorted_idx:
    print(f"{feature_names[idx]}: {avg_importance[idx]:.6f}")

# -------- Feature_importance_heatmap --------
print("Feature_importance_heatmap:")

sorted_idx_asc = np.argsort(avg_importance)
ranking_array = sorted_idx_asc.tolist()

print("\nPFI特征重要性排序数组（最不重要 → 最重要）:")
print(ranking_array)

# -------- GA排序 --------
print("GA排序：")

rank_array = np.empty(len(avg_importance), dtype=int)
rank_array[np.argsort(avg_importance)[::-1]] = np.arange(1, len(avg_importance) + 1)

print(rank_array.tolist())

ga_df = pd.DataFrame({
    "GA_rank": rank_array
})

ga_df.to_excel("GA_ranking_SVR_KernelSHAP.xlsx", index=False)

print("GA排序已保存到 GA_ranking.xlsx")