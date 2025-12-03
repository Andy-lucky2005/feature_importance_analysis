import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- 基本设置 ----------------
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
random_seed = 1412
np.random.seed(random_seed)
start_time = time.time()


out_pdf = "MDI.pdf"

# ---------------- 数据读取 ----------------
file_path = "../Feature_Value/Science_feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

y = data.iloc[:, 1]  # 第二列为目标变量
X = data.iloc[:, 2:16]  # 第3列到第16列为特征

# 特征名
feature_names = [
    r"$\chi_p^M$", r"$\chi_p^{M'}$", r"$IE^M$", r"$IE^{M'}$", r"$r^M$", r"$r^{M'}$",
    r"$\Delta H_f^{MO}$", r"$\Delta H_f^{M'O}$", r"$\Delta H_f^{M'M}$",
    r"$\Delta H_{sub}^M$", r"$\Delta H_{sub}^{M'}$", r"$\gamma^M$", r"$n_{ws}^M$", r"$E_g^{M'O}$"
]

# ---------------- 十折交叉验证 ----------------
kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)

# 用于存储每个模型在每一折的特征重要性及其 MAE 和 R²
rf_importances = []
gbr_importances = []
xgb_importances = []

rf_mae = []
gbr_mae = []
xgb_mae = []

rf_r2 = []
gbr_r2 = []
xgb_r2 = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 1. 随机森林 (最佳超参数)
    rf_model = RandomForestRegressor(
        random_state=random_seed,
        n_estimators=66,
        max_depth=9,
        max_features=7,
        min_samples_split=2,
        min_samples_leaf=1
    )

    rf_model.fit(X_train, y_train)
    rf_importances.append(rf_model.feature_importances_)
    rf_pred = rf_model.predict(X_val)
    rf_mae.append(mean_absolute_error(y_val, rf_pred))
    rf_r2.append(r2_score(y_val, rf_pred))

    # 2. GBRT (最佳超参数)
    gbr_model = GradientBoostingRegressor(
        random_state=random_seed,
        max_depth=3,
        learning_rate= 0.07400928184149287,
        n_estimators= 266,
        subsample= 0.8314787464970493,
        min_samples_split= 9,
        min_samples_leaf= 3
    )
    gbr_model.fit(X_train, y_train)
    gbr_importances.append(gbr_model.feature_importances_)
    gbr_pred = gbr_model.predict(X_val)
    gbr_mae.append(mean_absolute_error(y_val, gbr_pred))
    gbr_r2.append(r2_score(y_val, gbr_pred))

    # 3. XGBoost (最佳超参数)
    xgb_model = xgb.XGBRegressor(
        random_state=random_seed,
        n_estimators= 199,
        max_depth= 3,
        learning_rate= 0.09222929445288928,
        subsample= 0.5003718198735058,
        colsample_bytree= 0.8310917506470121,
        # gamma= 0.13566644039562242,
        reg_alpha= 0.029867618709022974,
        reg_lambda= 0.0005818694744638248
    )
    xgb_model.fit(X_train, y_train)
    xgb_importances.append(xgb_model.feature_importances_)
    xgb_pred = xgb_model.predict(X_val)
    xgb_mae.append(mean_absolute_error(y_val, xgb_pred))
    xgb_r2.append(r2_score(y_val, xgb_pred))

# ---------------- 计算十折平均值 ----------------
rf_mean = np.mean(rf_importances, axis=0)
gbr_mean = np.mean(gbr_importances, axis=0)
xgb_mean = np.mean(xgb_importances, axis=0)

rf_mean_mae = np.mean(rf_mae)
gbr_mean_mae = np.mean(gbr_mae)
xgb_mean_mae = np.mean(xgb_mae)

rf_mean_r2 = np.mean(rf_r2)
gbr_mean_r2 = np.mean(gbr_r2)
xgb_mean_r2 = np.mean(xgb_r2)

# ---------------- 输出 ----------------
print('---------------------------------------' + '\n')
print(f"随机森林平均 MAE: {rf_mean_mae}")
print(f"随机森林平均 R²: {rf_mean_r2}")
print('---------------------------------------' + '\n')

print(f"GBRT平均 MAE: {gbr_mean_mae}")
print(f"GBRT平均 R²: {gbr_mean_r2}")
print('---------------------------------------' + '\n')

print(f"XGBoost平均 MAE: {xgb_mean_mae}")
print(f"XGBoost平均 R²: {xgb_mean_r2}")
print('---------------------------------------' + '\n')

# ---------------- 绘图 ----------------
width = 0.25
x = np.arange(len(feature_names))

fig = plt.figure(figsize=(10, 7))

plt.bar(x - width, rf_mean, label="RF", color="#E24B36", width=width)
plt.bar(x, gbr_mean, label="GBRT", color="#3C5382", width=width)
plt.bar(x + width, xgb_mean, label="XGBoost", color="#00A088", width=width)

# 按平均重要性排序并输出
def print_sorted_importance(mean_importance, model_name):
    sorted_idx = np.argsort(mean_importance)[::-1]  # 从大到小排序索引
    print(f"\n{model_name} 平均特征重要性（降序）:")
    for idx in sorted_idx:
        print(f"{feature_names[idx]}: {mean_importance[idx]}")

# 输出模型的特征重要性
print_sorted_importance(rf_mean, "随机森林")
print('---------------------------------------' + '\n')

print_sorted_importance(gbr_mean, "GBRT")
print('---------------------------------------' + '\n')

print_sorted_importance(xgb_mean, "XGBoost")

plt.xticks(np.arange(len(feature_names)), feature_names, rotation=45, ha="center", fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('MDI Importance', fontsize=16)
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
plt.legend(
    loc='lower center',  # 定位在中心下方
    bbox_to_anchor=(0.5, 1.0),  # 在正上方，调整正值可以控制距离
    ncol=3,                       # 横排排列3个
    fontsize=18,                   # 图例字体大小
    # frameon=False
)
plt.tight_layout()

# 保存为 PDF
plt.savefig(out_pdf, dpi=300, edgecolor='white')
plt.close(fig)

print("保存完成：", out_pdf)
