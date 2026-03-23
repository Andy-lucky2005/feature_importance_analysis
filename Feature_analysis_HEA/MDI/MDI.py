# RF GBRT
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
def print_full_ranking(mean_importance, model_name):

    # ---------------- 降序排序 ----------------
    sorted_idx = np.argsort(mean_importance)[::-1]

    print(f"\n{model_name} 平均特征重要性（降序）:")
    for idx in sorted_idx:
        print(f"{feature_names[idx]}: {mean_importance[idx]:.6f}")

    # ---------------- Heatmap排序（升序） ----------------
    print("Feature_importance_heatmap:")
    sorted_idx_asc = np.argsort(mean_importance)
    ranking_array = sorted_idx_asc.tolist()

    print("\n特征重要性排序数组（最不重要 → 最重要）:")
    print(ranking_array)

    # ---------------- GA排序 ----------------
    print("GA排序：")

    rank_array = np.empty(len(mean_importance), dtype=int)
    rank_array[np.argsort(mean_importance)[::-1]] = np.arange(1, len(mean_importance) + 1)

    print(rank_array.tolist())

    # ================= 写入 Excel =================
    df_rank = pd.DataFrame({
        "GA_rank": rank_array
    })

    file_name = f"{model_name}_GA_ranking.xlsx"

    df_rank.to_excel(file_name, index=False)

    print(f"{model_name} GA排序已保存 -> {file_name}")

# ---------------- 基本设置 ----------------
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
random_seed = 1412
np.random.seed(random_seed)
start_time = time.time()


out_pdf = "MDI.pdf"

# ---------------- 数据读取 ----------------
file_path = "../HEA_dataset/feature_data.xlsx"
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
    # ---------- 标准化 ----------
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)   # 只在训练集上 fit
    X_val_std   = scaler.transform(X_val)         # 验证集只 transform

    # 1. 随机森林 (最佳超参数)
    rf_model = RandomForestRegressor(
        random_state=random_seed,
        n_estimators=31,
        max_depth=10,
        max_features=14,
        min_samples_split=5,
        min_samples_leaf=1
    )
    rf_model.fit(X_train_std, y_train)
    rf_importances.append(rf_model.feature_importances_)
    rf_pred = rf_model.predict(X_val_std)
    rf_mae.append(mean_absolute_error(y_val, rf_pred))
    rf_r2.append(r2_score(y_val, rf_pred))

    # 2. GBRT (最佳超参数)
    gbr_model = GradientBoostingRegressor(
        random_state=random_seed,
        max_depth=9,
        learning_rate= 0.03723143650955813,
        n_estimators= 261,
        subsample= 0.5982094859828948,
        min_samples_split= 4,
        min_samples_leaf= 1
    )
    gbr_model.fit(X_train_std, y_train)
    gbr_importances.append(gbr_model.feature_importances_)
    gbr_pred = gbr_model.predict(X_val_std)
    gbr_mae.append(mean_absolute_error(y_val, gbr_pred))
    gbr_r2.append(r2_score(y_val, gbr_pred))


    # 3. XGBoost (最佳超参数)
    xgb_model = xgb.XGBRegressor(
        random_state=random_seed,
        n_estimators= 183,
        max_depth= 5,
        learning_rate= 0.056096157465759605,
        subsample= 0.7631547739293182,
        colsample_bytree= 0.6831883449711068,
        gamma= 0.8032243781288182,
        reg_alpha= 0.00010042295258819418,
        reg_lambda= 0.0004365708289039927
    )

    xgb_model.fit(X_train_std, y_train)
    xgb_importances.append(xgb_model.feature_importances_)
    xgb_pred = xgb_model.predict(X_val_std)
    xgb_mae.append(mean_absolute_error(y_val, xgb_pred))
    xgb_r2.append(r2_score(y_val , xgb_pred))

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


print_full_ranking(rf_mean, "随机森林")
print('---------------------------------------\n')

print_full_ranking(gbr_mean, "GBRT")
print('---------------------------------------\n')

print_full_ranking(xgb_mean, "XGBoost")
