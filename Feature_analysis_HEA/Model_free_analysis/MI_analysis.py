import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

# 全局字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

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

# ========== 2. 特征归一化（推荐）==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== 3. 计算互信息 MI ==========
mi_scores = mutual_info_regression(X_scaled, y, random_state=1412)

# DataFrame 并排序
mi_df = pd.DataFrame({
    "Feature": feature_names,
    "MI Score": mi_scores
}).sort_values(by="MI Score", ascending=False)

# 打印结果
print("\n=== Mutual Information Scores ===")
print(mi_df)

# ========== 4. 可视化绘图 ==========
plt.figure(figsize=(10, 6))
colors = plt.cm.coolwarm(np.linspace(0, 1, len(mi_df)))

plt.bar(mi_df["Feature"], mi_df["MI Score"], color=colors)
plt.xticks(rotation=45, ha='right', fontsize=18)
plt.yticks(fontsize=17)
plt.ylabel("Mutual Information (MI)", fontsize=17)

plt.tight_layout()
plt.savefig("MI.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nMI.png 已保存！")
# ---------------- MI排序输出 ----------------
sorted_idx = np.argsort(mi_scores)[::-1]

print("\nMI 平均特征重要性（降序）:")
for idx in sorted_idx:
    print(f"{feature_names[idx]}: {mi_scores[idx]:.6f}")

# -------- Feature_importance_heatmap --------
print("Feature_importance_heatmap:")
sorted_idx_asc = np.argsort(mi_scores)
ranking_array = sorted_idx_asc.tolist()

print("\nMI特征重要性排序数组（最不重要 → 最重要）:")
print(ranking_array)

# -------- GA排序 --------
print("GA排序：")

rank_array = np.empty(len(mi_scores), dtype=int)
rank_array[np.argsort(mi_scores)[::-1]] = np.arange(1, len(mi_scores) + 1)

print(rank_array.tolist())

ga_df = pd.DataFrame({
    "GA_rank": rank_array
})

ga_df.to_excel("GA_ranking_MI.xlsx", index=False)

print("GA排序已保存到 GA_ranking.xlsx")