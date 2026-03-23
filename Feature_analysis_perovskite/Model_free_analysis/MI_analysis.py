import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

# 全局字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

file_path = "../perovskite_dataset/feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

# y：第2列（目标变量），X：第3~16列（14个特征）
y = data.iloc[:, 1]
X = data.iloc[:, 2:74]
# 特征名
feature_names = [
    r"$r_A^{\mathrm{atom}}$",
    r"$r_B^{\mathrm{atom}}$",
    r"$r_X^{\mathrm{atom}}$",
    r"$r_A^{\mathrm{ion}}$",
    r"$r_B^{\mathrm{ion}}$",
    r"$r_X^{\mathrm{ion}}$",
    r"$IE_A$",
    r"$IE_B$",
    r"$IE_X$",
    r"$EA_A$",
    r"$EA_B$",
    r"$EA_X$",
    r"$\chi_A$",
    r"$\chi_B$",
    r"$\chi_X$",
    r"$N_A$",
    r"$N_B$",
    r"$N_X$",
    r"$M_A$",
    r"$M_B$",
    r"$M_X$",
    r"$E_{\mathrm{HOMO}}$",
    r"$E_{\mathrm{LUMO}}$",
    r"$\Delta E_{\mathrm{AO}}$",
    r"$\mu_{\mathrm{bond}}$",
    r"$\sigma^2_{\mathrm{bond}}$",
    r"$CV_{\mathrm{bond}}$",
    r"$\Delta \mu_{X-B-X,\mathrm{bond}}$",
    r"$\Delta \sigma^2_{X-B-X,\mathrm{bond}}$",
    r"$\rho_{X-B-X,\mathrm{bond}}$",
    r"$\sigma^2_{\rho,X-B-X,\mathrm{bond}}$",
    r"$\Delta \mu_{EA-B-X,\mathrm{bond}}$",
    r"$\Delta \sigma^2_{EA-B-X,\mathrm{bond}}$",
    r"$\rho_{EA-B-X,\mathrm{bond}}$",
    r"$\sigma^2_{\rho,EA-B-X,\mathrm{bond}}$",
    r"$\Delta CV_{X-B-X,\mathrm{bond}}$",
    r"$CV_{\rho,X-B-X,\mathrm{bond}}$",
    r"$\Delta CV_{EA-B-X,\mathrm{bond}}$",
    r"$CV_{\rho,EA-B-X,\mathrm{bond}}$",
    r"$(\chi_B - \chi_X)$",
    r"$\chi_B / \chi_X$",
    r"$\frac{\chi_B - \chi_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    r"$\frac{\chi_B / \chi_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    r"$\chi_B / r_B^{\mathrm{ion}}$",
    r"$\chi_X / r_X^{\mathrm{ion}}$",
    r"$\frac{\chi_B}{r_B^{\mathrm{ion}}} + \frac{\chi_X}{r_X^{\mathrm{ion}}}$",
    r"$\frac{\chi_B}{r_B^{\mathrm{ion}}} - \frac{\chi_X}{r_X^{\mathrm{ion}}}$",
    r"$\frac{\chi_B}{r_B^{\mathrm{ion}}} \cdot \frac{\chi_X}{r_X^{\mathrm{ion}}}$",
    r"$\frac{\chi_B / r_B^{\mathrm{ion}}}{\chi_X / r_X^{\mathrm{ion}}}$",
    r"$(EA_B - EA_X)$",
    r"$EA_B / EA_X$",
    r"$\frac{EA_B - EA_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    r"$\frac{EA_B / EA_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    r"$EA_B / r_B^{\mathrm{ion}}$",
    r"$EA_X / r_X^{\mathrm{ion}}$",
    r"$\frac{EA_B}{r_B^{\mathrm{ion}}} + \frac{EA_X}{r_X^{\mathrm{ion}}}$",
    r"$\frac{EA_B}{r_B^{\mathrm{ion}}} - \frac{EA_X}{r_X^{\mathrm{ion}}}$",
    r"$\frac{EA_B}{r_B^{\mathrm{ion}}} \cdot \frac{EA_X}{r_X^{\mathrm{ion}}}$",
    r"$\frac{EA_B / r_B^{\mathrm{ion}}}{EA_X / r_X^{\mathrm{ion}}}$",
    r"$r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}$",
    r"$r_B^{\mathrm{ion}} / r_X^{\mathrm{ion}}$",
    r"$\frac{r_A^{\mathrm{ion}} + r_X^{\mathrm{ion}}}{1.414(r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}})}$",
    r"$\log(\sigma^2_{\mathrm{bond}})$",
    r"$\log(CV_{\mathrm{bond}})$",
    r"$\log(\Delta \sigma^2_{X-B-X,\mathrm{bond}})$",
    r"$\log(\Delta \sigma^2_{EA-B-X,\mathrm{bond}})$",
    r"$\log(\sigma^2_{\rho,X-B-X,\mathrm{bond}})$",
    r"$\log(\sigma^2_{\rho,EA-B-X,\mathrm{bond}})$",
    r"$\log(\Delta CV_{X-B-X,\mathrm{bond}})$",
    r"$\log(CV_{\rho,X-B-X,\mathrm{bond}})$",
    r"$\log(\Delta CV_{EA-B-X,\mathrm{bond}})$",
    r"$\log(CV_{\rho,EA-B-X,\mathrm{bond}})$"
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