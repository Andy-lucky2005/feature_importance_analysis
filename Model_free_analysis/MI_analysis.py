import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

# 全局字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 读取数据 ==========
file_path = "../Feature_Value/Science_feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

# y: 目标值（Eadh）
y = data.iloc[:, 1].values

# X: 特征（第3~16列）
X = data.iloc[:, 2:16]

feature_names = [
    r"$\chi_p^M$", r"$\chi_p^{M'}$", r"$IE^M$", r"$IE^{M'}$", r"$r^M$", r"$r^{M'}$",
    r"$\Delta H_f^{MO}$", r"$\Delta H_f^{M'O}$", r"$\Delta H_f^{M'M}$",
    r"$\Delta H_{sub}^M$", r"$\Delta H_{sub}^{M'}$", r"$\gamma^M$", r"$n_{ws}^M$", r"$E_g^{M'O}$"
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

# 保存到 CSV
# mi_df.to_csv("MI_scores.csv", index=False)
# print("\nMI_scores.csv 已保存！")

# ========== 4. 可视化绘图 ==========
plt.figure(figsize=(10, 6))
colors = plt.cm.coolwarm(np.linspace(0, 1, len(mi_df)))

plt.bar(mi_df["Feature"], mi_df["MI Score"], color=colors)
plt.xticks(rotation=45, ha='right', fontsize=21)
plt.yticks(fontsize=17)
plt.ylabel("Mutual Information (MI)", fontsize=17)

plt.tight_layout()
plt.savefig("MI.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nMI.png 已保存！")
