import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# 读取数据
file_path = "../Feature_Value/Science_feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

# 目标列（Eadh）
y = data.iloc[:, 1].values  # 连续目标
# 特征列（第3列到第16列）
X = data.iloc[:, 2:16]
feature_names = [
    r"$\chi_p^M$", r"$\chi_p^{M'}$", r"$IE^M$", r"$IE^{M'}$", r"$r^M$", r"$r^{M'}$",
    r"$\Delta H_f^{MO}$", r"$\Delta H_f^{M'O}$", r"$\Delta H_f^{M'M}$",
    r"$\Delta H_{sub}^M$", r"$\Delta H_{sub}^{M'}$", r"$\gamma^M$", r"$n_{ws}^M$", r"$E_g^{M'O}$"
]

# 计算互信息 (MI)
mi_scores = mutual_info_regression(X, y, random_state=1412)  # 可加: n_neighbors=3

# 转成 DataFrame 并排序
mi_df = pd.DataFrame({'Feature': feature_names, 'MI Score': mi_scores}) \
         .sort_values(by='MI Score', ascending=False)
print(mi_df)

# 可视化
plt.figure(figsize=(10, 6))
colors = plt.cm.coolwarm(np.linspace(0, 1, len(mi_df)))
plt.bar(mi_df['Feature'], mi_df['MI Score'], color=colors)
plt.xticks(rotation=45, ha='right', rotation_mode="anchor",fontsize=21)
plt.yticks(fontsize = 17)
plt.ylabel("Mutual Information (MI)",fontsize= 17)
# plt.title("Feature-wise Mutual Information with Target")
plt.tight_layout()
plt.savefig("MI.png", dpi=300, bbox_inches="tight")
plt.show()
