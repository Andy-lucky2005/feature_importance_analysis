import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# 数据
models = ['RF', 'GBRT', 'XGBoost', 'LR', 'MLP', 'SVR','Formula']
mae_values = [0.1870353575169314,
              0.15971294204210046,
              0.18230688077998997,
              0.2560138487148428,
              0.15106691741058792,
              0.1666361057678659,
              0.1844]

# 按 MAE 值从大到小排序
sorted_indices = np.argsort(mae_values)[::-1]
sorted_models = [models[i] for i in sorted_indices]
sorted_mae_values = [mae_values[i] for i in sorted_indices]

# 自定义颜色（按你给的顺序）
custom_colors = ['#EA9393', '#FFBF86', '#FFDB99', '#95CF95', '#8FBBD9', '#C9B3DE','#E599E5']
border_colors = ['#D83031', '#FF8D28', '#FFAE1C', '#39A639', '#257AB6', '#9B72C2', '#C109C1']
# 画图
fig, ax = plt.subplots(figsize=(9, 8))
bars = ax.bar(sorted_models, sorted_mae_values, color=custom_colors,edgecolor=border_colors,)

# 去掉上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加标题和坐标轴标签
ax.set_title('', fontsize=16)
ax.set_xlabel('', fontsize=20)
ax.set_ylabel('Average MAE (meV/Å²)', fontsize=24)

# 美化坐标轴刻度
plt.xticks(rotation=45, ha='right', rotation_mode="anchor",fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()

# 保存与展示
plt.savefig('MAE.png', dpi=300, bbox_inches='tight')
plt.show()
