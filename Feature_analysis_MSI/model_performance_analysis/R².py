import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# 数据
models = ['RF', 'GBRT', 'XGBoost', 'LR', 'MLP', 'SVR','Formula[28]']
mae_values = [0.8628023451069552,
              0.894294539235849,
              0.874114668356474,
              0.7697133174882456,
              0.9086517925062634,
              0.900270204865867,
              0.8964]

# 按 MAE 值从大到小排序
sorted_indices = np.argsort(mae_values)[::-1]  # 获取排序后的索引
sorted_models = [models[i] for i in sorted_indices]  # 排序后的模型名称
sorted_mae_values = [mae_values[i] for i in sorted_indices]  # 排序后的 MAE 值

# 自定义颜色（按排序后的顺序使用）
# 自定义颜色（按你给的顺序）
custom_colors = ['#EA9393', '#FFBF86', '#FFDB99', '#95CF95', '#8FBBD9', '#C9B3DE','#E599E5']
border_colors = ['#D83031', '#FF8D28', '#FFAE1C', '#39A639', '#257AB6', '#9B72C2', '#C109C1']
# 创建柱状图
plt.figure(figsize=(9, 8))
bars = plt.bar(sorted_models, sorted_mae_values, color=custom_colors,edgecolor=border_colors)

# 去掉上边框和右边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加标题和标签
plt.title('', fontsize=16)
plt.xlabel('', fontsize=20)
plt.ylabel('Average R²', fontsize=24)

# 美化刻度
plt.xticks(rotation=45, ha='right', rotation_mode="anchor",fontsize=24)  # 优化x轴标签显示
plt.yticks(fontsize=24)  # 优化y轴标签显示

# 自动调整布局并保存
plt.tight_layout()
plt.savefig('R².png', dpi=300)
plt.show()
