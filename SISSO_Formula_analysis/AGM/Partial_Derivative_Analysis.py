import pandas as pd
import numpy as np

# 读取偏导数结果
file_path_partial = "partials_results.xlsx"
data_partial = pd.read_excel(file_path_partial, header=0)

# 偏导数列名
partial_columns = [
    "Hf_MO_partial",
    "Hsub_M_partial",
    "Hf_M_prime_M_partial",
    "Hsub_M_prime_partial",
    "γ_M_partial",
    "Xp_M_partial",
    "Xp_M_prime_partial",
    "IE_M_partial",
    "Hf_M_prime_O_partial"
]

# 提取需要的列
abs_data = data_partial[partial_columns].abs()

# 计算每一列的总和与平均值
col_sums = abs_data.sum()
col_means = abs_data.mean()

# 按从大到小排序
sorted_sums = col_sums.sort_values(ascending=False)
sorted_means = col_means.sort_values(ascending=False)

# 打印结果
print("===== 偏导数列的总和 =====")
print(sorted_sums)

print("\n===== 偏导数列的平均值 =====")
print(sorted_means)