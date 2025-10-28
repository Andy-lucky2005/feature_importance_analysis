import pandas as pd
import numpy as np

# 读取偏导数结果
file_path_partial = "../AGM/partials_results.xlsx"
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

# 取需要的列并转为绝对值
abs_data = data_partial[partial_columns].abs()

# 对每一行进行排名（1=最大，9=最小）
ranked_data = abs_data.rank(axis=1, method="min", ascending=False).astype(int)
print(ranked_data)

rank_sum = ranked_data.sum(axis = 0)
print(rank_sum)
# 为了区分，保留原始绝对值和排名
output_data = pd.concat([abs_data.add_suffix("_abs"), ranked_data.add_suffix("_rank")], axis=1)

# 保存到 Excel
output_file = "partials_ranked.xlsx"
output_data.to_excel(output_file, index=False)

print(f"结果已保存到 {output_file}")

