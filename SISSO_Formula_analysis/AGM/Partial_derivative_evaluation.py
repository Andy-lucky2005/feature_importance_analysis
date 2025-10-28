import pandas as pd
import numpy as np

# 读取数据
# file_path = "../Chart/Science_feature_data.xlsx"
file_path = "../../Feature_Value/Science_feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

# 提取相关特征列
γ_M = data['γ_M (J/m^2)']
Xp_M_prime = data["Xp_M'"]  # 氧化物中金属电负性
Xp_M = data['Xp_M']  # Xp_M

Hf_MO = data['Hf_MO (eV M)']  # Hf_MO
Hsub_M = data['Hsub_M (eV)']  # Hsub_M

IE_M = data['IE_M (eV)']  # IE_M
Hf_M_prime_O = data['Hf_M\'O (eV M\')']  # Hf_M'O Hf_M'O (eV M')

Hf_M_prime_M = data['Hf_M\'(M) (eV)']  # Hf_M'(M)
Hsub_M_prime = data['Hsub_M\' (eV)']  # Hsub_M'

# 常数
c1 = 0.0985
c2 = -0.00306

# -------- 偏导数公式 --------
partials = pd.DataFrame({
    "Hf_MO_partial": abs((c1 * γ_M * Xp_M_prime) / Xp_M),
    "Hsub_M_partial": abs((-c1 * γ_M * Xp_M_prime / Xp_M) - 16.0217 * (c2 * IE_M / Hf_M_prime_O)),
    "Hf_M_prime_M_partial": abs(16.0217 * c2 * IE_M / Hf_M_prime_O),
    "Hsub_M_prime_partial": abs(-16.0217 * c2 * IE_M / Hf_M_prime_O),
    "γ_M_partial": abs(c1 * Xp_M_prime * (Hf_MO - Hsub_M) / Xp_M),
    "Xp_M_partial": abs((-c1 * γ_M * Xp_M_prime * (Hf_MO - Hsub_M)) / (Xp_M * Xp_M)),
    "Xp_M_prime_partial": abs(c1 * γ_M * (Hf_MO - Hsub_M) / Xp_M),
    "IE_M_partial": abs(16.0217 * c2 * (Hf_M_prime_M - Hsub_M - Hsub_M_prime) / Hf_M_prime_O),
    "Hf_M_prime_O_partial": abs(16.0217 * c2 * IE_M * (Hsub_M + Hsub_M_prime - Hf_M_prime_M) / (Hf_M_prime_O * Hf_M_prime_O))
})



# -------- 计算每个特征的偏导数总和 --------
total_partials = {column: partials[column].sum() for column in partials.columns}

# -------- 按照总和大小排序 --------
sorted_partials = sorted(total_partials.items(), key=lambda x: x[1], reverse=True)

# -------- 输出排序后的结果 --------
print("偏导数总和按大小排序：")
for feature, total in sorted_partials:
    print(f"{feature}: {total}")

# -------- 保存结果 --------
output_path = "partials_results.xlsx"
partials.to_excel(output_path, index=True)
print(f"偏导数结果已保存到: {output_path}")