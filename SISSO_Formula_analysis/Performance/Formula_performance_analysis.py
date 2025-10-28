# 说明： 1 J/m² = 62.415 meV/A²
# 1eV/A²=1000meV/A²
# 1 meV/A² = 0.0160217 J/m²

import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_absolute_error, r2_score

# 读取数据
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



# 公式常数 J/M²
a1 = 9.85 * 10**-2  # 常数9.85 * 10^-2
a2 = -3.06 * 10**-3  # 常数 -3.06 * 10^-3
constant = 16 * 0.0160217  # 常数16

# 计算Eadh（单位 mev/A²）
Eadh_calculated = ((a1 * γ_M * Xp_M_prime * (Hf_MO - Hsub_M)) / Xp_M) + ((a2 * IE_M * (Hf_M_prime_M - Hsub_M - Hsub_M_prime)) / Hf_M_prime_O) * 16.0217 + constant
data['Eadh_calculated'] = abs(Eadh_calculated)

# 计算 MAE 和 R²
# 目标值 Eadh
y_true = data['Eadh_Experimental_Data (J/m2)']  # 目标值 Eadh
y_pred = data['Eadh_calculated'] # 预测值


# 计算 MAE
mae = mean_absolute_error(y_true, y_pred)

# 计算 R²
r_squared = r2_score(y_true, y_pred)
print(y_true)
# 输出结果
print(f"模型的 MAE: {mae:.4f}")
print(f"模型的 R²: {r_squared:.4f}")

print(y_pred)
# 计算模型执行时间
start_time = time.time()
end_time = time.time()
execution_time = end_time - start_time
print(f"模型计算执行时间: {execution_time:.2f}秒")
