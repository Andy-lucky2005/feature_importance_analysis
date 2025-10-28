import pandas as pd

from sklearn.preprocessing import StandardScaler
# 1️读取数据
file_path = "../Science_feature_data.xlsx"
df = pd.read_excel(file_path)

# 2️基础清洗（可选）
df = df.dropna(how='all').drop_duplicates()
df = df.fillna(0)  # 或者用其他方法填充缺失值

feature_labels = {
    "Xp_M": r"$\chi_p^M$",
    "Xp_M'": r"$\chi_p^{M'}$",
    "IE_M (eV)": r"$IE^M$",
    "IE_M' (eV)": r"$IE^{M'}$",
    "r_M (Å)": r"$r^M$",
    "r_M' (Å)": r"$r^{M'}$",
    "Hf_MO (eV M)": r"$\Delta H_f^{MO}$",
    "Hf_M'O (eV M')": r"$\Delta H_f^{M'O}$",
    "Hf_M'(M) (eV)": r"$\Delta H_f^{M'M}$",
    "Hsub_M (eV)": r"$\Delta H_{sub}^M$",
    "Hsub_M' (eV)": r"$\Delta H_{sub}^{M'}$",
    "γ_M (J/m^2)": r"$\gamma^M$",
    "Nws_M (d.u.)": r"$n_{ws}^M$",
    "Eg_M'O (eV)": r"$E_g^{M'O}$"
}

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()     # 均值为0，方差为1
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
scaled_path = "scaled_data.xlsx"
df.to_excel(scaled_path, index=False)
print(f"Scaling 已完成，保存为：{scaled_path}")
