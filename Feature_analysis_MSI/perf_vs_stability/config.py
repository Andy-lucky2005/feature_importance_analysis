# 配置参数、模型性能字典、模型-方法映射

# config.py
# 配置文件：路径、模型性能、方法映射等

import os

# 文件路径
DATA_FILE = "all_sort.xlsx"  # 请确保文件在当前目录，或修改为绝对路径

# 模型性能（10折交叉验证 MAE）—— 根据论文 Figure 1 填写，请替换为实际值
# 注意：如果某些模型没有性能数据，可以注释掉或设为 None
MODEL_PERFORMANCE = {
    "RF": 0.1870353575169314,        # Random Forest
    "GBRT":  0.15971294204210046,      # Gradient Boosting Regression Tree
    "XGBoost":  0.18230688077998997,   # XGBoost
    "LR":  0.2560138487148428,        # Linear Regression
    "MLP": 0.15106691741058792,       # Multi-Layer Perceptron
    "SVR":  0.1666361057678659,       # Support Vector Regression
    "Formula": 0.1844,
}

# 模型家族及其对应的归因方法列名
# 注意：每个模型至少需要 2 种方法才能计算内部一致性
MODEL_METHOD_KEYWORDS = {
    "RF": [
        "RF-TreeSHAP",
        "RF-KernelSHAP",
        "RF-permutation",
        "RF"
    ],
    "GBRT": [
        "GBRT-TreeSHAP",
        "GBRT-KernelSHAP",
        "GBRT-permutation",
        "GBRT"                   # MDI
    ],
    "XGBoost": [
        "XGBoost-TreeSHAP",
        "XGBoost-KernelSHAP",
        "XGBoost-permutation",
        "XGBoost"                # MDI
    ],
    "LR": [
        "LR-KernelSHAP",
        "LR-permutation",
        "LR-coef"
    ],
    "MLP": [
        "MLP-KernelSHAP",
        "MLP-permutation"
    ],
    "SVR": [
        "SVR-KernelSHAP",
        "SVR-PFI"
    ],
    "Formula": [
        "Formula-SHAP",
        "Eadh formula(Average of all data)",
        "Eadh formula(Feature mean)",
        "Eadh formula(Ranking average)",
    ]
}

# 特征名称（与 Excel 第一列保持一致）
FEATURE_NAMES = [
    "Xp_M", "Xp_M'", "IE_M (eV)", "IE_M' (eV)", "r_M (Å)", "r_M' (Å)",
    "Hf_MO (eV M)", "Hf_M'O (eV M')", "Hf_M'(M) (eV)", "Hsub_M (eV)",
    "Hsub_M' (eV)", "γ_M (J/m^2)", "Nws_M (d.u.)", "Eg_M'O (eV)"
]
