import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 全局字体设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# 定义计算模型
def model(X):
    Xp_M, Xp_M_prime, IE_M, Hf_MO, Hf_M_prime_O, Hf_M_prime_M, Hsub_M, Hsub_M_prime, γ_M = X.T
    return (((0.0985 * γ_M * Xp_M_prime * (Hf_MO - Hsub_M)) / Xp_M) +
            ((-0.00306 * IE_M * (Hf_M_prime_M - Hsub_M - Hsub_M_prime)) / Hf_M_prime_O) * 16.0217 +
            16 * 0.0160217) * -1

def main():
    # 数据准备
    file_path = "../../Feature_Value/Science_feature_data.xlsx"  # 原始数据集路径
    data = pd.read_excel(file_path, header=0)

    # 提取特征
    X = data.iloc[:, 2:16]
    X = X.drop(columns=['Eg_M\'O (eV)', 'r_M (Å)', 'r_M\' (Å)', 'Nws_M (d.u.)', 'IE_M\' (eV)'])
    X = np.array(X, dtype=float)

    # 特征名称
    feature_names = [
        r"$\chi_p^M$",
        r"$\chi_p^{M'}$",
        r"$IE^M$",
        r"$\Delta H_f^{MO}$",
        r"$\Delta H_f^{M'O}$",
        r"$\Delta H_f^{M'M}$",
        r"$\Delta H_{sub}^M$",
        r"$\Delta H_{sub}^{M'}$",
        r"$\gamma^M$"
    ]

    # 输出目录
    output_dir = "../../Formula_research/shap_plots"
    os.makedirs(output_dir, exist_ok=True)

    # SHAP 解释器与计算
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # 打印 SHAP 值统计
    print("SHAP values matrix shape:", shap_values.values.shape)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    print("Mean(|SHAP|) per feature:")
    for fname, val in sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True):
        print(f"  {fname}: {val:.5f}")
    # 保存【全局柱状图】
    plt.figure(figsize=(2.5, 8))  # 宽度缩小，高度加大
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    bar_path = os.path.join(output_dir, "shap_bar_plot.png")
    plt.savefig(bar_path, dpi=300)  # dpi保持原来
    plt.show()
    plt.close()

    # 生成并保存【SHAP概要图】
    plt.figure(figsize=(4, 12))  # 宽度缩小，高度加大
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    summary_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.savefig(summary_path, dpi=300)  # dpi保持原来
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
