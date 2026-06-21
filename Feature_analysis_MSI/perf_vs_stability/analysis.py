# analysis.py
# 主分析流程：计算各模型指标、与 MAE 的相关性、输出表格

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from config import MODEL_PERFORMANCE
from data_loader import load_rank_matrix, extract_model_rankings
from metrics import mean_pairwise_spearman


def run_analysis():
    # 1. 加载数据
    df_rank = load_rank_matrix()

    # 2. 提取模型排名矩阵
    model_matrices = extract_model_rankings(df_rank)

    # 3. 计算每个模型的指标
    results = []
    for model, mat in model_matrices.items():
        if model not in MODEL_PERFORMANCE:
            print(f"跳过模型 {model}：缺少 MAE 数据")
            continue
        mae = MODEL_PERFORMANCE[model]
        mean_rho = mean_pairwise_spearman(mat)
        results.append({
            "Model": model,
            "MAE": mae,
            "Mean_Spearman_ρ": mean_rho,
        })

    # 4. 转换为 DataFrame 并保存
    df_results = pd.DataFrame(results)
    print("\n=== 各模型内部稳定性指标 ===")
    print(df_results.to_string(index=False))

    csv_path = f"model_instability_metrics.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n表格已保存至: {csv_path}")

    # 5. 计算与 MAE 的 Spearman 相关性
    if len(df_results) >= 3:
        rho_rho, p_rho = spearmanr(df_results["MAE"], df_results["Mean_Spearman_ρ"])
        print("\n=== 与 MAE 的相关性 ===")
        print(f"MAE vs Mean_Spearman_ρ : ρ = {rho_rho:.3f}, p = {p_rho:.4f}")
    else:
        print("警告：有效模型数量不足，无法计算相关性")

    return df_results