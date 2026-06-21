# metrics.py
# 计算两个指标：平均成对 Spearman ρ、平均排名标准差

import numpy as np
from scipy.stats import spearmanr

# 计算所有方法两两之间的Spearman相关性系数的平均值ρ
def mean_pairwise_spearman(rank_matrix):
    """
    计算同一模型内所有归因方法之间的平均 Spearman 相关系数
    rank_matrix: (n_features, n_methods)
    返回: 平均 ρ (float)
    """
    n_methods = rank_matrix.shape[1]
    if n_methods < 2:
        return np.nan
    rhos = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            rho, _ = spearmanr(rank_matrix[:, i], rank_matrix[:, j])
            rhos.append(rho)
    return np.mean(rhos)
