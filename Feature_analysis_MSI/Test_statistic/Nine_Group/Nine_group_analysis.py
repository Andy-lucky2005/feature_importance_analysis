import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(1412)
# ========== 1. 读取数据 ==========
df = pd.read_excel("all_sort.xlsx", index_col=0)

# ========== 方法分组 ==========
method_groups = {
    # "group1": ["XGBoost"],
    "group1": [
"RF-TreeSHAP", "GBRT-TreeSHAP", "XGBoost-TreeSHAP",
"RF-KernelSHAP", "GBRT-KernelSHAP", "XGBoost-KernelSHAP",
"RF-permutation","GBRT-permutation","XGBoost-permutation",
"LR-KernelSHAP","MLP-KernelSHAP",
"LR-permutation", "MLP-permutation", "LR-coef", "RF","GBRT","XGBoost","MI",
             "pearson correlation","spearman correlation",
            "Formula-SHAP",
           "SVR-KernelSHAP", "SVR-PFI",


            "Eadh formula(Average of all data)",
           "Eadh formula(Feature mean)",
           "Eadh formula(Ranking average)",

        ],
    "group2": ["Eadh formula(Average of all data)",
               "Eadh formula(Feature mean)",
               "Eadh formula(Ranking average)",
               "Formula-SHAP"],
    "group3": ["GBRT-KernelSHAP", "GBRT-TreeSHAP", "GBRT-permutation", "GBRT"],
    "group4": ["XGBoost-permutation", "XGBoost-KernelSHAP", "XGBoost-TreeSHAP"],
    "group5": ["RF-permutation", "MLP-permutation", "MLP-KernelSHAP"],
    "group6": ["SVR-PFI", "SVR-KernelSHAP"],
    "group7": ["MI", "RF", "RF-KernelSHAP", "RF-TreeSHAP"],
    "group8":[ "spearman correlation", "pearson correlation"],
    "group9": ["LR-permutation", "LR-KernelSHAP", "LR-coef"],

}

# ========== feature标签 ==========
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

# ========== Lead advantage函数 ==========
def calculate_lead_advantage(rank_df, n_bootstrap=50000, confidence=0.95):
    features = rank_df.index.tolist()
    methods = rank_df.columns.tolist()
    results = {}
    for target_feature in features:
        lead_margins = []
        for method in methods:
            method_ranks = rank_df[method]
            other_ranks = method_ranks.drop(target_feature)
            best_other_rank = other_ranks.min()
            target_rank = method_ranks[target_feature]
            N = len(method_ranks)  # 特征数
            lead_margin = (best_other_rank - target_rank) / (N - 1)
            lead_margins.append(lead_margin)
        lead_margins = np.array(lead_margins)
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(
                lead_margins,
                size=len(lead_margins),
                replace=True
            )
            bootstrap_means.append(np.mean(sample))
        alpha = (1 - confidence) / 2
        lower_bound = np.percentile(bootstrap_means, alpha * 100)
        upper_bound = np.percentile(bootstrap_means, (1 - alpha) * 100)
        results[target_feature] = {
            "mean_advantage": np.mean(lead_margins),
            "ci_lower": lower_bound,
            "ci_upper": upper_bound
        }

    return results

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

# ========== 主循环：对每个group分析 ==========
for i, (group_name, group_methods) in enumerate(method_groups.items()):
    print("\n===============================")
    print("分析:", group_name)
    print("===============================")
    #绑定子图
    ax = axes[i]

    # 只保留数据中存在的方法
    valid_methods = [m for m in group_methods if m in df.columns]
    print("vaild_methods:",valid_methods)
    # if len(valid_methods) == 0:
    #     print("该组没有可用方法")
    #     continue
    df_group = df[valid_methods]
    total_methods = len(valid_methods)
    features = df_group.index.tolist()
    # ========== 最大排名 ==========
    max_rank = int(np.ceil(np.nanmax(df_group.values)))

    # ========== 排名统计 ==========
    rank_results = {
        rank: {feat: 0 for feat in features}
        for rank in range(1, max_rank + 1)
    }
    for method in valid_methods:
        col = df_group[method]
        for feature, rank_val in col.items():
            if pd.isna(rank_val):
                continue
            rank_int = int(round(rank_val))
            if 1 <= rank_int <= max_rank:
                rank_results[rank_int][feature] += 1

    # ========== 打印排名概率 ==========
    for rank in range(1, max_rank + 1):
        print(f"\n{group_name} 排名第 {rank}")
        sorted_items = sorted(
            rank_results[rank].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, count in sorted_items:
            prob = count / total_methods * 100
            print(feature, count, f"{prob:.2f}%")

    # ========== 领先优势 ==========
    lead_results = calculate_lead_advantage(df_group)
    print("\n领先优势分析")
    print("=" * 50)

    for feature, result in lead_results.items():
        print(f"{feature:20} | 平均领先: {result['mean_advantage']:6.3f} | "
              f"95%CI: [{result['ci_lower']:6.3f}, {result['ci_upper']:6.3f}]")

    # ========== 森林图数据 ==========
    forest_data = []
    for feature, res in lead_results.items():
        forest_data.append([
            feature_labels.get(feature, feature),
            res['mean_advantage'],
            res['ci_lower'],
            res['ci_upper']
        ])

    forest_df = pd.DataFrame(
        forest_data,
        columns=["feature", "mean", "ci_low", "ci_high"]
    )
    forest_df = forest_df.sort_values(
        by="mean",
        ascending=True
    ).reset_index(drop=True)

    # ========== 绘图 ==========
    # plt.figure(figsize=(8, 0.4 * len(forest_df) + 0.5))
    # plt.figure(figsize=(6,6))
    y_pos = np.arange(len(forest_df))
    ax.errorbar(
        forest_df["mean"],
        y_pos,
        xerr=[forest_df["mean"] - forest_df["ci_low"],
              forest_df["ci_high"] - forest_df["mean"]],
        fmt="o",
        markersize=3,
        capsize = 4,
        linewidth = 3,
        color="darkblue",
        ecolor="gray"
    )
    # ax.axvline(x=0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(forest_df["feature"], fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlim(0.25,-1.25)
    # ax = plt.gca()
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    # 所有图先设置相同xtick位置
    # ax.set_xticks(np.linspace(-1.2, 0.4, 4))  # [-1, -0.5, 0, 0.5, 1]
    ax.set_xticks([-1,-0.5, 0])  # [-1, -0.5, 0, 0.5, 1]
    if i >= 6:  # 最底行三张
        ax.tick_params(axis='x', labelsize=20)
    else:
        ax.set_xticklabels([])  # 只隐藏文字
        ax.tick_params(axis='x', length=0)

    ax.text(
        0.05, 0.96,  # 位置（左上角）
        f"{chr(65+i)}",
        transform=ax.transAxes,  # 相对坐标
        fontsize=14,
        fontweight='bold',
        va='top'
    )

plt.tight_layout()
plt.savefig("Figure5_combined.png", dpi=300, bbox_inches="tight")
plt.savefig("Figure5_combined.pdf", dpi=300, bbox_inches="tight")
plt.show()