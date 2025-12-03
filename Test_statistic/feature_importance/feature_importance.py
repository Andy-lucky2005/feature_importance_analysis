import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# ========== 1. 读取数据 ==========
df = pd.read_excel("all_sort.xlsx", index_col=0)

# 可选：查看前几行，调试用
# print(df.head())
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

features = df.index.tolist()
methods = df.columns.tolist()
total_methods = len(methods)

# ========== 2. 确定最大排名（稳健方式） ==========
# 忽略 NaN 后取最大，并向上取整成整数（假如你期望排名为整数）
max_value = np.nanmax(df.values)  # np.nanmax 会忽略 NaN
if np.isnan(max_value):
    raise ValueError("数据中全是 NaN 或无法识别的值，请检查 Excel 内容")

# 如果 max_value 是 14.0 -> 转为 14
max_rank = int(np.ceil(max_value))

# ========== 3. 初始化计数字典 ==========
rank_results = {rank: {feat: 0 for feat in features} for rank in range(1, max_rank + 1)}
print(rank_results)

# ========== 4. 计数：逐列扫描每个 feature 的排名 ==========
for method in methods:
    col = df[method]
    for feature, rank_val in col.items():
        # 跳过缺失值
        if pd.isna(rank_val):
            continue

        # 如果排名是浮点型但代表整数（如 1.0），安全转换
        try:
            rank_int = int(round(float(rank_val)))
        except Exception:
            # 如果不能转换，跳过或记录异常
            continue

        # 只统计在 1..max_rank 范围内的排名
        if 1 <= rank_int <= max_rank:
            rank_results[rank_int][feature] += 1

# ========== 5. 输出结果（按排名位置） ==========
for rank in range(1, max_rank + 1):
    print(f"\n排名第 {rank}（Top {rank}）的概率")
    print("特征\t出现次数\t概率")

    sorted_items = sorted(rank_results[rank].items(), key=lambda x: x[1], reverse=True)

    for feature, count in sorted_items:
        prob = count / total_methods * 100
        print(f"{feature}\t{count}\t{prob:.2f}%")



def calculate_lead_advantage(rank_df, n_bootstrap=500000, confidence=0.95):
    """
    计算每个特征相对于最强竞争者的领先优势
    """
    features = rank_df.index.tolist()
    methods = rank_df.columns.tolist()

    results = {}

    for target_feature in features:
        lead_margins = []

        # 对每个方法计算领先幅度
        for method in methods:
            # 获取当前方法的所有排名
            method_ranks = rank_df[method]

            # 排除目标特征本身
            other_ranks = method_ranks.drop(target_feature)

            # 找到剩余特征中的最佳排名（数值最小）
            best_other_rank = other_ranks.min()
            #  获取最小排名 即最重要的特征
            # beat_other_rank = other_ranks.min()
            target_rank = method_ranks[target_feature]

            # 计算领先幅度：其他特征的最佳排名 - 目标特征的排名
            # 正数表示目标特征领先，负数表示落后
            lead_margin = best_other_rank - target_rank
            lead_margins.append(lead_margin)

        # 转换为numpy数组便于重抽样
        lead_margins = np.array(lead_margins)

        # 重抽样计算置信区间
        bootstrap_means = []
        for _ in range(n_bootstrap):
            # 有放回抽样
            sample = np.random.choice(lead_margins, size=len(lead_margins), replace=True)
            bootstrap_means.append(np.mean(sample))

        # 计算置信区间
        alpha = (1 - confidence) / 2
        lower_bound = np.percentile(bootstrap_means, alpha * 100)
        upper_bound = np.percentile(bootstrap_means, (1 - alpha) * 100)
        mean_advantage = np.mean(lead_margins)

        # 统计领先、持平、落后的次数
        # n_leading = np.sum(lead_margins > 0)
        # n_tie = np.sum(lead_margins == 0)
        # n_trailing = np.sum(lead_margins < 0)

        results[target_feature] = {
            'mean_advantage': mean_advantage,
            'ci_lower': lower_bound,
            'ci_upper': upper_bound,
            # 'n_leading': n_leading,
            # 'n_tie': n_tie,
            # 'n_trailing': n_trailing,
            # 'leading_prob': n_leading / len(methods),
            'all_lead_margins': lead_margins
        }

    return results


# 应用分析
lead_results = calculate_lead_advantage(df)

# 打印结果
print("各特征的领先优势分析:")
print("=" * 50)
for feature, result in lead_results.items():
    print(f"{feature:20} | 平均领先: {result['mean_advantage']:6.3f} | "
          f"95%CI: [{result['ci_lower']:6.3f}, {result['ci_upper']:6.3f}] | ")
          # f"领先概率: {result['leading_prob']:6.1%} | ")
          # f"领先/持平/落后: {result['n_leading']:2d}/{result['n_tie']:2d}/{result['n_trailing']:2d}")



# ========== 1. 整理森林图数据 ==========
forest_data = []

for feature, res in lead_results.items():
    forest_data.append([
        feature_labels.get(feature, feature),  # 使用LaTeX标签
        res['mean_advantage'],
        res['ci_lower'],
        res['ci_upper']
    ])

forest_df = pd.DataFrame(forest_data, columns=["feature", "mean", "ci_low", "ci_high"])

# 按平均领先值从大到小排序
forest_df = forest_df.sort_values(by="mean", ascending=True).reset_index(drop=True)

# --- 颜色列表（从上到下） ---
ci_colors = [

"#EF04F7",
"#910EF2",
"#360EF2",
"#0E3DF2",
"#029EED",

"#16E1D7",

"#008100",
"#02D40D",


"#FDD000",
"#F99302",
"#F29601",
"#F95F02",
"#FF3401",
"#FF0A00",
]


# ========== 2. 绘制森林图 ==========
plt.figure(figsize=(8, 0.4 * len(forest_df) + 0.5))  # 行间距更紧

y_pos = np.arange(len(forest_df))

# 遍历每一行，分别绘制（让颜色独立）
plt.errorbar(
    forest_df["mean"],
    y_pos, xerr=[forest_df["mean"] - forest_df["ci_low"],
                 forest_df["ci_high"] - forest_df["mean"]],
    fmt="o",
    markersize=6,
    capsize=5,
    linewidth=2,
    color="darkblue",
    ecolor="gray", )
# for i in range(len(forest_df)):
#     mean_val = forest_df["mean"].iloc[i]
#     low = forest_df["ci_low"].iloc[i]
#     high = forest_df["ci_high"].iloc[i]
#
#     # xerr 转成 (2,1) 形状
#     xerr = np.array([
#         [mean_val - low],
#         [high - mean_val]
#     ])
#
#     plt.errorbar(
#         mean_val,
#         y_pos[i],
#         xerr=xerr,
#         fmt="o",
#         markersize=6,
#         capsize=3,
#         linewidth=3.3,
#         color=ci_colors[i],      # 点颜色
#         ecolor=ci_colors[i],     # 误差线颜色
#         markeredgecolor="#100A0A",  # 点的外边框颜色
#         markeredgewidth=1.4,        # 外边框宽度
#         alpha=0.7,
#     )


# 参考竖线（如需要可启用）
plt.axvline(x=0, color="red", linestyle="--", linewidth=1)

# y 轴标签
plt.yticks(y_pos, forest_df["feature"], fontsize=14)
plt.gca().invert_yaxis()

# 边距更紧凑
plt.xlabel("", fontsize=28)
plt.title("", fontsize=28)

# 更轻的网格线
plt.grid(axis="x", linestyle="--", alpha=0.25)
# 设置 x 轴范围与刻度
plt.xlim(2, -14 )
plt.xticks(np.arange(-14, 3, 2),fontsize = 16)  # 从 -14 到 2，每 2 一格

plt.tight_layout()

plt.savefig("forest_plot.png", dpi=300, bbox_inches="tight")
plt.savefig("forest_plot.pdf", dpi=300, bbox_inches="tight")
plt.show()
