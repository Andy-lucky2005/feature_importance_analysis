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
    # "bandgap": r"$E_g$",
    "A_r_atom": r"$r_A^{\mathrm{atom}}$",
    "B_r_atom": r"$r_B^{\mathrm{atom}}$",
    "X_r_atom": r"$r_X^{\mathrm{atom}}$",
    "A_r_ion": r"$r_A^{\mathrm{ion}}$",
    "B_r_ion": r"$r_B^{\mathrm{ion}}$",
    "X_r_ion": r"$r_X^{\mathrm{ion}}$",
    "A_ie": r"$IE_A$",
    "B_ie": r"$IE_B$",
    "X_ie": r"$IE_X$",
    "A_ea": r"$EA_A$",
    "B_ea": r"$EA_B$",
    "X_ea": r"$EA_X$",
    "A_x": r"$\chi_A$",
    "B_x": r"$\chi_B$",
    "X_x": r"$\chi_X$",
    "A_N": r"$N_A$",
    "B_N": r"$N_B$",
    "X_N": r"$N_X$",
    "A_M": r"$M_A$",
    "B_M": r"$M_B$",
    "X_M": r"$M_X$",
    "HOMO": r"$E_{\mathrm{HOMO}}$",
    "LUMO": r"$E_{\mathrm{LUMO}}$",
    "gap_AO": r"$\Delta E_{\mathrm{AO}}$",
    "bond_mean": r"$\mu_{\mathrm{bond}}$",
    "bond_var": r"$\sigma^2_{\mathrm{bond}}$",
    "bond_cv": r"$CV_{\mathrm{bond}}$",
    "delta_X_bx_bond_mean": r"$\Delta \mu_{X-B-X,\mathrm{bond}}$",
    "delta_X_bx_bond_var": r"$\Delta \sigma^2_{X-B-X,\mathrm{bond}}$",
    "ratio_X_bx_bond_mean": r"$\rho_{X-B-X,\mathrm{bond}}$",
    "ratio_X_bx_bond_var": r"$\sigma^2_{\rho,X-B-X,\mathrm{bond}}$",
    "delta_ea_bx_bond_mean": r"$\Delta \mu_{EA-B-X,\mathrm{bond}}$",
    "delta_ea_bx_bond_var": r"$\Delta \sigma^2_{EA-B-X,\mathrm{bond}}$",
    "ratio_ea_bx_bond_mean": r"$\rho_{EA-B-X,\mathrm{bond}}$",
    "ratio_ea_bx_bond_var": r"$\sigma^2_{\rho,EA-B-X,\mathrm{bond}}$",
    "delta_X_bx_bond_cv": r"$\Delta CV_{X-B-X,\mathrm{bond}}$",
    "ratio_X_bx_bond_cv": r"$CV_{\rho,X-B-X,\mathrm{bond}}$",
    "delta_ea_bx_bond_cv": r"$\Delta CV_{EA-B-X,\mathrm{bond}}$",
    "ratio_ea_bx_bond_cv": r"$CV_{\rho,EA-B-X,\mathrm{bond}}$",
    "(B_x - X_x)": r"$(\chi_B - \chi_X)$",
    "(B_x / X_x)": r"$\chi_B / \chi_X$",
    "(B_x - X_x)/(B_r_ion + X_r_ion)": r"$\frac{\chi_B - \chi_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    "(B_x / X_x)/(B_r_ion + X_r_ion)": r"$\frac{\chi_B / \chi_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    "(B_x/B_r_ion)": r"$\chi_B / r_B^{\mathrm{ion}}$",
    "(X_x/X_r_ion)": r"$\chi_X / r_X^{\mathrm{ion}}$",
    "(B_x/B_r_ion)+(X_x/X_r_ion)": r"$\frac{\chi_B}{r_B^{\mathrm{ion}}} + \frac{\chi_X}{r_X^{\mathrm{ion}}}$",
    "(B_x/B_r_ion)-(X_x/X_r_ion)": r"$\frac{\chi_B}{r_B^{\mathrm{ion}}} - \frac{\chi_X}{r_X^{\mathrm{ion}}}$",
    "(B_x/B_r_ion)*(X_x/X_r_ion)": r"$\frac{\chi_B}{r_B^{\mathrm{ion}}} \cdot \frac{\chi_X}{r_X^{\mathrm{ion}}}$",
    "(B_x/B_r_ion)/(X_x/X_r_ion)": r"$\frac{\chi_B / r_B^{\mathrm{ion}}}{\chi_X / r_X^{\mathrm{ion}}}$",
    "(B_ea - X_ea)": r"$(EA_B - EA_X)$",
    "(B_ea / X_ea)": r"$EA_B / EA_X$",
    "(B_ea - X_ea)/(B_r_ion + X_r_ion)": r"$\frac{EA_B - EA_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    "(B_ea / X_ea)/(B_r_ion + X_r_ion)": r"$\frac{EA_B / EA_X}{r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}}$",
    "(B_ea/B_r_ion)": r"$EA_B / r_B^{\mathrm{ion}}$",
    "(X_ea/X_r_ion)": r"$EA_X / r_X^{\mathrm{ion}}$",
    "(B_ea/B_r_ion)+(X_ea/X_r_ion)": r"$\frac{EA_B}{r_B^{\mathrm{ion}}} + \frac{EA_X}{r_X^{\mathrm{ion}}}$",
    "(B_ea/B_r_ion)-(X_ea/X_r_ion)": r"$\frac{EA_B}{r_B^{\mathrm{ion}}} - \frac{EA_X}{r_X^{\mathrm{ion}}}$",
    "(B_ea/B_r_ion)*(X_ea/X_r_ion)": r"$\frac{EA_B}{r_B^{\mathrm{ion}}} \cdot \frac{EA_X}{r_X^{\mathrm{ion}}}$",
    "(B_ea/B_r_ion)/(X_ea/X_r_ion)": r"$\frac{EA_B / r_B^{\mathrm{ion}}}{EA_X / r_X^{\mathrm{ion}}}$",
    "(B_r_ion + X_r_ion)": r"$r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}}$",
    "(B_r_ion / X_r_ion)": r"$r_B^{\mathrm{ion}} / r_X^{\mathrm{ion}}$",
    "(A_r_ion + X_r_ion)/(1.414*(B_r_ion + X_r_ion))": r"$\frac{r_A^{\mathrm{ion}} + r_X^{\mathrm{ion}}}{1.414(r_B^{\mathrm{ion}} + r_X^{\mathrm{ion}})}$",
    "log(bond_var)": r"$\log(\sigma^2_{\mathrm{bond}})$",
    "log(bond_cv)": r"$\log(CV_{\mathrm{bond}})$",
    "log(delta_X_bx_bond_var)": r"$\log(\Delta \sigma^2_{X-B-X,\mathrm{bond}})$",
    "log(delta_ea_bx_bond_var)": r"$\log(\Delta \sigma^2_{EA-B-X,\mathrm{bond}})$",
    "log(ratio_X_bx_bond_var)": r"$\log(\sigma^2_{\rho,X-B-X,\mathrm{bond}})$",
    "log(ratio_ea_bx_bond_var)": r"$\log(\sigma^2_{\rho,EA-B-X,\mathrm{bond}})$",
    "log(delta_X_bx_bond_cv)": r"$\log(\Delta CV_{X-B-X,\mathrm{bond}})$",
    "log(ratio_X_bx_bond_cv)": r"$\log(CV_{\rho,X-B-X,\mathrm{bond}})$",
    "log(delta_ea_bx_bond_cv)": r"$\log(\Delta CV_{EA-B-X,\mathrm{bond}})$",
    "log(ratio_ea_bx_bond_cv)": r"$\log(CV_{\rho,EA-B-X,\mathrm{bond}})$"
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



def calculate_lead_advantage(rank_df, n_bootstrap=50000, confidence=0.95):
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
            N = len(method_ranks)  # 特征数
            lead_margin = (best_other_rank - target_rank) / (N - 1)
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

        results[target_feature] = {
            'mean_advantage': mean_advantage,
            'ci_lower': lower_bound,
            'ci_upper': upper_bound,
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

# forest_df = forest_df.sort_values(by="mean", ascending=False).head(14)
top_n = 10
bottom_n = 10

sorted_df = forest_df.sort_values(by="mean", ascending=False)

top_df = sorted_df.head(top_n)
bottom_df = sorted_df.tail(bottom_n)

gap_size = 1

gap_rows = pd.DataFrame(
    [["", np.nan, np.nan, np.nan]] * gap_size,
    columns=forest_df.columns
)

ellipsis_row = pd.DataFrame(
    [["...", np.nan, np.nan, np.nan]],
    columns=forest_df.columns
)

plot_df = pd.concat(
    [top_df, gap_rows, ellipsis_row, gap_rows, bottom_df],
    ignore_index=True
)

# 反转顺序（森林图视觉更好）
# forest_df = forest_df.iloc[::-1].reset_index(drop=True)
plot_df = plot_df.iloc[::-1].reset_index(drop=True)

# ========== 2. 绘制森林图 ==========
plt.figure(figsize=(10,10))

y_pos = np.arange(len(plot_df))

for i, row in plot_df.iterrows():
    # 空白行 → 什么都不画
    if row["feature"] == "":
        continue
    if row["feature"] == "...":
        x_center = -0.5

        offsets = [-0.35, 0, 0.35]

        plt.scatter(
            [x_center] * 3,
            [i + dy for dy in offsets],
            s=5,
            color="black",
            zorder=3
        )

        # 在右侧添加说明文字
        plt.text(
            x_center - 0.05,  # 向右偏一点
            i,  # 与中间点对齐
            "52 features omitted",
            va='center',
            fontsize=18
        )

        continue
    # 省略号 → 单独画
    # if row["feature"] == "...":
    #     plt.text(-0.5, i, "⋯", ha='center', va='center', fontsize=22)
    #     continue

    # 正常点
    plt.errorbar(
        row["mean"], i,
        xerr=[[row["mean"] - row["ci_low"]],
              [row["ci_high"] - row["mean"]]],
        fmt="o",
        color="darkblue",
        ecolor="gray",
        capsize=5,
        markersize=6
    )

# y轴标签
plt.yticks(y_pos, plot_df["feature"], fontsize=18)
plt.gca().invert_yaxis()

# 边距更紧凑
plt.xlabel("", fontsize=28)
plt.title("", fontsize=28)

# 更轻的网格线
plt.grid(axis="x", linestyle="--", alpha=0.25)
# 设置 x 轴范围与刻度
plt.xlim(0.01,-1.01 )
plt.xticks([-1,-0.5, 0],fontsize = 25)  # # [-1, -0.5, 0, 0.5, 1]
plt.subplots_adjust(left=0.21, right=0.97, top=0.99, bottom=0.05)
plt.savefig("forest_plot_perovs.png", dpi=300)
plt.savefig("forest_plot_perovs.pdf", dpi=300)
plt.show()
