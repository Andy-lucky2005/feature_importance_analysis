import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
np.random.seed(1412)
# ========== 1. 读取数据 ==========
df = pd.read_excel("all_sort.xlsx", index_col=0)

feature_labels = {
    "E": r"$E$",
    "K": r"$K$",
    "G": r"$G$",
    "ν": r"$\nu$",
    "Rm": r"$R_m$",
    "Ri": r"$R_i$",
    "Rc": r"$R_c$",
    "Vm": r"$V_m$",
    "Hs": r"$H_s$",
    "Hc": r"$H_c$",
    "VEC": r"$\mathrm{VEC}$",
    "e/a": r"$e/a$",
    "Ew": r"$E_w$",
    "Xp": r"$\chi_p$",
    "Xa": r"$\chi_a$",
    "Xm": r"$\chi_m$",
    "Xr": r"$\chi_r$",
    "Ec": r"$E_c$",
    "δE": r"$\delta E$",
    "δK": r"$\delta K$",
    "δG": r"$\delta G$",
    "δν": r"$\delta \nu$",
    "δRm": r"$\delta R_m$",
    "δRi": r"$\delta R_i$",
    "δRc": r"$\delta R_c$",
    "δVm": r"$\delta V_m$",
    "δHs": r"$\delta H_s$",
    "δHc": r"$\delta H_c$",
    "δVEC": r"$\delta \mathrm{VEC}$",
    "δe/a": r"$\delta (e/a)$",
    "δEw": r"$\delta E_w$",
    "δXp": r"$\delta \chi_p$",
    "δXa": r"$\delta \chi_a$",
    "δXm": r"$\delta \chi_m$",
    "δXr": r"$\delta \chi_r$",
    "δEc": r"$\delta E_c$",
    "ΔE": r"$\Delta E$",
    "ΔK": r"$\Delta K$",
    "ΔG": r"$\Delta G$",
    "Δν": r"$\Delta \nu$",
    "ΔRm": r"$\Delta R_m$",
    "ΔRi": r"$\Delta R_i$",
    "ΔRc": r"$\Delta R_c$",
    "ΔVm": r"$\Delta V_m$",
    "ΔHs": r"$\Delta H_s$",
    "ΔHc": r"$\Delta H_c$",
    "ΔVEC": r"$\Delta \mathrm{VEC}$",
    "Δe/a": r"$\Delta (e/a)$",
    "Δew": r"$\Delta E_w$",
    "ΔXp": r"$\Delta \chi_p$",
    "ΔXa": r"$\Delta \chi_a$",
    "ΔXm": r"$\Delta \chi_m$",
    "ΔXr": r"$\Delta \chi_r$",
    "ΔEc": r"$\Delta E_c$"
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

        # 只统计在 1.max_rank 范围内的排名
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


# 500000
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
            N = len(method_ranks)
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

# 选出 mean 最大的10个
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

        plt.text(
            x_center - 0.05,  # 向右偏一点
            i,  # 与中间点对齐
            "34 features omitted",
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
# 固定边距
plt.subplots_adjust(left=0.19, right=0.97, top=0.99, bottom=0.05)
plt.savefig("forest_plot_HEAs.png", dpi=300)
plt.savefig("forest_plot_HEAs.pdf", dpi=300)
plt.show()
