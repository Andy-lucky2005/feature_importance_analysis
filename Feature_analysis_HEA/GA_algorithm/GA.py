import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from sklearn.utils import shuffle
import random
import os
# os.makedirs("results", exist_ok=True)

# =======================
# 设置随机种子
# =======================
random.seed(42)
np.random.seed(42)
# =======================
# 1读取Excel数据 & 预处理
# =======================
file_path = "all_sort.xlsx"
df = pd.read_excel(file_path, header=0)

# 去掉第一列 + 特定列
exclude_cols = []

all_columns = df.columns.tolist()
selected_features = [col for col in all_columns[1:] if col not in exclude_cols]

# 转数值型并清除NaN
df_selected = df[selected_features].apply(pd.to_numeric, errors='coerce')
dropped_cols = df_selected.columns[df_selected.isna().all()].tolist()
if dropped_cols:
    print(f"[提示] 以下列全为NaN已自动删除: {dropped_cols}")
df_selected = df_selected.dropna(axis=1, how='all').dropna(axis=0)

# 打印放在dropna之后，数量才准确
print(f"共保留 {len(df_selected.columns)} 个方法/特征用于优化。\n")

# 5. 特征简化名称
feature_labels = {
    "RF-TreeSHAP": r"RF-TreeSHAP",
    "GBRT-TreeSHAP": r"GBRT-TreeSHAP",
    "XGBoost-TreeSHAP": r"XGBoost-TreeSHAP",

    "RF-KernelSHAP": r"RF-KernelSHAP",
    "GBRT-KernelSHAP": r"GBRT-KernelSHAP",
    "XGBoost-KernelSHAP": r"XGBoost-KernelSHAP",

    "RF-permutation": r"RF-PFI",
    "GBRT-permutation": r"GBRT-PFI",
    "XGBoost-permutation": r"XGBoost-PFI",

    "LR-KernelSHAP": r"LR-KernelSHAP",
    "MLP-KernelSHAP": r"MLP-KernelSHAP",

    "LR-permutation": r"LR-PFI",
    "MLP-permutation": r"MLP-PFI",

    "LR-coef": r"LR-Coefficient",

    "RF": r"RF-MDI",
    "GBRT": r"GBRT-MDI",
    "XGBoost": r"XGBoost-MDI",

    "MI": r"MI",

    "pearson correlation": r"Pearson",
    "spearman correlation": r"Spearman",

    "SVR-KernelSHAP": r"SVR-KernelSHAP",
    "SVR-PFI": r"SVR-PFI",
    #
    # "Eadh formula(Feature mean)": r"Formula-MVPD",
    # "Eadh formula(Average of all data)": r"Formula-AGM",
    # "Eadh formula(Ranking average)": r"Formula-SGR",
    # "Eadh formula-SHAP": r"Formula-SHAP",
}

# =======================
# 工具函数定义
# =======================
def compute_spearman_matrix(df):
    """计算Spearman相关矩阵（取绝对值）"""
    return df.corr(method='spearman').abs()

def generate_target_matrix(n, decay_rate=0.04):
    target = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            # 计算点 (i, j) 到反对角线 i + j = n - 1 的距离
            distance_to_anti_diagonal = abs((n - 1) - (i + j))
            # 目标值从 1.0 开始，根据距离衰减。
            target[i, j] = max(0.0, 1.0 - decay_rate * distance_to_anti_diagonal)
    return target

def compute_loss(matrix, target):
    """
    计算 loss：当 matrix 与 target 偏差超过预期方向时，对偏差进行平方惩罚
    """
    loss = 0.0
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if target[i, j] > 0.50:
                # 惩罚 matrix < target（偏小）
                diff = target[i, j] - matrix[i, j]
                if diff > 0:
                    loss += diff**2
            else:
                # 惩罚 matrix > target（偏大）
                diff = matrix[i, j] - target[i, j]
                if diff > 0:
                    loss += diff**2
    return loss

def draw_heatmap(matrix, labels, save_name="Spearman_MVPD"):
    """绘制相关性热图"""
    display_labels = [feature_labels.get(l, l) for l in labels]
    bounds = np.linspace(0, 1, 11)
    colors = ["#3654A5", "#0070B3", "#1F9FD4", "#35B79C", "#C0BEC0", "#E4E0E4",
              "#F7D635", "#F4A153", "#F48E98", "#EB3A4B"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(12,10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[30, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])

    im = ax.imshow(matrix, cmap=cmap, norm=norm)

    # =============================
    # 在格子中标注 Spearman 数值
    # =============================
    # n = matrix.shape[0]
    # for i in range(n):
    #     for j in range(n):
    #         value = matrix[i, j]
    #         ax.text(j, i,
    #                 f"{value:.2f}",
    #                 ha="center",
    #                 va="center",
    #                 fontsize=6,
    #                 color="black")

    ax.set_xticks(np.arange(len(display_labels)))
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=15)
    ax.set_yticklabels(display_labels, fontsize=15)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    # 去掉y轴刻度线
    ax.tick_params(axis='y', which='both', length=0)
    ax.invert_yaxis()

    # 添加白色网格
    ax.set_xticks(np.arange(len(labels)+1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(labels)+1)-0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="both", bottom=False, left=False)

    # 添加颜色条
    cbar_ax = fig.add_subplot(gs[1])
    cbar_ax.axis('off')
    # ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
    #              ticks=[0,0.2,0.4,0.6,0.8,1.0],
    #              orientation='vertical')

    # ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    # 保存为 PDF 和 PNG
    pdf_path = os.path.join(f"{save_name}.pdf")
    png_path = os.path.join(f"{save_name}.png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # 关闭图像，避免阻塞

# 新增 1： 2-opt 局部搜索精化
def two_opt_improve(order, M, target):
    n = len(order)
    best = order[:]
    best_loss = compute_loss(M[np.ix_(best, best)], target)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new = best[:]
                new[i], new[j] = new[j], new[i]
                # 修改2
                M_sub = M[np.ix_(new, new)]
                M_sub = np.fliplr(M_sub)
                l = compute_loss(M_sub, target)
                # l = compute_loss(M[np.ix_(new, new)], target)
                if l < best_loss:
                    best = new
                    best_loss = l
                    improved = True
    return best, best_loss

# =======================
# 基因算法优化排序
# =======================
def genetic_algorithm(df, pop_size=600, n_generations=5000,
                           crossover_prob=0.8, mutation_prob=0.2,
                           stagnation_limit=80):
    features = df.columns.tolist()
    n = len(features)

    # 预计算 Spearman
    # spearman_full = compute_spearman_matrix(df)
    spearman_full = compute_spearman_matrix(df).values

    # 目标矩阵
    target = generate_target_matrix(n)

    # 初始化种群
    population = [random.sample(range(n), n) for _ in range(pop_size)]

    best_loss = float('inf')
    best_order = None
    stagnation = 0

    for gen in range(n_generations):
        losses = []
        for individual in population:
            # 修改1：
            # M = spearman_full[np.ix_(individual, individual)]
            # loss = compute_loss(M, target)
            M = spearman_full[np.ix_(individual, individual)]
            M = np.fliplr(M)
            loss = compute_loss(M, target)
            losses.append(loss)

        # 排序
        idx = np.argsort(losses)
        elites = [population[i][:] for i in idx[:pop_size // 2]]

        # 更新最优
        current_best_loss = losses[idx[0]]
        if current_best_loss < best_loss:
            best_loss = current_best_loss
            best_order = elites[0][:]
            # print("best order",best_order,"best loss:",best_loss)
            stagnation = 0
        else:
            stagnation += 1

        print(f"第 {gen:2d} 代 | 最优 loss = {best_loss:.6f}")

        # 早停
        if stagnation >= stagnation_limit:
            print(f"连续 {stagnation_limit} 代未优化 - 提前停止")
            break

        # 生成子代
        offspring = []
        while len(offspring) < pop_size - len(elites):
            p1, p2 = random.sample(elites, 2)
            if random.random() < crossover_prob:
                child = ox_crossover(p1, p2)
            else:
                child = random.choice([p1, p2])[:]

            # 突变
            if random.random() < mutation_prob:
                i, j = random.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]

            offspring.append(child)

        # 更新种群
        population = elites + offspring

    print(f"\nGA完成，最优 loss = {best_loss:.6f}，开始2-opt精化...")
    refined_order, refined_loss = two_opt_improve(best_order, spearman_full, target)
    print(f"2-opt精化完成: {best_loss:.6f} -> {refined_loss:.6f}")
    # 用精化后的结果构造返回值
    best_corr = spearman_full[np.ix_(refined_order, refined_order)]
    return refined_order, best_corr, refined_loss, target
    # # 返回最终排序（列名字）
    # best_corr = spearman_full[np.ix_(best_order, best_order)]
    #
    # return best_order,best_corr, best_loss, target


# 交叉
def ox_crossover(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n

    # 保留片段
    child[a:b+1] = p1[a:b+1]

    # 用 p2 填空位
    fill = [gene for gene in p2 if gene not in child]
    fill_idx = 0

    for i in range(n):
        if child[i] is None:
            child[i] = fill[fill_idx]
            fill_idx += 1

    return child


def verify_result(best_order, best_loss, df_selected):
    # """
    # 验证 genetic_algorithm 返回的 best_order 与 best_loss 是否一致。
    #
    # 参数：
    #     best_order : GA 返回的最优排序索引列表
    #     best_loss  : GA 返回的最优 loss 值
    #     df_selected: 原始数据 DataFrame（用于重新计算 Spearman 矩阵）
    #
    # 返回：
    #     passed : bool，True 表示验证通过
    # """
    print("=" * 55)
    print("           结果一致性验证")
    print("=" * 55)

    # ── 步骤1：独立重新计算 Spearman 矩阵 ────────────────────
    # 完全不依赖 GA 内部的矩阵，重新算一遍
    spearman_matrix = compute_spearman_matrix(df_selected).values

    # ── 步骤2：独立重新生成 target 矩阵 ──────────────────────
    n = len(best_order)
    target = generate_target_matrix(n)

    # ── 步骤3：按 best_order 重排 Spearman 矩阵 ──────────────
    reordered_matrix = spearman_matrix[np.ix_(best_order, best_order)]

    # ── 步骤4：独立重新计算 loss ──────────────────────────────
    reordered_matrix = np.fliplr(reordered_matrix)
    recalc_loss = compute_loss(reordered_matrix, target)
    # recalc_loss = compute_loss(reordered_matrix, target)

    # ── 步骤5：对比两个 loss 值 ───────────────────────────────
    diff = abs(recalc_loss - best_loss)

    print(f"\nGA 返回的 loss       : {best_loss:.10f}")
    print(f"独立重算的 loss      : {recalc_loss:.10f}")
    print(f"两者差值             : {diff:.2e}")
    print("=" * 55)

# =======================
# 主流程
# =======================
if __name__ == "__main__":
    print("=== 开始基因算法优化特征排序 ===\n")
    best_order, best_corr, best_loss, target_matrix = genetic_algorithm(df_selected)

    print("\n=== 优化完成 ===")
    print(f"GA 输出的最优 loss 值: {best_loss:.6f}")
    print("最优特征排序顺序如下：")
    print("best_order:",best_order)
    # === 从 Excel 中按 best_order 提取对应的表头 ===
    print("\n=== 按最佳排序提取列头 ===")

    # === 将索引映射回列名 ===
    colnames = df_selected.columns.tolist()
    best_order_names = [colnames[i] for i in best_order]

    print("\n=== 最优排序对应的列名 ===")
    for i, name in enumerate(best_order_names):
        print(f"{i+1:2d}. {name}")

    # ---------------------------------------------------
    # 使用 GA 内部的最佳排序计算 target 与 spearman
    # ---------------------------------------------------

    best_index = best_order  # 直接用 GA 输出的索引

    spearman_full = compute_spearman_matrix(df_selected).values
    best_corr_verify = spearman_full[np.ix_(best_index, best_index)]

    # 调用验证函数
    verify_result(best_order, best_loss, df_selected)
    draw_heatmap(np.fliplr(target_matrix), best_order_names, save_name="GA_Target_HEAs")
    draw_heatmap(best_corr_verify, best_order_names, save_name="GA_Spearman_HEAs")