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

    "Eadh formula(Feature mean)": r"Formula-MVPD",
    "Eadh formula(Average of all data)": r"Formula-AGM",
    "Eadh formula(Ranking average)": r"Formula-SGR",
    "Eadh formula-SHAP": r"Formula-SHAP",
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
    ax.invert_yaxis()

    # 添加白色网格
    ax.set_xticks(np.arange(len(labels)+1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(labels)+1)-0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="both", bottom=False, left=False)

    # 添加颜色条
    cbar_ax = fig.add_subplot(gs[1])
    cbar = ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                 ticks=[0,0.2,0.4,0.6,0.8,1.0],
                 orientation='vertical')
    # 调整 colorbar 字体大小
    cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    # 保存为 PDF 和 PNG
    pdf_path = os.path.join(f"{save_name}.pdf")
    png_path = os.path.join(f"{save_name}.png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # 关闭图像，避免阻塞
def calculate_loss_from_names(order_names, df_selected):
    """
    直接输入方法名称顺序计算 loss 并绘制热图

    Parameters
    ----------
    order_names : list
        方法名称列表，例如：
        [
            "XGBoost-TreeSHAP",
            "SVR-PFI",
            "RF-TreeSHAP",
            ...
        ]
    """

    # 获取原始列名
    all_cols = df_selected.columns.tolist()

    # 名称 -> 索引
    order = []

    for name in order_names:

        if name not in all_cols:
            raise ValueError(
                f"\n方法名称不存在:\n{name}\n\n"
                f"请检查Excel中的列名是否完全一致"
            )

        order.append(all_cols.index(name))

    # Spearman矩阵
    spearman_full = compute_spearman_matrix(df_selected).values

    # Target矩阵
    target = generate_target_matrix(len(order))

    # 重排
    M = spearman_full[np.ix_(order, order)]

    # 与GA保持一致
    M_flip = np.fliplr(M)

    loss = compute_loss(M_flip, target)

    print("=" * 60)
    print("手动名称排序 Loss 计算结果")
    print("=" * 60)

    print("\n排序顺序：")

    for i, name in enumerate(order_names):
        print(f"{i+1:2d}. {name}")

    print(f"\nLoss = {loss:.10f}")
    print("=" * 60)

    # 绘图
    draw_heatmap(
        M,
        order_names,
        save_name="Manual_Order"
    )

    return loss

def calculate_loss_from_order(order, df_selected):
    """
    根据用户指定排序计算 loss

    Parameters
    ----------
    order : list
        例如:
        [2,22,0,19,4,1,...]

    df_selected : DataFrame
    """

    # Spearman矩阵
    spearman_full = compute_spearman_matrix(df_selected).values

    n = len(order)

    # target矩阵
    target = generate_target_matrix(n)

    # 重排序
    M = spearman_full[np.ix_(order, order)]

    # 与GA保持一致
    M = np.fliplr(M)

    loss = compute_loss(M, target)

    print("=" * 60)
    print("手动排序 Loss 计算结果")
    print("=" * 60)
    print("排序：")
    print(order)
    print()
    print(f"Loss = {loss:.10f}")
    print("=" * 60)

    return loss

# 主流程
# =======================
if __name__ == "__main__":
    # =====================================================
    # 直接输入方法名称
    # =====================================================

    manual_order_names = [

        "LR-KernelSHAP",
        "LR-coef",
        "LR-permutation",
        "MI",
        "pearson correlation",
        "spearman correlation",

        "RF",

        "RF-TreeSHAP",
        "RF-KernelSHAP",
        "RF-permutation",
        "SVR-KernelSHAP",
        "SVR-PFI",
        "MLP-KernelSHAP",
        "MLP-permutation",
        "XGBoost-TreeSHAP",
        "XGBoost-KernelSHAP",
        "XGBoost-permutation",
        "GBRT",
        "GBRT-TreeSHAP",
        "GBRT-KernelSHAP",
        "GBRT-permutation",
        "Eadh formula(Ranking average)",
        "Eadh formula(Feature mean)",
        "Eadh formula(Average of all data)",
        "Formula-SHAP",
        "XGBoost",
    ]

    loss = calculate_loss_from_names(
        manual_order_names,
        df_selected
    )