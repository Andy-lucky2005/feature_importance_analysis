import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# 读取数据
file_path = "standardization/scaled_data.xlsx"
data = pd.read_excel(file_path, header=0)

# 提取特征列
X = data.iloc[:, 2:16]

# 特征简化名称
feature_labels = {
    "Xp_M": "$\chi_p^M$",
    "Xp_M'": "$\chi_p^{M'}$",
    "IE_M (eV)": "$IE^M$\n(eV)",
    "IE_M' (eV)": "$IE^{M'}$\n(eV)",
    "r_M (Å)": "$r^M$\n(Å)",
    "r_M' (Å)": "$r^{M'}$\n(Å)",
    "Hf_MO (eV M)": "$\Delta H_f^{MO}$\n(eV)",
    "Hf_M'O (eV M')": "$\Delta H_f^{M'O}$\n(eV)",
    "Hf_M'(M) (eV)": "$\Delta H_f^{M'M}$\n(eV)",
    "Hsub_M (eV)": "$\Delta H_{sub}^M$\n(eV)",
    "Hsub_M' (eV)": "$\Delta H_{sub}^{M'}$\n(eV)",
    "γ_M (J/m^2)": "$\gamma^M$\n(J/m²)",
    "Nws_M (d.u.)": "$n_{ws}^M$\n(a.u.)",
    "Eg_M'O (eV)": "$E_g^{M'O}$\n(eV)"
}

# 创建保存图像的文件夹
output_dir = "Figures_Features_importance"
os.makedirs(output_dir, exist_ok=True)

for index, feature in enumerate(X.columns, start=1):
    feature_data = X[feature].dropna()
    n_samples = len(feature_data)

    # 固定尺寸，确保输出一致
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)


    # 判断该图是否显示 X/Y 轴刻度
    show_x = index in [11, 12, 13, 14]
    show_y = index in [1, 5, 9, 13]

    # 根据刻度存在与否，动态调整空白边
    left_margin = 0.12 if show_y else 0.02  # 无Y轴刻度 → 几乎贴边
    bottom_margin = 0.075 if show_x else 0.075  # 无X轴刻度 → 几乎贴边

    fig.subplots_adjust(left=left_margin, right=0.995, top=0.995, bottom=bottom_margin)

    # 手动压缩空白，防止裁剪文字
    # fig.subplots_adjust(left=0.099, right=0.99, top=0.99, bottom=0.075)

    # 绘制直方图（移除原`color`参数，后续手动设置渐变）
    hist = sns.histplot(
        data=feature_data,
        bins=15,
        stat='count',
        edgecolor="#0A4D7E",  # 保持柱体边缘颜色
        alpha=0.8,  # 保持透明度
        kde=False,
        ax=ax
    )

    # 根据 index 所属分组，选择渐变的起始色和结束色
    if index in {1, 5, 9, 13}:
        start_color, end_color = "#E85529", "#F2962C"
    elif index in {2, 6, 10, 14}:
        start_color, end_color = "#79EB46", "#37C5ED"
    elif index in {3, 7, 11}:
        start_color, end_color = "#48BAEE", "#6468F8"
    elif index in {4, 8, 12}:
        start_color, end_color = "#E14FEE", "#EE4FB4"
    else:
        start_color, end_color = "#DF84F1", "#FA61CA"

    # -------- 新增：为柱体设置渐变色 --------
    # 创建“#8081FF → #72D0F6”的自定义渐变色映射
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_gradient",  # 色图名称（自定义）
        [start_color, end_color]  # 渐变的起始色和结束色
    )

    patches = hist.patches  # 获取所有柱体的“patch”对象
    n_patches = len(patches)  # 柱体数量

    for i, patch in enumerate(patches):
        # 计算当前柱体在渐变中的位置（0~1之间）
        color_pos = i / (n_patches - 1) if n_patches > 1 else 0
        # 为柱体设置渐变颜色
        patch.set_facecolor(custom_cmap(color_pos))
        # 保持柱体边缘颜色（与原代码一致）
        patch.set_edgecolor("#0A4D7E")
        # 保持透明度（与原代码一致）
        patch.set_alpha(0.8)

    bin_width = hist.patches[0].get_width()

    # 绘制 KDE 曲线并转换为计数
    kde = sns.kdeplot(
        data=feature_data,
        color='#1F4864',
        linewidth=2,
        ax=ax
    )
    kde_line = kde.get_lines()[0]
    y_count = kde_line.get_ydata() * n_samples * bin_width
    kde_line.set_ydata(y_count)

    # 坐标范围和刻度设置
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(0, 80)

    ax.xaxis.set_major_locator(MultipleLocator(3))
    ax.yaxis.set_major_locator(MultipleLocator(25))

    # 控制刻度显示
    if index in [1, 5, 9, 13]:
        ax.tick_params(axis='y', labelsize=32, width=1, length=7)
    else:
        ax.tick_params(axis='y', labelleft=False, width=0, length=0)

    if index in [11, 12, 13, 14]:
        ax.tick_params(axis='x', labelsize=28, width=1, length=7)
    else:
        ax.tick_params(axis='x', labelbottom=False, width=0, length=0)

    # 去掉轴标签
    ax.set_xlabel('')
    ax.set_ylabel('')

    # 左上角最小值和最大值
    ax.text(
        0.03, 0.97,
        f'Min: {feature_data.min():.2f}\nMax: {feature_data.max():.2f}',
        ha='left', va='top',
        transform=ax.transAxes,
        fontsize=32,
        linespacing=1.1
    )

    # 右上角特征名称
    ax.text(
        0.97, 0.97,
        feature_labels.get(feature, feature),
        ha='right', va='top',
        transform=ax.transAxes,
        fontsize=42,
        linespacing=1.3
    )

    # 保存图片（禁止 tight 与 bbox_inches，确保一致性）
    plt.savefig(
        os.path.join(output_dir, f"Figure_{index}.png"),
        dpi=300,
        pad_inches=0.0,
        bbox_inches=None
    )
    plt.close()
