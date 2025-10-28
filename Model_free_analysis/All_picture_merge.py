from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# ---------- 配置区 ----------
png_paths = [
    "Pearson_Spearman_Combined_manual.png",
    "MI.png"
]
out_pdf = "correlation_abs.pdf"
labels = ["", "C"]


# ----------------------------

def pick_label_color(pil_img, sample_box=(0, 0, 50, 50), white_thresh=180):
    """根据图片左上角亮度选择标签颜色"""
    w, h = pil_img.size
    left, top, right, bottom = sample_box
    right = min(right, w)
    bottom = min(bottom, h)
    region = pil_img.convert('L').crop((left, top, right, bottom))
    arr = np.asarray(region)
    mean_brightness = arr.mean()
    return "black" if mean_brightness > white_thresh else "white"


# 读取图片
images = [Image.open(p).convert("RGB") for p in png_paths]

# 使用 GridSpec 控制行高比例
fig = plt.figure(figsize=(20, 22))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)

# 上下两张图片
for i, ax_idx in enumerate(range(2)):
    ax = fig.add_subplot(gs[ax_idx, 0])
    img_rgb = images[i]
    ax.imshow(img_rgb, aspect='auto', interpolation='bilinear')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('white')

    color = pick_label_color(img_rgb, sample_box=(0, 0, 40, 40))
    ax.text(-0.01, 1.01, labels[i],
            transform=ax.transAxes,
            fontsize=34, fontweight="bold",
            va="top", ha="left",
            color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)

# 保存 PDF
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
plt.close(fig)
print("保存完成：", out_pdf)
