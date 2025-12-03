from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# ---------- 配置区 ----------
png_paths = [
    "MAE.png",
    "R².png"
]
out_pdf = "performance.pdf"
labels = ["A","B","C","D","E","F"]  # 改成字母 A-F
# ----------------------------

def pick_label_color(pil_img, sample_box=(0,0,50,50), white_thresh=180):
    """根据左上角亮度选择黑或白标签颜色。"""
    w, h = pil_img.size
    left, top, right, bottom = sample_box
    right = min(right, w)
    bottom = min(bottom, h)
    region = pil_img.convert('L').crop((left, top, right, bottom))
    arr = np.asarray(region)
    mean_brightness = arr.mean()
    return "black" if mean_brightness > white_thresh else "white"

# 读取 PNG 图片（只取前两张）
images = [Image.open(p).convert("RGB") for p in png_paths[:2]]

# --- 一行两列 ---
fig, axes = plt.subplots(1, 2, figsize=(14.5, 7))

# 子图紧凑排布
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                    wspace=0.05, hspace=0.05)

for i, ax in enumerate(axes):
    if i < len(images):
        img_rgb = images[i]
        ax.imshow(img_rgb, aspect='auto', interpolation='bilinear')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('white')
        color = pick_label_color(img_rgb, sample_box=(0,0,40,40), white_thresh=180)
        ax.text(-0.01, 1.01, labels[i],
                transform=ax.transAxes,
                fontsize=30, fontweight="bold",
                va="top", ha="left",
                color=color)
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.axis("off")

# 保存为 PDF
plt.savefig(out_pdf, dpi=300, edgecolor='white')
plt.close(fig)
print("保存完成：", out_pdf)
