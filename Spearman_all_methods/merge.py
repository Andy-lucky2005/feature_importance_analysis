from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ---------- 配置区 ----------
png_paths = [
    "Eadh-all_data/Spearman_Correlation_with_Formula_all.png",
    "Eadh_MVPD/Spearman_Correlation_with_Formula_mean.png",
    "Eadh_SGR/Spearman_Correlation_with_Formula_Ranking.png",
    "Eadh-SHAP/Spearman_Correlation_with_Formula_SHAP.png",
]

out_pdf = "Formula_spearman.pdf"
labels = ["A","B","C","D"]  # 2x2 图片标注 A-D
# ----------------------------

def pick_label_color(pil_img, sample_box=(0,0,50,50), white_thresh=180):
    """
    根据左上角亮度选择黑或白标签颜色
    """
    w, h = pil_img.size
    left, top, right, bottom = sample_box
    right = min(right, w)
    bottom = min(bottom, h)
    region = pil_img.convert('L').crop((left, top, right, bottom))
    arr = np.asarray(region)
    mean_brightness = arr.mean()
    return "black" if mean_brightness > white_thresh else "white"

# 读取 PNG 图片
images = []
for p in png_paths:
    img = Image.open(p).convert("RGB")
    images.append(img)

# 绘图：2x2 布局，纵向尺寸缩小
fig, axes = plt.subplots(2, 2, figsize=(11, 9))  # 宽12，高9
axes = axes.flat

# 子图紧凑排布
fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02,
                    wspace=0.01, hspace=0.01)  # 减小间距

for i, ax in enumerate(axes):
    if i < len(images):
        img_rgb = images[i]
        ax.imshow(img_rgb, aspect='equal', interpolation='bilinear')  # 保持正方形
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('white')
        color = pick_label_color(img_rgb, sample_box=(0,0,40,40), white_thresh=180)
        # 左上角添加缩小字母
        ax.text(0.01, 0.99, labels[i],
                transform=ax.transAxes,
                fontsize=18, fontweight="bold",  # 缩小字体
                va="top", ha="left",
                color=color)
        # 隐藏子图边框
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.axis("off")

# 保存为 PDF
plt.savefig(out_pdf, dpi=300, edgecolor='white')
plt.close(fig)
print("保存完成：", out_pdf)
