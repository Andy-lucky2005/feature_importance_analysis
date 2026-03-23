from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ---------- 配置区 ----------
png_paths = [
    "RF_PFI_custom_order.png",
    "GBRT_PFI_custom_order.png",
    "XGBoost_PFI_custom_order.png",
    "LR_PFI_custom_order.png",
    "MLP_PFI_custom_order.png",
    "SVR_PFI_custom_order.png"
]


# out_pdf = "merged_SHAP_letters.pdf"
out_pdf = "PFI.pdf"
labels = ["A","B","C","D","E","F"]  # 改成字母 A-F
# ----------------------------

def pick_label_color(pil_img, sample_box=(0,0,50,50), white_thresh=180):
    """
    根据左上角亮度选择黑或白标签颜色。
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

single_fig_width = 15
single_fig_height = 23
fig_width = 3 * single_fig_width
fig_height = 2 * single_fig_height

# 绘图：2x3 布局
fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height))
axes = axes.flat

# 子图紧凑排布
# 子图紧凑排布
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                    wspace=0.02, hspace=0.05)

for i, ax in enumerate(axes):
    if i < len(images):
        img_rgb = images[i]
        ax.imshow(img_rgb, aspect='auto', interpolation='bilinear')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('white')
        color = pick_label_color(img_rgb, sample_box=(0,0,40,40), white_thresh=180)
        # 左上角添加字母
        ax.text(0.01, 0.99, labels[i],
                transform=ax.transAxes,
                fontsize=65, fontweight="bold",
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