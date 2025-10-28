import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, BoundaryNorm
plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# ================== 读取已有图片 ===================
img_pearson = mpimg.imread("pearson.png")
img_spearman = mpimg.imread("spearman.png")

# ================== colormap 设置 ===================
bounds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
colors = ["#3654A5", "#0070B3", "#1F9FD4", "#35B79C", "#C0BEC0",
          "#E4E0E4", "#F7D635", "#F4A153", "#F48E98", "#EB3A4B"]
cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, cmap.N)

# ================== 手动布局绘图 ===================
fig = plt.figure(figsize=(18, 9))

# 左图 Pearson
ax1 = fig.add_axes([0.0, 0.0, 0.48, 1.0])  # [left, bottom, width, height]
ax1.imshow(img_pearson,aspect='auto')
ax1.axis('off')
ax1.text(0.02, 1.04, 'A', transform=ax1.transAxes,
         fontsize=34, fontweight='bold', color='black', va='top')

# 右图 Spearman
ax2 = fig.add_axes([0.48, 0.0, 0.48, 1.0])
ax2.imshow(img_spearman,aspect='auto')
ax2.axis('off')
ax2.text(0.02, 1.04, 'B', transform=ax2.transAxes,
         fontsize=34, fontweight='bold', color='black', va='top')

# 色条，高度 80%，垂直居中
cbar_ax = fig.add_axes([0.975, 0.1, 0.015, 0.9])  # [left, bottom, width, height]
cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                  ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                  spacing='proportional',
                  orientation='vertical')
cb.ax.tick_params(labelsize=22)  # 调整色条刻度字体大小

# ================== 保存输出 ===================
plt.savefig("Pearson_Spearman_Combined_manual.png", dpi=300, bbox_inches="tight", pad_inches=0)

