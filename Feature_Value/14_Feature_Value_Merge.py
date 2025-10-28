from PIL import Image, ImageDraw, ImageFont, ImageChops
import os

from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'Arial'       # 全局字体设置为 Arial
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# ======================
# 1自动裁剪函数（去掉图片四周白边，避免空隙）
# ======================
def trim(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    return img.crop(bbox) if bbox else img

# ======================
# 2️读取与裁剪图片
# ======================
image_paths = [f"Figures_Features_importance/Figure_{i}.png" for i in range(1, 15)]
images = [trim(Image.open(img_path)) for img_path in image_paths]

# 确保尺寸一致
img_width, img_height = images[0].size

# ======================
# 3网格布局
# ======================
columns = 4
rows = 4

# ======================
# 4️边距设置（为坐标轴标签预留空间）
# ======================
# left_margin = 400    # 左边为Y轴标签
# right_margin = 80
# top_margin = 80
# bottom_margin = 300  # 底部为X轴标签

left_margin = 250     # 从400 缩到 250
bottom_margin = 200   # 从300 缩到 200
top_margin = 50       # 可从80 缩到 50
right_margin = 50     # 从80 缩到 50


# 创建整体大图
large_width = left_margin + columns * img_width + right_margin
large_height = top_margin + rows * img_height + bottom_margin
large_image = Image.new('RGB', (large_width, large_height), (255, 255, 255))

# ======================
# 5️拼接图片（无缝隙）
# ======================
for i, img in enumerate(images):
    row = i // columns
    col = i % columns
    x_offset = left_margin + col * img_width
    y_offset = top_margin + row * img_height
    large_image.paste(img, (x_offset, y_offset))

# ======================
# 6️添加轴标签
# ======================
draw = ImageDraw.Draw(large_image)

# 选择字体（兼容无 Arial 环境）
try:
    font_x = ImageFont.truetype("arial.ttf", 180)
    font_y = ImageFont.truetype("arial.ttf", 180)
except:
    font_x = ImageFont.load_default()
    font_y = ImageFont.load_default()

# 添加 X 轴标签
x_label_text = "Z-Score Normalized Values"
bbox = draw.textbbox((0, 0), x_label_text, font=font_x)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
x_pos = left_margin + (columns * img_width - text_width) // 2
y_pos = top_margin + rows * img_height + (bottom_margin - text_height) // 2
draw.text((x_pos, y_pos), x_label_text, font=font_x, fill=(0, 0, 0))

# 添加 Y 轴标签（旋转）
y_label_text = "Count"
y_temp = Image.new('RGBA', (600, 2000), (255, 255, 255, 0))
y_draw = ImageDraw.Draw(y_temp)
y_draw.text((300, 1000), y_label_text, font=font_y, fill=(0, 0, 0), anchor="mm")
y_temp_rotated = y_temp.rotate(90, expand=True)
x_pos_y = (left_margin - y_temp_rotated.width) // 2
y_pos_y = top_margin + (rows * img_height - y_temp_rotated.height) // 2
large_image.paste(y_temp_rotated, (x_pos_y, y_pos_y), y_temp_rotated)

# ======================
# 7️保存为 PDF
# ======================
output_path = "value_combined_images_labeled_final.pdf"
large_image.save(output_path)
print(f"成功保存整合图片: {output_path}")
print(f"大图尺寸: {large_width} x {large_height}")
