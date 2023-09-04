# 导入必要的库
import matplotlib.pyplot as plt  # 用于绘图
import numpy as np  # 用于数学运算

# 提供示例函数 y = 3x+1
x_data = [1, 2, 3, 4]  # 输入数据
y_data = [4, 7, 10, 13]  # 对应的目标值


# 定义线性回归模型的前向传播函数
def forward(x):
    return x * w + b


# 定义损失函数，衡量模型的预测与真实值之间的差距
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# 穷举法创建一组可能的权重和偏差的值用于可视化
W = np.arange(1, 4, 0.1)  # 权重的范围
B = np.arange(-1, 2, 0.1)  # 偏差的范围
[w, b] = np.meshgrid(W, B)  # 创建权重和偏差的网格

"""
np.meshgrid 是 NumPy 库中的一个函数，用于创建多维网格坐标矩阵。它的主要作用是生成两个或多个一维数组的网格坐标矩阵，这些坐标矩阵可以用于构建多维数据的网格。
具体来说，np.meshgrid 接受两个或多个一维数组作为输入，然后返回一个网格坐标矩阵，其中的每个元素都是输入数组中对应位置的坐标值。
"""

# 初始化损失总和矩阵
l_sum = np.zeros_like(w)

# 遍历输入数据点，计算每个点的损失并将它们累加
for x_val, y_val in zip(x_data, y_data):
    loss_grid = loss(x_val, y_val)
    l_sum += loss_grid

# 创建一个三维图形对象
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(projection='3d')

# 找到最小损失的位置和值
# np.argmin() 函数用于找到给定数组中的最小值的索引
# np.unravel_index 是一个 NumPy 函数，用于将一维索引转换为多维索引，其参数包括一个一维索引 index 和一个多维数组的形状 shape，在这里，它的作用是将第一步找到的最小损失值的一维索引转换为对应的二维索引，以便在 l_sum 矩阵中定位最小损失值的位置。
min_loss_index = np.unravel_index(np.argmin(l_sum), l_sum.shape)
min_w = W[min_loss_index[0]]
min_b = B[min_loss_index[1]]
min_loss = l_sum[min_loss_index]

# 绘制损失表面
surf = ax.plot_surface(w, b, l_sum / 4, cmap='coolwarm', alpha=0.618)  # 使用'coolwarm'色图表示损失

# 设置图形的属性
ax.grid(True)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')

# 在图上标注最小损失的位置和数值
ax.text(min_w, min_b, min_loss / 4, f'Min Loss = {min_loss:.2f}\nw = {min_w:.2f}, b = {min_b:.2f}',
        fontsize=10, ha='center')

# 添加以下代码来标注最小损失值的点
ax.scatter([min_w], [min_b], [min_loss / 4], color='brown', s=50, label='Min Loss')
ax.legend()

# 设置图的标题并显示图形
plt.title('Loss Surface Visualization')
plt.show()
