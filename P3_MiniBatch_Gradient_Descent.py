# 导入必要的库
import matplotlib.pyplot as plt  # 用于绘图
import numpy as np

# 提供一些示例数据
x_data = np.linspace(0, 10, 5000)  # 创建一个包含5000个点的线性空间，范围从0到10
y_data = 5 * x_data + 3 + np.random.normal(0, 1, len(x_data))  # 生成带有噪声的线性数据

w = 0  # 初始化权重
b = 0  # 初始化偏差


# 定义线性回归模型的前向传播函数
def forward(x):
    return x * w + b


# 定义损失函数，衡量模型的预测与真实值之间的差距
def loss(x, y):
    y_pred = forward(x)  # 使用前向传播函数计算预测值
    return (y_pred - y) ** 2  # 返回平方损失


# 定义学习率和迭代次数
learning_rate = 0.005
num_epochs = 200
batch_size = 500  # 小批量样本的大小
losses = []  # 用于存储每个epoch的损失值

# 使用小批量随机梯度下降法更新权重和偏差
for epoch in range(num_epochs):
    # 将数据集打乱顺序
    indices = np.arange(len(x_data))
    np.random.shuffle(indices)

    for i in range(0, len(x_data), batch_size):
        # 选择小批量样本
        batch_indices = indices[i:i + batch_size]
        x_batch = x_data[batch_indices]
        y_batch = y_data[batch_indices]

        # 计算模型的预测值
        y_pred = forward(x_batch)
        # 计算平均损失值
        loss_val = np.mean(loss(x_batch, y_batch))

        # 计算梯度
        dw = np.mean(2 * x_batch * (y_pred - y_batch))  # 权重的梯度
        db = np.mean(2 * (y_pred - y_batch))  # 偏差的梯度

        # 更新权重和偏差
        w = w - learning_rate * dw  # 使用梯度下降法更新权重
        b = b - learning_rate * db  # 使用梯度下降法更新偏差
    losses.append(loss_val)  # 将最后一个小批量的损失值添加到损失列表中

# 绘制训练损失随epoch的变化图
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# 打印最终的权重和偏差
print("最终的权重 w:", w)
print("最终的偏差 b:", b)
