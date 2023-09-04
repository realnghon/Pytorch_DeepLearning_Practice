import matplotlib.pyplot as plt
import numpy as np

# 提供示例数据
x_data = np.linspace(0, 10, 5000)
y_data = 5 * x_data + 3 + np.random.normal(0, 1, len(x_data))

# 初始化参数
w = np.random.randn()
b = np.random.randn()


# 定义线性回归模型的前向传播函数
def forward(x):
    return x * w + b


# 定义损失函数
def loss(x, y):
    y_pred = forward(x)
    return np.mean((y_pred - y) ** 2)


# 超参数设置
learning_rate = 0.001
num_epochs = 100
losses = []

# 训练模型
for epoch in range(num_epochs):
    dw = np.mean(2 * x_data * (forward(x_data) - y_data))
    db = np.mean(2 * (forward(x_data) - y_data))
    w -= learning_rate * dw
    b -= learning_rate * db
    loss_val = loss(x_data, y_data)
    losses.append(loss_val)

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
