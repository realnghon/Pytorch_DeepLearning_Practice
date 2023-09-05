# 导入必要的库
import matplotlib.pyplot as plt  # 用于绘制图表
import numpy as np  # 用于生成示例数据
import torch  # 用于构建深度学习模型
from torch.optim import SGD  # 使用随机梯度下降优化器

# 提供一些示例数据
x_data = torch.tensor(np.linspace(0, 10, 200), dtype=torch.float32)  # 生成从0到10的等间隔的200个数据点作为输入特征
noise = torch.tensor(2 * np.random.rand(200) - 1, dtype=torch.float32)  # 生成随机噪声数据
y_data = 5 * x_data + 3 + noise.clone().detach()  # 生成目标数据，模拟线性关系，加入随机噪声

# 初始化模型参数和优化器
w = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)  # 初始化权重 w 为1.0，并启用梯度计算
b = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)  # 初始化偏差 b 为1.0，并启用梯度计算
optimizer = SGD([w, b], lr=0.01)  # 使用随机梯度下降优化器初始化，指定学习率为0.005
loss_fn = torch.nn.MSELoss()  # 使用均方误差损失函数初始化
losses = []  # 用于存储每个 epoch 的损失值

# 训练模型
for epoch in range(30):  # 进行30个训练周期
    optimizer.zero_grad()  # 清零梯度，防止梯度累积
    y_pred = w * x_data + b  # 计算模型的预测值
    loss = loss_fn(y_pred, y_data)  # 计算预测值与真实值之间的均方误差损失
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新模型参数
    losses.append(loss.item())  # 将当前损失值添加到损失列表中

# 绘制训练损失随 epoch 的变化图
plt.figure(figsize=(10, 6))
plt.plot(range(30), losses, label='Loss')  # 绘制损失值随 epoch 的变化曲线
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# 打印最终的权重和偏差
print("最终的权重 w:", w.item())
print("最终的偏差 b:", b.item())
