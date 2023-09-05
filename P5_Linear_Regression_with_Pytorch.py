import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
import numpy as np  # 导入NumPy库，用于数学计算
import torch  # 导入PyTorch库，用于构建和训练神经网络

# 生成数据
x_data = torch.tensor(np.linspace(0, 10, 500), dtype=torch.float32).reshape(-1,
                                                                            1)  # 生成一组从0到10的500个等间距的数据点，并将其转换为PyTorch张量
noise = torch.tensor(2 * np.random.rand(500) - 1, dtype=torch.float32).reshape(-1,
                                                                               1)  # 生成一组随机噪声数据，噪声范围在-1到1之间，并将其转换为PyTorch张量
y_data = 5 * x_data + 3 + noise.detach()  # 根据线性关系生成目标数据，并使用detach()方法将其从计算图中分离，使其不参与梯度计算


# 定义线性模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 创建一个线性层，输入维度为1，输出维度为1

    def forward(self, x):
        y_pred = self.linear(x)  # 前向传播：将输入数据传递给线性层进行计算
        return y_pred


# 初始化模型、损失函数和优化器
model = LinearModel()  # 创建线性模型的实例
criterion = torch.nn.MSELoss()  # 创建均方误差损失函数实例，用于度量模型预测和实际值之间的差异
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 创建随机梯度下降优化器实例，用于更新模型参数，设置学习率为0.01
losses = []  # 用于存储每个epoch的损失值

# 训练模型
for epoch in range(100):  # 迭代训练100个epoch
    y_pred = model(x_data)  # 使用模型进行预测
    loss = criterion(y_pred, y_data)  # 计算预测值和真实值之间的均方误差损失
    losses.append(loss.item())  # 将损失值添加到损失列表中
    optimizer.zero_grad()  # 清零梯度，防止梯度累积
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新模型参数

# 打印最终参数
print('w=', model.linear.weight.item())  # 打印模型学到的权重参数
print('b=', model.linear.bias.item())  # 打印模型学到的偏置参数

# 绘制损失随epoch的变化曲线
plt.figure(figsize=(10, 6))
plt.plot(range(100), losses, label='Loss')  # 绘制损失值随epoch的变化曲线
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()
