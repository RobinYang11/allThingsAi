


import torch
import torch.nn as nn
import torch.optim as optim
import random

# 生成上海房价数据
datas = [
    [1, 100, 400, 3],
    [2, 100, 300, 1],
    [1, 80, 800, 1],
    [3, 100, 300, 2],
]

# 生成额外 100 条数据
for _ in range(100):
    row = [
        random.randint(1, 3),  # 第一列：房屋类型 (类别变量)
        random.randint(80, 100),  # 第二列：面积 (数值变量)
        random.randint(300, 800),  # 第三列：价格 (目标值)
        random.randint(1, 3)  # 第四列：位置 (类别变量)
    ]
    datas.append(row)

# 转换为张量
data = torch.tensor(datas, dtype=torch.float32)

# 归一化处理（仅对数值变量进行标准化）
max_size = data[:, 1].max()
max_price = data[:, 2].max()
data[:, 1] /= max_size  # 归一化面积
data[:, 2] /= max_price  # 归一化价格（目标变量）

# One-hot 编码类别变量（房屋类型 和 位置）
house_type = torch.nn.functional.one_hot(data[:, 0].to(torch.int64) - 1, num_classes=3)
location = torch.nn.functional.one_hot(data[:, 3].to(torch.int64) - 1, num_classes=3)

# 构造最终输入特征（拼接 one-hot 编码 和 数值特征）
X = torch.cat([house_type, data[:, 1:2], location], dim=1)  # 输入特征
y = data[:, 2:3]  # 目标值（房价）

# 定义模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(X.shape[1], 1)  # 输入维度是 one-hot 编码 + 数值特征
       
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# 训练设置
criterion = nn.MSELoss()  # 均方误差
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)  # 计算损失函数的
    loss.backward() # 反向传播：自动计算梯度
    optimizer.step() #执行一次梯度更新

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# 预测新房价
new_house = torch.tensor([[1, 85, 2]], dtype=torch.float32)  # 例子: 1-新房, 85平米, 2-静安
new_house[:, 1] /= max_size  # 归一化面积 加速收敛 防止梯度爆炸
house_type = torch.nn.functional.one_hot(new_house[:, 0].to(torch.int64) - 1, num_classes=3)
location = torch.nn.functional.one_hot(new_house[:, 2].to(torch.int64) - 1, num_classes=3)
new_house_features = torch.cat([house_type, new_house[:, 1:2], location], dim=1)

predicted_price = model(new_house_features) * max_price  # 反归一化
print(f"Predicted Price: {predicted_price.item():.2f} 万")



