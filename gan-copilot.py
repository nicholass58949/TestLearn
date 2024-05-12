import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 设置超参数
LR_G = 0.0001
LR_D = 0.0001
BATCH_SIZE = 64
N_IDEAS = 5
ART_COMPONENTS = 15

# 创建一个范围内的数据，使得Generator可以生成类似范围的数据
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])


# 理想数据生成器
def artist_work():
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paints = a * np.power(PAINT_POINTS, 2) + (a - 1)
    paints = torch.from_numpy(paints).float()
    return paints


# 定义Generator和Discriminator网络
G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS)
)

D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

# 定义优化器
optimizer_G = torch.optim.Adam(G.parameters(), lr=LR_G)
optimizer_D = torch.optim.Adam(D.parameters(), lr=LR_D)

# 训练GAN网络
for step in range(10000):
    artist_painting = artist_work()
    G_idea = torch.randn(BATCH_SIZE, N_IDEAS)
    G_paintings = G(G_idea)
    pro_artist0 = D(artist_painting)
    pro_artist1 = D(G_paintings)

    G_loss = -1 / torch.mean(torch.log(1. - pro_artist1))
    D_loss = -torch.mean(torch.log(pro_artist0) + torch.log(1 - pro_artist1))

    optimizer_D.zero_grad()
    D_loss.backward(retain_graph=True)
    optimizer_D.step()

    optimizer_G.zero_grad()
    G_loss.backward()
    optimizer_G.step()

# 生成结果
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.legend(loc='upper right')
plt.show()
