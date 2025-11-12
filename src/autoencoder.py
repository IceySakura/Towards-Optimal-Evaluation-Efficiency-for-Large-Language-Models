import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 自编码器定义
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

if __name__ == '__main__':

    # 训练数据：? 个 200 维向量
    with open('lb.pickle', 'rb') as file:
        data = pickle.load(file)
    list_data = []
    for s in data.keys():
        cor = data[s]['correctness']
        for x in cor:
            list_data.append(list(x)[0:200])
    # print(list_data)
    data = np.array(list_data, dtype=np.float32)
    data_tensor = torch.tensor(data)

    # 模型、损失函数和优化器
    input_dim = 200
    latent_dim = 20
    model = Autoencoder(input_dim, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # 数据加载器
    batch_size = 64
    dataset = TensorDataset(data_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练
    epochs = 30
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            batch_data = batch[0]
            encoded, decoded = model(batch_data)
            loss = criterion(decoded, batch_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), "model.pth")

    # 提取低维表示
    model.eval()
    with torch.no_grad():
        latent_features = model.encoder(data_tensor)
        print(f"低维表示形状: {latent_features.shape}")  # (?, 20)
