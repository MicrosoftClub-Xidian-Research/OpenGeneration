import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20, num_labels=10):
        super(CVAE, self).__init__()
        self.image_size = image_size
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.num_labels = num_labels

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(image_size + num_labels, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(h_dim // 2, z_dim)
        self.fc_logvar = nn.Linear(h_dim // 2, z_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + num_labels, h_dim // 2),
            nn.ReLU(),
            nn.Linear(h_dim // 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid()
        )

    def encode(self, x, c):
        c = F.one_hot(c, num_classes=self.num_labels).float()
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        c = F.one_hot(c, num_classes=self.num_labels).float()
        inputs = torch.cat([z, c], dim=1)
        return self.decoder(inputs)

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.image_size), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar