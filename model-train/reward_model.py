import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelForRewardGeneration(nn.Module):
    def __init__(self, encoder, hidden_size):
        super(ModelForRewardGeneration, self).__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(768, hidden_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.encoder(x)[:, 0, :]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

def reference_reward_loss(reward, pred):
    return - torch.log10(1 + torch.exp(-reward * pred))
