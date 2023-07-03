import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class ModelForRewardGeneration(nn.Module):
    def __init__(self, encoder_path, hidden_size=256):
        super(ModelForRewardGeneration, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
        self.hidden_size = hidden_size
        # TODO: head 설계
        self.head = nn.Sequential(
            nn.Linear(768, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout1d(0.1),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids=None, attention_mask=None):
        x = self.encoder(input_ids, attention_mask).pooler_output
        x = self.head(x)
        return x

def reference_reward_loss(reward, pred):
    return - torch.log10(1 + torch.exp(-reward * pred))