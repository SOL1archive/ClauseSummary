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
        self.head1 = nn.Sequential(
            nn.Linear(768, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
        )
        self.head2 = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids=None, attention_mask=None):
        x = self.encoder(input_ids, attention_mask).pooler_output
        x = self.head1(x)
        x = self.head2(x)
        return x
    
    def representation_forward(self, input_ids=None, attention_mask=None):
        x = self.encoder(input_ids, attention_mask).pooler_output
        x = self.head1(x)
        return x
    
class DummyHeadModel(nn.Module):
    def __init__(self, hidden_size=256, num_labels=3):
        super(DummyHeadModel, self).__init__()
        self.hidden_size = hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, num_labels),
            nn.Softmax(dim=-1),
        )

def reference_reward_loss(reward, pred):
    return - torch.log10(1 + torch.exp(-reward * pred))

class AMSoftmaxLoss:
    def __init__(self, margin):
        super(AMSoftmaxLoss, self).__init__()
        self.margin = margin

    def forward(self, x, labels):
        labels = labels.float()
        loss = - torch.log(torch.exp(labels * x + self.margin) 
                           / torch.sum(torch.exp(x + self.margin))
        )
        return loss
    