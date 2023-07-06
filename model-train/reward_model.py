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

class AMSoftmaxLoss(nn.Module):
    def __init__(self, embedding_dim, no_classes, scale = 30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.no_classes = no_classes
        self.embedding = nn.Embedding(no_classes, embedding_dim, max_norm=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        n, m = x.shape        
        assert n == len(labels)
        assert m == self.embedding_dim
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.no_classes

        x = F.normalize(x, dim=1)
        w = self.embedding.weight        
        cos_theta = torch.matmul(w, x.T).T
        psi = cos_theta - self.margin
        
        onehot = F.one_hot(labels, self.no_classes)
        logits = self.scale * torch.where(onehot == 1, psi, cos_theta)        
        loss = self.loss(logits, labels)
        
        return loss, logits
    