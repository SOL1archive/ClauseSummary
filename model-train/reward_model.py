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
            nn.Linear(768, hidden_size * 2, bias=False),
            nn.BatchNorm1d(hidden_size * 2),
            nn.GELU(),
            nn.Dropout1d(0.1),
            nn.Linear(hidden_size * 2, hidden_size, bias=False),
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
    
    def load(self, path):
        self.head1.load_state_dict(torch.load(path + '-head1.pt'))
        self.head2.load_state_dict(torch.load(path + '-head2.pt'))
        self.encoder = AutoModel.from_pretrained(path + '-encoder')
    
class DummyHeadModel(nn.Module):
    def __init__(self, hidden_size=256, num_labels=3):
        super(DummyHeadModel, self).__init__()
        self.hidden_size = hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, num_labels),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.head(x)
        return x

def reference_reward_loss(reward, pred):
    return - torch.log10(1 + torch.exp(-reward * pred))

class AMSoftmaxLoss(nn.Module):
    def __init__(self, in_features, n_classes, scale=30, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.linear = nn.Linear(in_features, n_classes, bias=False)
        self.scale = scale
        self.margin = margin
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, ouput, target):
        x_vector = F.normalize(ouput, p=2, dim=-1)
        self.linear.weight.data = F.normalize(self.linear.weight.data, p=2, dim=-1)
        logits = self.linear(x_vector)
        scaled_logits = (logits - self.margin)*self.scale
        logits = scaled_logits - self._am_logsumexp(logits)
        loss = self.cross_entropy(logits, target)
        return loss

    def _am_logsumexp(self, logits):
        max_x = torch.max(logits, dim=-1)[0].unsqueeze(-1)
        term1 = (self.scale * (logits - (max_x + self.margin))).exp()
        term2 = (self.scale * (logits - max_x)).exp().sum(-1).unsqueeze(-1) \
                - (self.scale * (logits - max_x)).exp()
        return self.scale * max_x + (term2 + term1).log()
    