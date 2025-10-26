
import torch
import torch.nn as nn
import torch.nn.functional as F
from rms_norm import RMSNorm
from neural_attention import NeuralAttention

class AdvancedMuonAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(AdvancedMuonAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.neural_attention = NeuralAttention(self.d_k)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm = RMSNorm(d_model)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        scores = self.neural_attention(Q, K) / self.temperature
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        output = self.W_o(self.combine_heads(context))
        return self.norm(output)
