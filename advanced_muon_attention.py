
import torch
import torch.nn as nn
import torch.nn.functional as F
from rms_norm import RMSNorm
from neural_attention import NeuralAttention

class AdvancedMuonAttention(nn.Module):
    """
    Implements an advanced multi-head attention mechanism with several enhancements.
    This module can be configured for ablation studies to evaluate the impact of each component.
    """
    def __init__(self, d_model, num_heads, use_neural_attention=True, use_rms_norm=True, use_adaptive_temperature=True):
        super(AdvancedMuonAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.use_neural_attention = use_neural_attention
        if self.use_neural_attention:
            self.neural_attention = NeuralAttention(self.d_k)

        self.use_adaptive_temperature = use_adaptive_temperature
        if self.use_adaptive_temperature:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.use_rms_norm = use_rms_norm
        if self.use_rms_norm:
            self.norm = RMSNorm(d_model)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None, return_attention=False):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        if self.use_neural_attention:
            scores = self.neural_attention(Q, K)
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if self.use_adaptive_temperature:
            scores = scores / self.temperature

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        output = self.W_o(self.combine_heads(context))

        if self.use_rms_norm:
            output = self.norm(output)

        if return_attention:
            return output, attn
        else:
            return output
