
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn

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

        context, attn = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(context))
        return output

class MuonOptimizer(nn.Module):
    def __init__(self, d_model, num_heads, tau=1.0):
        super(MuonOptimizer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.tau = tau

    def get_scaled_q_k(self, Q, K):
        Q_proj = self.attention.split_heads(self.attention.W_q(Q))
        K_proj = self.attention.split_heads(self.attention.W_k(K))

        scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1))
        S_max_per_head = scores.view(scores.size(0), scores.size(1), -1).max(dim=-1)[0]

        for i in range(self.attention.num_heads):
            S_max = S_max_per_head[:, i].max()
            if S_max > self.tau:
                gamma = self.tau / S_max
                Q_proj[:, i, :, :] *= torch.sqrt(gamma)
                K_proj[:, i, :, :] *= torch.sqrt(gamma)
        return Q_proj, K_proj

    def forward(self, Q, K, V, mask=None):
        Q_proj, K_proj = self.get_scaled_q_k(Q, K)
        V_proj = self.attention.split_heads(self.attention.W_v(V))
        context, attn = self.attention.scaled_dot_product_attention(Q_proj, K_proj, V_proj, mask)
        output = self.attention.W_o(self.attention.combine_heads(context))
        return output
