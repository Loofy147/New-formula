
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention mechanism"""

    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert d_model > 0 and num_heads > 0, "d_model and num_heads must be positive"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32, device=Q.device)
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split into multiple heads: (batch, seq, d_model) -> (batch, heads, seq, d_k)"""
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine heads: (batch, heads, seq, d_k) -> (batch, seq, d_model)"""
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through multi-head attention"""
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        context, attn = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(context))
        return output


class MuonOptimizerFixed(nn.Module):
    """
    Fixed Muon Optimizer with per-sample scaling and differentiable operations
    """
    def __init__(self, d_model: int, num_heads: int, tau: float = 1.0, eps: float = 1e-8):
        super(MuonOptimizerFixed, self).__init__()

        assert isinstance(d_model, int) and d_model > 0
        assert isinstance(num_heads, int) and num_heads > 0
        assert d_model % num_heads == 0
        assert isinstance(tau, (int, float)) and tau > 0
        assert isinstance(eps, float) and eps > 0

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.tau = float(tau)
        self.eps = eps

    def validate_inputs(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        """Validate input tensors"""
        assert Q.dim() == 3, f"Q must be 3D (batch, seq, d_model), got {Q.dim()}D"
        assert K.dim() == 3, f"K must be 3D (batch, seq, d_model), got {K.dim()}D"
        assert V.dim() == 3, f"V must be 3D (batch, seq, d_model), got {V.dim()}D"
        assert Q.shape == K.shape == V.shape, \
            f"Q, K, V shapes must match, got Q:{Q.shape}, K:{K.shape}, V:{V.shape}"
        assert Q.size(-1) == self.attention.d_model, \
            f"Expected d_model={self.attention.d_model}, got {Q.size(-1)}"
        assert not torch.isnan(Q).any(), "Q contains NaN values"
        assert not torch.isnan(K).any(), "K contains NaN values"
        assert not torch.isnan(V).any(), "V contains NaN values"
        assert not torch.isinf(Q).any(), "Q contains Inf values"
        assert not torch.isinf(K).any(), "K contains Inf values"
        assert not torch.isinf(V).any(), "V contains Inf values"

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if self.training:
            self.validate_inputs(Q, K, V)

        Q_proj = self.attention.split_heads(self.attention.W_q(Q))
        K_proj = self.attention.split_heads(self.attention.W_k(K))
        V_proj = self.attention.split_heads(self.attention.W_v(V))

        scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1))

        if mask is not None:
            scores_for_max = scores.masked_fill(mask == 0, -1e9)
        else:
            scores_for_max = scores

        S_max_per_head = scores_for_max.view(
            scores.size(0),
            scores.size(1),
            -1
        ).max(dim=-1)[0]

        gamma = torch.clamp(self.tau / (S_max_per_head + self.eps), max=1.0)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)

        scaled_scores = scores * torch.sqrt(gamma + self.eps)

        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scaled_scores, dim=-1)
        context = torch.matmul(attn, V_proj)

        output = self.attention.W_o(self.attention.combine_heads(context))
        return output

# ... (rest of the file is the same)
