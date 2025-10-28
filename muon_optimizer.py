import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import unittest
import time

class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention mechanism (batch-first)"""
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

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) -> (batch, heads, seq, d_k)"""
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, heads, seq, d_k) -> (batch, seq, d_model)"""
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        d_k = float(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, device=Q.device, dtype=Q.dtype))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        scores = scores - scores.amax(dim=-1, keepdim=True)
        attn = F.softmax(scores, dim=-1)
        attn = attn.clamp(min=1e-12)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        context = torch.matmul(attn, V)
        output = self.W_o(self.combine_heads(context))
        return output


class MuonOptimizerFixed(nn.Module):
    def __init__(self, d_model: int, num_heads: int, tau: float = 1.0):
        super(MuonOptimizerFixed, self).__init__()
        assert tau > 0, "tau must be positive"
        self.d_model = d_model
        self.num_heads = num_heads
        self.tau = float(tau)
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.size()
        return x.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, s, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(b, s, self.d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Qh = self.split_heads(self.W_q(Q))
        Kh = self.split_heads(self.W_k(K))
        Vh = self.split_heads(self.W_v(V))

        device = Qh.device
        dtype = Qh.dtype
        scale = torch.sqrt(torch.tensor(float(self.d_k), device=device, dtype=dtype))

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / scale

        row_max = scores.amax(dim=-1, keepdim=True)
        scores = scores - row_max
        scores = scores.clamp(min=-1e9, max=1e9)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn = F.softmax(scores, dim=-1)
        attn = attn.clamp(min=1e-12)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        context = torch.matmul(attn, Vh)
        out = self.W_o(self.combine_heads(context))
        return out


class MuonOptimizerDifferentiable(nn.Module):
    def __init__(self, d_model: int, num_heads: int, tau: float = 1.0):
        super(MuonOptimizerDifferentiable, self).__init__()
        assert tau > 0, "tau must be positive"
        self.d_model = d_model
        self.num_heads = num_heads
        self.tau = float(tau)
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.size()
        return x.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, s, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(b, s, self.d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Qh = self.split_heads(self.W_q(Q))
        Kh = self.split_heads(self.W_k(K))
        Vh = self.split_heads(self.W_v(V))

        device = Qh.device
        dtype = Qh.dtype
        scale = torch.sqrt(torch.tensor(float(self.d_k), device=device, dtype=dtype))

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / scale

        scores = scores / self.tau
        scores = scores - scores.amax(dim=-1, keepdim=True)
        scores = scores.clamp(min=-1e9, max=1e9)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn = F.softmax(scores, dim=-1)
        attn = attn.clamp(min=1e-12)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        context = torch.matmul(attn, Vh)
        out = self.W_o(self.combine_heads(context))
        return out


class TestMuonOptimizers(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8
        self.seq_length = 16
        self.d_model = 128
        self.num_heads = 4
        self.tau = 1.0

        self.model_fixed = MuonOptimizerFixed(self.d_model, self.num_heads, self.tau)
        self.model_smooth = MuonOptimizerDifferentiable(self.d_model, self.num_heads, self.tau)

        torch.manual_seed(0)
        self.Q = torch.randn(self.batch_size, self.seq_length, self.d_model)
        self.K = torch.randn(self.batch_size, self.seq_length, self.d_model)
        self.V = torch.randn(self.batch_size, self.seq_length, self.d_model)

    def test_batch_poisoning_resistance(self):
        print("\n🛡️  Testing Batch Poisoning Resistance...")
        Q_poison = torch.randn(self.batch_size, self.seq_length, self.d_model) * 0.1
        Q_normal = Q_poison.clone()
        Q_poison[0] *= 1000.0

        output_normal = self.model_fixed(Q_normal, self.K, self.V)
        output_poisoned = self.model_fixed(Q_poison, self.K, self.V)

        self.assertTrue(torch.allclose(output_normal[1:], output_poisoned[1:], atol=1e-6))
        self.assertFalse(torch.allclose(output_normal[0], output_poisoned[0], atol=1e-6))
        print("   ✅ No batch contamination!")

    def test_edge_cases(self):
        print("\n🔍 Edge Case Testing...")
        models = [self.model_fixed, self.model_smooth]
        for model in models:
            Q_small = torch.randn(2, self.seq_length, self.d_model) * 1e-9
            out_small = model(Q_small, Q_small, self.V[:2])
            self.assertFalse(torch.isnan(out_small).any())

            Q_large = torch.randn(2, self.seq_length, self.d_model) * 1e3
            out_large = model(Q_large, Q_large, self.V[:2])
            self.assertFalse(torch.isnan(out_large).any())
        print("   ✅ Edge cases handled!")

if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
