
import torch
import unittest
from muon_transformer import MuonOptimizer

class TestMuonOptimizer(unittest.TestCase):
    def test_per_head_qk_clip(self):
        d_model = 128
        num_heads = 4
        seq_length = 10
        batch_size = 2
        tau = 1.0

        model = MuonOptimizer(d_model, num_heads, tau)

        # Set weights to identity for test isolation
        with torch.no_grad():
            model.attention.W_q.weight.data = torch.eye(d_model)
            model.attention.W_k.weight.data = torch.eye(d_model)
            model.attention.W_q.bias.data.zero_()
            model.attention.W_k.bias.data.zero_()

        # Create input tensors
        Q = torch.randn(batch_size, seq_length, d_model) * 0.1
        K = torch.randn(batch_size, seq_length, d_model) * 0.1

        # Make one head have a large S_max
        with torch.no_grad():
            Q[:, 0, 0:32] = 100.0
            K[:, 0, 0:32] = 100.0

        Q_proj_before = model.attention.split_heads(model.attention.W_q(Q))
        K_proj_before = model.attention.split_heads(model.attention.W_k(K))

        Q_proj_after, K_proj_after = model.get_scaled_q_k(Q, K)

        # Check that the first head is scaled
        self.assertFalse(torch.allclose(Q_proj_before[:, 0, :, :], Q_proj_after[:, 0, :, :]))
        self.assertFalse(torch.allclose(K_proj_before[:, 0, :, :], K_proj_after[:, 0, :, :]))

        # Check that the other heads are not scaled
        for i in range(1, num_heads):
            self.assertTrue(torch.allclose(Q_proj_before[:, i, :, :], Q_proj_after[:, i, :, :]))
            self.assertTrue(torch.allclose(K_proj_before[:, i, :, :], K_proj_after[:, i, :, :]))

    def test_multi_head_attention(self):
        d_model = 128
        num_heads = 4
        seq_length = 10
        batch_size = 2

        model = MuonOptimizer(d_model, num_heads)

        # Create input tensors
        Q = torch.randn(batch_size, seq_length, d_model)
        K = torch.randn(batch_size, seq_length, d_model)
        V = torch.randn(batch_size, seq_length, d_model)

        output = model(Q, K, V)

        self.assertEqual(output.size(), (batch_size, seq_length, d_model))

if __name__ == '__main__':
    unittest.main()
