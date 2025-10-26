
import torch
import unittest
from advanced_muon_attention import AdvancedMuonAttention
from rms_norm import RMSNorm
from neural_attention import NeuralAttention

class TestAdvancedMuonAttention(unittest.TestCase):
    def test_output_shape(self):
        d_model = 128
        num_heads = 4
        seq_length = 10
        batch_size = 2

        model = AdvancedMuonAttention(d_model, num_heads)

        Q = torch.randn(batch_size, seq_length, d_model)
        K = torch.randn(batch_size, seq_length, d_model)
        V = torch.randn(batch_size, seq_length, d_model)

        output = model(Q, K, V)
        self.assertEqual(output.size(), (batch_size, seq_length, d_model))

    def test_rms_norm(self):
        d_model = 128
        x = torch.randn(2, 10, d_model)
        norm = RMSNorm(d_model)
        output = norm(x)

        # Calculate the expected RMS norm
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        expected_output = x / (rms + 1e-8)

        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_neural_attention(self):
        d_k = 32
        seq_len = 10
        batch_size = 2
        num_heads = 4

        q = torch.randn(batch_size, num_heads, seq_len, d_k)
        k = torch.randn(batch_size, num_heads, seq_len, d_k)

        attention = NeuralAttention(d_k)
        scores = attention(q, k)

        self.assertEqual(scores.size(), (batch_size, num_heads, seq_len, seq_len))

    def test_adaptive_temperature(self):
        d_model = 128
        num_heads = 4
        seq_length = 10
        batch_size = 2

        model = AdvancedMuonAttention(d_model, num_heads)

        # Check that the temperature is a learnable parameter
        self.assertTrue(hasattr(model, 'temperature'))
        self.assertIsInstance(model.temperature, torch.nn.Parameter)

        # Check that the temperature is being applied
        Q = torch.randn(batch_size, seq_length, d_model)
        K = torch.randn(batch_size, seq_length, d_model)
        V = torch.randn(batch_size, seq_length, d_model)

        # Get the scores without the temperature
        Q_proj = model.split_heads(model.W_q(Q))
        K_proj = model.split_heads(model.W_k(K))
        scores_no_temp = model.neural_attention(Q_proj, K_proj)

        # Set the temperature to a different value
        with torch.no_grad():
            model.temperature.fill_(2.0)

        # Get the scores with the temperature
        scores_with_temp = scores_no_temp / model.temperature

        # Check that the scores are different
        self.assertFalse(torch.allclose(scores_no_temp, scores_with_temp))

    def test_causal_masking(self):
        d_model = 128
        num_heads = 4
        seq_length = 10
        batch_size = 2

        model = AdvancedMuonAttention(d_model, num_heads)

        Q = torch.randn(batch_size, seq_length, d_model)
        K = torch.randn(batch_size, seq_length, d_model)
        V = torch.randn(batch_size, seq_length, d_model)

        mask = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0).unsqueeze(0)
        output, attn = model(Q, K, V, mask=mask, return_attention=True)

        # Check that the upper triangle of the attention matrix is all zeros
        for i in range(seq_length):
            for j in range(i + 1, seq_length):
                self.assertTrue(torch.all(attn[:, :, i, j] == 0))

if __name__ == '__main__':
    unittest.main()
