
import torch
import torch.nn as nn

class NeuralAttention(nn.Module):
    def __init__(self, d_k):
        super(NeuralAttention, self).__init__()
        self.d_k = d_k
        self.ffn = nn.Sequential(
            nn.Linear(d_k * 2, d_k * 4),
            nn.ReLU(),
            nn.Linear(d_k * 4, 1)
        )

    def forward(self, Q, K):
        """
        Calculates attention scores using a feed-forward network in a vectorized and memory-efficient way.
        """
        batch_size, num_heads, seq_len, d_k = Q.size()

        # Expand Q and K to create all pairs of query and key vectors
        Q_expanded = Q.unsqueeze(3).expand(-1, -1, -1, seq_len, -1)
        K_expanded = K.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)

        # Concatenate the expanded tensors
        cat_qk = torch.cat([Q_expanded, K_expanded], dim=-1)

        # Reshape for the FFN
        cat_qk = cat_qk.view(batch_size, num_heads, seq_len * seq_len, -1)

        # Pass through the FFN and reshape back to the desired scores shape
        scores = self.ffn(cat_qk).view(batch_size, num_heads, seq_len, seq_len)

        return scores
