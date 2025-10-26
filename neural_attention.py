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
        Calculates attention scores using a feed-forward network in a memory-efficient way.
        """
        batch_size, num_heads, seq_len, d_k = Q.size()

        # Initialize an empty tensor to store the scores
        scores = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=Q.device, dtype=Q.dtype)

        # Iterate over each query vector to calculate its attention scores against all key vectors
        for i in range(seq_len):
            # Get the i-th query vector across all batches and heads
            Q_i = Q[:, :, i, :].unsqueeze(2)  # Shape: (batch_size, num_heads, 1, d_k)

            # Expand the i-th query vector to match the sequence length of the keys
            Q_i_expanded = Q_i.expand(-1, -1, seq_len, -1) # Shape: (batch_size, num_heads, seq_len, d_k)

            # Concatenate the expanded query vector with all key vectors
            # K has shape: (batch_size, num_heads, seq_len, d_k)
            cat_qk = torch.cat([Q_i_expanded, K], dim=-1) # Shape: (batch_size, num_heads, seq_len, 2 * d_k)

            # Pass the concatenated tensor through the FFN. The output contains the attention scores
            # of the i-th query with respect to all keys.
            score_i = self.ffn(cat_qk).squeeze(-1) # Shape: (batch_size, num_heads, seq_len)

            # Assign the calculated scores to the appropriate row in the final scores matrix
            scores[:, :, i, :] = score_i

        return scores
