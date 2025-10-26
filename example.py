
import torch
from muon_optimizer import MuonOptimizerFixed

def main():
    """
    Provides a simple example of how to use the MuonOptimizerFixed module.
    """
    # Parameters
    d_model = 128
    num_heads = 4
    seq_length = 10
    batch_size = 2

    # Create the model
    model = MuonOptimizerFixed(d_model, num_heads)

    # Create some dummy data
    Q = torch.randn(batch_size, seq_length, d_model)
    K = torch.randn(batch_size, seq_length, d_model)
    V = torch.randn(batch_size, seq_length, d_model)

    # --- Example 1: Standard forward pass ---
    print("--- Example 1: Standard forward pass ---")
    output = model(Q, K, V)
    print("Output shape:", output.shape)
    print("-" * 40)

    # --- Example 2: Forward pass with causal masking ---
    print("--- Example 2: Forward pass with causal masking ---")
    mask = torch.tril(torch.ones(seq_length, seq_length)).bool()
    output_masked = model(Q, K, V, mask=mask)
    print("Output shape (masked):", output_masked.shape)
    print("-" * 40)

if __name__ == '__main__':
    main()
