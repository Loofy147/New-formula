
import time
import torch
import torch.nn as nn
from advanced_muon_attention import AdvancedMuonAttention
import math
import requests
from collections import Counter

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# ----------------------------------------------------------------------------
# Model Definitions
# ----------------------------------------------------------------------------
class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5, attention_module=nn.MultiheadAttention, attention_config={}):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        if attention_module == nn.MultiheadAttention:
            self.transformer_encoder = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            self.transformer_encoder = attention_module(d_model, nhead, **attention_config)

        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
        self.attention_module = attention_module
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        if self.attention_module == nn.MultiheadAttention:
            output, _ = self.transformer_encoder(src, src, src, attn_mask=src_mask)
        else:
            src = src.permute(1, 0, 2)
            mask_expanded = src_mask.unsqueeze(0).unsqueeze(0) if src_mask is not None else None
            output = self.transformer_encoder(src, src, src, mask=mask_expanded)
            output = output.permute(1, 0, 2)

        output = self.decoder(output)
        return output

# ----------------------------------------------------------------------------
# Data Loading (Manual Implementation)
# ----------------------------------------------------------------------------
def download_data(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def build_vocab(text):
    words = text.split()
    word_counts = Counter(words)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
    word_to_idx['<unk>'] = 0
    return word_to_idx

def text_to_sequence(text, word_to_idx):
    words = text.split()
    return [word_to_idx.get(word, 0) for word in words]

def get_data_iter(batch_size, bptt):
    base_url = "https://cosmo.zip/pub/datasets/wikitext-2-raw/"
    train_text = download_data(base_url + "wiki.train.raw")
    valid_text = download_data(base_url + "wiki.valid.raw")

    train_text = " ".join(train_text.split()[:10000])
    valid_text = " ".join(valid_text.split()[:1000])

    word_to_idx = build_vocab(train_text)

    train_data = torch.tensor(text_to_sequence(train_text, word_to_idx), dtype=torch.long)
    val_data = torch.tensor(text_to_sequence(valid_text, word_to_idx), dtype=torch.long)

    def batchify(data, bsz):
        nbatch = data.size(0) // bsz
        data = data.narrow(0, 0, nbatch * bsz)
        data = data.view(bsz, -1).t().contiguous()
        return data

    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, batch_size)

    return train_data, val_data, word_to_idx

# ----------------------------------------------------------------------------
# Training and Evaluation
# ----------------------------------------------------------------------------
def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_binary_mask(sz):
    mask = torch.tril(torch.ones(sz, sz))
    return mask

def train_and_evaluate(model, train_data, val_data, ntokens, bptt):
    criterion = nn.CrossEntropyLoss()
    lr = 5.0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    total_loss = 0.

    for i in range(0, train_data.size(0) - 1, bptt):
        data, targets = get_batch(train_data, i, bptt)
        if model.attention_module == nn.MultiheadAttention:
            src_mask = generate_square_subsequent_mask(data.size(0))
        else:
            src_mask = generate_binary_mask(data.size(0))
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    total_val_loss = 0.
    with torch.no_grad():
        for i in range(0, val_data.size(0) - 1, bptt):
            data, targets = get_batch(val_data, i, bptt)
            if model.attention_module == nn.MultiheadAttention:
                src_mask = generate_square_subsequent_mask(data.size(0))
            else:
                src_mask = generate_binary_mask(data.size(0))
            output = model(data, src_mask)
            total_val_loss += len(data) * criterion(output.view(-1, ntokens), targets).item()

    return total_val_loss / (len(val_data) - 1)

# ----------------------------------------------------------------------------
# Main Benchmarking Function
# ----------------------------------------------------------------------------
def main():
    batch_size = 20
    bptt = 35
    train_data, val_data, vocab = get_data_iter(batch_size, bptt)
    ntokens = len(vocab)
    d_model = 200
    nhead = 2
    dropout = 0.2

    # --- Standard Model ---
    print("--- Training Standard Transformer ---")
    model_standard = TransformerModel(ntokens, d_model, nhead, d_hid=d_model, nlayers=1, dropout=dropout).to("cpu")
    val_loss = train_and_evaluate(model_standard, train_data, val_data, ntokens, bptt)
    print(f"Validation PPL: {math.exp(val_loss):.2f}")
    print("-" * 50)

    # --- Ablation Studies ---
    ablation_configs = [
        {'use_neural_attention': True, 'use_rms_norm': True, 'use_adaptive_temperature': True},
        {'use_neural_attention': False, 'use_rms_norm': True, 'use_adaptive_temperature': True},
        {'use_neural_attention': True, 'use_rms_norm': False, 'use_adaptive_temperature': True},
        {'use_neural_attention': True, 'use_rms_norm': True, 'use_adaptive_temperature': False},
    ]

    for config in ablation_configs:
        print(f"\n--- Training Ablation Model: {config} ---")
        model_ablation = TransformerModel(ntokens, d_model, nhead, d_hid=d_model, nlayers=1, dropout=dropout,
                                          attention_module=AdvancedMuonAttention, attention_config=config).to("cpu")
        val_loss = train_and_evaluate(model_ablation, train_data, val_data, ntokens, bptt)
        print(f"Validation PPL: {math.exp(val_loss):.2f}")
        print("-" * 50)

if __name__ == '__main__':
    main()
