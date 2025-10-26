
import time
import torch
import torch.nn as nn
import math
import requests
from collections import Counter
from muon_optimizer import MuonOptimizerFixed

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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------
class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers=1, dropout=0.5, attention_module=nn.MultiheadAttention):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        if attention_module == nn.MultiheadAttention:
            self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.use_multihead = True
        else:
            self.attention = attention_module(d_model, nhead)
            self.use_multihead = False

        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
        self.init_weights()
        try:
            self.decoder.weight = self.encoder.weight
        except Exception:
            pass

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        if self.use_multihead:
            output, _ = self.attention(src, src, src, attn_mask=src_mask)
        else:
            output = self.attention(src, src, src, mask=src_mask)

        output = self.decoder(output)
        return output

# ----------------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------------
def download_data(url, timeout=10):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text

def build_vocab(text, max_vocab=None):
    words = text.split()
    word_counts = Counter(words)
    most_common = word_counts.most_common(max_vocab)
    vocab = [w for w, _ in most_common]
    word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
    word_to_idx['<unk>'] = 0
    return word_to_idx

def text_to_sequence(text, word_to_idx):
    words = text.split()
    return [word_to_idx.get(word, 0) for word in words]

def get_data_iter(batch_size, bptt):
    base_url = "https://cosmo.zip/pub/datasets/wikitext-2-raw/"
    try:
        train_text = download_data(base_url + "wiki.train.raw")
        valid_text = download_data(base_url + "wiki.valid.raw")
    except Exception as e:
        raise RuntimeError("Failed to download dataset.") from e

    train_text = " ".join(train_text.split()[:10000])
    valid_text = " ".join(valid_text.split()[:1000])

    word_to_idx = build_vocab(train_text)
    train_seq = torch.tensor(text_to_sequence(train_text, word_to_idx), dtype=torch.long)
    val_seq = torch.tensor(text_to_sequence(valid_text, word_to_idx), dtype=torch.long)

    def batchify(data, bsz):
        nbatch = data.size(0) // bsz
        data = data.narrow(0, 0, nbatch * bsz)
        data = data.view(bsz, -1).contiguous()
        return data

    train_data = batchify(train_seq, batch_size)
    val_data = batchify(val_seq, batch_size)
    return train_data, val_data, word_to_idx

# ----------------------------------------------------------------------------
# Training and Evaluation
# ----------------------------------------------------------------------------
def get_batch(source, i, bptt):
    seq_len = min(bptt, source.size(1) - 1 - i)
    data = source[:, i:i+seq_len]
    target = source[:, i+1:i+1+seq_len].reshape(-1)
    return data, target

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    attn_mask = torch.zeros(sz, sz, dtype=torch.float)
    attn_mask.masked_fill_(mask, float('-inf'))
    return attn_mask

def train_and_evaluate(model, train_data, val_data, ntokens, bptt, device, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for i in range(0, train_data.size(1) - 1, bptt):
            data, targets = get_batch(train_data, i, bptt)
            data = data.to(device)
            targets = targets.to(device)
            seq_len = data.size(1)
            src_mask = generate_square_subsequent_mask(seq_len).to(device)

            output = model(data, src_mask)
            loss = criterion(output.view(-1, ntokens), targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item() * data.size(0) * seq_len
            count += data.size(0) * seq_len

        scheduler.step()
        avg_train_loss = total_loss / max(1, count)
        model.eval()
        total_val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for i in range(0, val_data.size(1) - 1, bptt):
                data, targets = get_batch(val_data, i, bptt)
                data = data.to(device)
                targets = targets.to(device)
                seq_len = data.size(1)
                src_mask = generate_square_subsequent_mask(seq_len).to(device)
                output = model(data, src_mask)
                loss = criterion(output.view(-1, ntokens), targets)
                total_val_loss += loss.item() * data.size(0) * seq_len
                val_count += data.size(0) * seq_len

        avg_val_loss = total_val_loss / max(1, val_count)
        print(f"Epoch {epoch}: Train loss {avg_train_loss:.4f}, Val loss {avg_val_loss:.4f}, Val PPL {math.exp(avg_val_loss):.2f}")

    return avg_val_loss

# ----------------------------------------------------------------------------
# Main Benchmarking Function
# ----------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 20
    bptt = 35
    train_data, val_data, vocab = get_data_iter(batch_size, bptt)
    ntokens = len(vocab)
    d_model = 200
    nhead = 2
    dropout = 0.2

    print("--- Training Standard Transformer ---")
    model_standard = TransformerModel(ntokens, d_model, nhead, d_hid=d_model, nlayers=1, dropout=dropout, attention_module=nn.MultiheadAttention)
    val_loss = train_and_evaluate(model_standard, train_data, val_data, ntokens, bptt, device, epochs=3)
    print(f"Validation PPL: {math.exp(val_loss):.2f}")
    print("-" * 50)

    print("\n--- Training Muon Optimizer Transformer ---")
    model_muon = TransformerModel(ntokens, d_model, nhead, d_hid=d_model, nlayers=1, dropout=dropout, attention_module=MuonOptimizerFixed)
    val_loss = train_and_evaluate(model_muon, train_data, val_data, ntokens, bptt, device, epochs=3)
    print(f"Validation PPL: {math.exp(val_loss):.2f}")
    print("-" * 50)

if __name__ == '__main__':
    main()
