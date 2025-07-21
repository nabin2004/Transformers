import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # Masked Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Encoder-Decoder Cross Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Feedforward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),  # or GELU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Normalization & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # Masked Self-Attention
        self_attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = x + self.dropout1(self_attn_output)
        x = self.norm1(x)

        # Encoder-Decoder Cross-Attention
        cross_attn_output, _ = self.cross_attn(x, enc_output, enc_output, attn_mask=memory_mask)
        x = x + self.dropout2(cross_attn_output)
        x = self.norm2(x)

        # Feedforward Network
        ff_output = self.ffn(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)

        return x

# batch_size = 32
# tgt_seq_len = 20
# src_seq_len = 15
# d_model = 512
# num_heads = 8
# d_ff = 2048

# decoder_layer = DecoderLayer(d_model, num_heads, d_ff)

# tgt = torch.rand(batch_size, tgt_seq_len, d_model)
# memory = torch.rand(batch_size, src_seq_len, d_model)

# output = decoder_layer(tgt, memory)
# print(output.shape)  # (32, 20, 512)