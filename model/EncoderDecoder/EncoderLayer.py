import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),  # or GELU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # LayerNorm and Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        # Self-attention block
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout1(attn_output)     # Residual connection
        x = self.norm1(x)                      # LayerNorm
        
        # Feedforward block
        ff_output = self.ffn(x)
        x = x + self.dropout2(ff_output)       # Residual connection
        x = self.norm2(x)                      # LayerNorm
        return x

# batch_size = 32
# seq_len = 10
# d_model = 512
# num_heads = 8
# d_ff = 2048

# encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
# x = torch.rand(batch_size, seq_len, d_model)
# output = encoder_layer(x)
# print(output.shape) 