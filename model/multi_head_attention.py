import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Learned linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Final output linear layer
        self.fc_out = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, queries, keys, values):
        scores = queries @ keys.transpose(-2, -1) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ values
        return output, weights
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 1. Linear projections
        Q = self.w_q(x)  # (batch_size, seq_len, d_model)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # 2. Split into heads and reshape
        # from (batch_size, seq_len, d_model) to (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply scaled dot-product attention on each head
        out, attn_weights = self.scaled_dot_product_attention(Q, K, V)
        # out shape: (batch_size, num_heads, seq_len, d_k)
        
        # 4. Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 5. Final linear layer
        out = self.fc_out(out)
        
        return out, attn_weights
