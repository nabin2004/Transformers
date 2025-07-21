import torch 

def scaled_dot_product_attention(queries, keys, values):
    first_part = queries @ keys.transpose(-2, -1)
    d_k = queries.size(-1)
    scaled_attention_logits = first_part / d_k**0.5
    attention_weights = scaled_attention_logits.softmax(dim=-1)
    output = attention_weights @ values
    return output, attention_weights


queries = torch.tensor([[[1.0, 0.0]]])  # shape (1, 1, 2)
keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # shape (1, 2, 2)
values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # shape (1, 2, 2)

output, attn = scaled_dot_product_attention(queries, keys, values)

print("Attention weights:", attn)
print("Output:", output)