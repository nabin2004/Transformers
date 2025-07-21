import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# batch_size = 2
# seq_len = 5
# d_model = 512
# d_ff = 2048

# ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)

# input_tensor = torch.randn(batch_size, seq_len, d_model)
# output_tensor = ffn(input_tensor)

# print(output_tensor.shape)  # Expected: (2, 5, 512)
