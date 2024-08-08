""" 
    IMPLEMENTING MULTIHEAD ATTENTION

"""

import torch.nn as nn 
from self_attention import *


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.out = nn.Linear(hidden_size * num_heads, hidden_size)
        self.heads = nn.ModuleList([
            Attention(hidden_size) 
            for _ in range(num_heads)
        ])
        
    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        outputs = torch.cat(outputs, dim=2)
        hidden_states = self.out(outputs)
        return hidden_states

NUM_HEADS = 3
multi_att = MultiheadAttention(HIDDEN_SIZE, NUM_HEADS)

sentence = "How are you doing ?"
input_tensor = convert2tensors(sentence)
embedded = embedding(input_tensor)
hidden_states = multi_att(embedded)

print(embedded)        