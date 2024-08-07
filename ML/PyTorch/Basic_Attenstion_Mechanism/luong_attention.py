""" 
    LUONG ATTENTION 

"""
from bahdanau_attention import * 
from logic import *
import torch.nn as nn 


class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, encoder_output):
        scores = self.W(encoder_output)
        scores = torch.sum(hidden * scores, dim=2)

        attention = F.softmax(scores, dim=-1)
        attention = attention.unsqueeze(1)
        context = torch.bmm(attention, encoder_output)

        return context, attention
    
attention = LuongAttention(HIDDEN_SIZE)
context, attention = attention_layer(hidden, output)

print(attention.size())