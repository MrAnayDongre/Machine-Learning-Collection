""" 
    BAHDANAU ATTENTION

"""

from logic import *
import torch.nn as nn 
import torch.nn.functional as F 

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        
        
    def forward(self, hidden, encoder_output):
        scores = torch.tanh(self.W(hidden) + self.U(encoder_output))
        scores = self.V(scores)
        scores = scores.squeeze(2).unsqueeze(1)
        
        attention = F.softmax(scores, dim=-1)
        context = torch.bmm(attention, encoder_output)
        
        return context, attention
    
attention_layer = BahdanauAttention(HIDDEN_SIZE)
context, attention = attention_layer(hidden, output)

print(attention)
print(context.size())

