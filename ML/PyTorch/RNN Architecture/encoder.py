"""
    Encoder Architecture
"""

import torch.nn as nn
from main import words2index, convert2tensors

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
    def forward(self, input_tensor):
        embedded = self.embedding(input_tensor)
        output, hidden = self.gru(embedded)
        return output, hidden
    
encoder = EncoderRNN(input_size=len(words2index), hidden_size=3)

sentence = "How are you doing?"
input_tensor = convert2tensors(sentence)
output, hidden = encoder(input_tensor)

print(output.size())
print(hidden.size())


    