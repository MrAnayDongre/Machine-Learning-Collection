"""  
    IMPLEMENTING SELF ATTENTION MECHANISM 

"""

import torch
import torch.nn as nn 
import torch.nn.functional as F 

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        queries = self.Q(x)
        keys = self.K(x)
        values = self.V(x)
        scores = torch.bmm(queries, keys.transpose(1, 2))
        scores = scores / (self.hidden_size ** 0.5) 
        attention = F.softmax(scores, dim=2)
        hidden_states = torch.bmm(attention, values)
        return hidden_states

    
SOS_token = 0
EOS_token = 1

index2words = {
    SOS_token: 'SOS',
    EOS_token: 'EOS'
}

words = "How are you doing ?"
words_list = set(words.lower().split(' '))
for word in words_list:
    index2words[len(index2words)] = word
    
words2index = {w: i for i, w in index2words.items()}
    
def convert2tensors(sentece):
    words_list = sentece.lower().split(' ')
    indexes = [words2index[word] for word in words_list]
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)

HIDDEN_SIZE = 10
VOCAB_SIZE = len(words2index)

embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
attention = Attention(HIDDEN_SIZE)


sentence = "How are you doing ?"
input_tensor = convert2tensors(sentence)
embedded = embedding(input_tensor)
hidden_states = attention(embedded)
print(hidden_states.size())