
SOS_token = 0
EOS_token = 1

index2words = {
    SOS_token: 'SOS',
    EOS_token: 'EOS'
}

words = "How are you doing ? I am good and you ?"
words_list = set(words.lower().split(' '))
for word in words_list:
    index2words[len(index2words)] = word 
    
    
#print(index2words)

words2index = {w: i for i, w in index2words.items()}
#print(word2index)

import torch

def convert2tensor(sentece):
    words_list = sentece.lower().split(' ')
    indexes = [words2index[word] for word in words_list]
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)

import torch.nn as nn 

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
    def forward(self, input_tensor):
        embedded = self.embedding(input_tensor)
        output, hidden = self.gru(embedded)
        return output, hidden 
    
HIDDEN_SIZE = 10
VOCAB_SIZE = len(words2index)

encoder = EncoderRNN(
    input_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE
)

sentence = "How are you doing ?"
input_tensor = convert2tensor(sentence)
output, hidden = encoder(input_tensor)
print(output.size())
print(hidden.size())