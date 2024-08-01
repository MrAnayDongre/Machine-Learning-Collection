SOS_token = 0
EOS_token = 1


index2words = {
    SOS_token: 'SOS',
    EOS_token: 'EOS'
}

words = "How are you doing? I am good you?"
words_list = set(words.lower().split(' '))

for word in words_list:
    index2words[len(index2words)] = word
    
#print(index2words)
""" Reverse Mapping """

words2index = {w: i for i, w in index2words.items()}
#print(words2index)

""" Convert sentences into PyTorch tenors """
import torch
def convert2tensors(sentence):
    words_list = sentence.lower().split(' ')
    indexes = [words2index[word] for word in words_list]
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)
sentence = "How are you doing?"
convert2tensors(sentence)
#print(convert2tensors)
#> tensor([[10,  6,  3,  8,  2]])
