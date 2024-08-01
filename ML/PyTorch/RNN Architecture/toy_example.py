import torch
import torch.nn as nn

BATCH_SIZE = 32
HIDDEN_SIZE = 10
VOCAB_SIZE = 1000
OUTPUT_VECT_NUM = 15

input_tensor = torch.randn(BATCH_SIZE, OUTPUT_VECT_NUM, HIDDEN_SIZE)
print(input_tensor.size())

linear_layer = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

out = linear_layer(input_tensor)
print(out.size())

_, indexes = out.topk(1)
print(indexes.size())

""" SQUEEZE FUNCTION """
print(indexes.squeeze().size())

import torch.nn.functional as F 

print(F.log_softmax(out, dim=-1).sum(-1))

