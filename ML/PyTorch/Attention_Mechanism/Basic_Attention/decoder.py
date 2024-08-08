""" 
    BUILDING DECODER

"""

from bahdanau_attention import *
from logic import *
from luong_attention import *

MAX_LENGTH = 10

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, encoder_outputs, encoder_hidden):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)

            _, topIdx = decoder_output.topk(1)
            decoder_input = topIdx.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs, decoder_hidden

    def forward_step(self, decoder_input, hidden, encoder_outputs):
        embedded = self.embedding(decoder_input)
        
        context, _ = self.attention(hidden, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden

    
    
""" 

    GENERATE OUTPUT SEQUENCE

"""

decoder = AttnDecoderRNN(
    hidden_size=HIDDEN_SIZE, 
    output_size=len(words2index)
)

sentence = "How are you doing ?"
input_tensor = convert2tensor(sentence)
output, hidden = encoder(input_tensor)
decoder_outputs, decoder_hidden = decoder(output, hidden)

def convert2sentence(tensor):
    words_list = [index2words[idx.item()] for idx in tensor]
    return ' '.join(words_list)

_, topIdx = decoder_outputs.topk(1)
decoded_ids = topIdx.squeeze()

print(convert2sentence(decoded_ids))



