""" 
    VISUALIZING ATTENTION MECHANISM USING BERTVIZ LIBRARY

"""

from transformers import AutoTokenizer, AutoModel
from bertviz import head_view, model_view
from bertviz.neuron_view import show
from bertviz.transformers_neuron_view import BertModel, BertTokenizer

model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, output_attentions=True)

sentence = "The cat sat on the mat"
inputs = tokenizer.encode(sentence, return_tensors='pt')
#print(inputs)

outputs = model(inputs)
#print(outputs)

attention = outputs[-1]
tokens = tokenizer.convert_ids_to_tokens(inputs[0])


""" 

    Visualize the attentions for a specific input sentence
    
""" 
head_view(attention, tokens)


""" 
    Visualize attention layers in the models

"""
model_view(attention, tokens)

""" 

    Visualize specific keys and queries

"""
model_type = 'bert'
sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"
model = BertModel.from_pretrained(model_id, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=True)
show(
    model, 
    model_type, 
    tokenizer, 
    sentence_a, 
    sentence_b, 
    layer=2, 
    head=0
)