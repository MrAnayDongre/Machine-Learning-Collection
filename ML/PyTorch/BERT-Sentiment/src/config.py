import transformers

DEVICE = "cpu"
MAX_LEN= 512
TRAIN_BATCH_SIZE=8
VALID_BATCH_SIZE=4
EPOCHS=10
BERT_PATH="C:/Users/91960/Machine-Learning-Collection/ML/PyTorch/BERT-Sentiment/input/bert_base_uncased"
MODEL_PATH="model.bin"
TRAINING_FILE="../input/IMDB Dataset.csv"
TOKENIZER= transformers.BertTokenizer.from_pretrained(
    BERT_PATH, 
    do_lower_case=True
)