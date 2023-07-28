import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 8
BASE_MODEL_PATH = "C:/Users/91960/Machine-Learning-Collection/ML/PyTorch/BERT-Entity-Extraction/input/bert-uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "C:/Users/91960/Machine-Learning-Collection/ML/PyTorch/BERT-Entity-Extraction/input/ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True,
)
