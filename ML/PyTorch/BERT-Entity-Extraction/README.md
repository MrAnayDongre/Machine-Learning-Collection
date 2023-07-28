# BERT-based Named Entity Recognition (NER) Model


This repository contains code for training a BERT-based Named Entity Recognition (NER) model on the Annotated Corpus for Named Entity Recognition dataset. The model is implemented in PyTorch and uses the Hugging Face Transformers library to work with BERT.

## [DATASET](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus)
The dataset used for training the model is provided in the input folder. It contains two CSV files:

1. `ner.csv`: Feature Engineered Corpus annotated with IOB and POS tags.
2. `ner_dataset`.csv: Another version of the dataset.


## REQUIREMENTS
To install the required packages, use the `requirements.txt` file:

```
pip install -r requirements.txt

```

## CONFIGURATION
The model's hyperparameters and other configurations are defined in the config.py file. It includes the following settings:

* **MAX_LEN:** Maximum sequence length for BERT input.
* **TRAIN_BATCH_SIZE:** Batch size for training.
* **VALID_BATCH_SIZE:** Batch size for validation.
* **EPOCHS:** Number of training epochs.
* **BASE_MODEL_PATH:** Path to the pre-trained BERT model.
* **MODEL_PATH:** Path to save the trained model.
* **TRAINING_FILE:** Path to the dataset for training.
* **TOKENIZER:** BERT tokenizer for tokenizing text.

## DATASET PROCESSING
The data processing is handled in the `dataset.py` file. The `EntityDataset` class processes the texts, POS tags, and IOB tags from the dataset and prepares them for training. It tokenizes the input using the BERT tokenizer and creates tensors for input IDs, masks, and token type IDs.

## MODEL ARCHITECTURE
The model architecture is defined in the `model.py` file. The `EntityModel` class implements the BERT-based NER model using PyTorch. It uses the pre-trained BERT model and adds dropout layers and linear layers for predicting POS tags and IOB tags. The model calculates the loss for both predictions and returns the combined loss for training.

## TRAINING AND EVALUATION
The training and evaluation loops are implemented in the `engine.py` file. The `train_fn` function is used for training the model, and the `eval_fn` function is used for evaluating the model on the validation set. The training loop performs backpropagation and optimization using the AdamW optimizer with a linear learning rate scheduler.

## TRAINING THE MODEL
To train the NER model, run the `train.py` script. It will process the dataset, split it into training and validation sets, train the model, and save the best-performing model based on validation loss.

```
python train.py

```


Make sure to use an appropriate virtual environment for your project to avoid conflicts with other packages in your system.

Feel free to use and modify the code according to your needs. Happy coding !!!




