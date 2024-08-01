# RNN Architecture

This repository contains the implementation of a simple Recurrent Neural Network (RNN) architecture using PyTorch. It is structured into several Python scripts, each serving a specific purpose within the overall project. The main components include data preprocessing, encoder, decoder, and a toy example to demonstrate the usage of PyTorch functionalities like linear layers and softmax functions.

## Contents

The repository is organized into the following files:

- `main.py`: Contains the initial setup, including token definitions, word indexing, and functions for converting sentences into PyTorch tensors.
- `encoder.py`: Implements the Encoder part of the RNN architecture. It uses GRU layers to process input sequences.
- `decoder.py`: Implements the Decoder part of the RNN architecture. It generates output sequences based on the encoder's hidden states.
- `toy_example.py`: Provides a simple example demonstrating the usage of linear layers and softmax functions in PyTorch, unrelated to the main RNN architecture but useful for understanding basic PyTorch operations.

## Code Overview

### main.py

This script sets up the basic environment for the RNN model. It defines special tokens (`SOS_token` and `EOS_token`) used to indicate the start and end of sentences. It also creates mappings between words and indices, which are essential for processing text data in neural networks. Additionally, it includes a function `convert2tensors` that converts sentences into PyTorch tensors suitable for input into the RNN model.

### encoder.py

The EncoderRNN class defined here takes input sequences (sentences converted into tensors) and processes them through an embedding layer followed by a GRU layer. The output of this encoder serves as a condensed representation of the input sequence, capturing its essential features for the decoder to generate meaningful output sequences.

### decoder.py

The DecoderRNN class generates output sequences based on the encoder's hidden states. It starts with a special start-of-sentence token and iteratively generates each word of the output sequence until it reaches a maximum length or produces an end-of-sentence token. The decoder uses a GRU layer and a linear layer to produce outputs, which are then passed through a softmax function to obtain probabilities over the vocabulary.

### toy_example.py

This script demonstrates basic PyTorch functionalities such as creating random tensors, applying linear transformations, and using the log softmax function. It serves as a simple example to understand how these operations work in PyTorch, independent of the RNN architecture.

## Usage

To run the code, ensure you have PyTorch installed in your environment. You can execute each script individually to see its output. For example, running `python main.py` will execute the initial setup and demonstrate how sentences are converted into tensors. Similarly, executing `python encoder.py` will show the output sizes of the encoder's operations, and so on for the other scripts.
