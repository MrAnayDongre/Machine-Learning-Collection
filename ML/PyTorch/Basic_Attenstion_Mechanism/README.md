# Attention Mechanisms in Seq2Seq Models

This repository contains implementations of Seq2Seq models with Bahdanau and Luong attention mechanisms. The project demonstrates how to build encoder-decoder models with attention mechanisms in PyTorch.

## Table of Contents

- [Introduction](#introduction)
- [File Structure](#file-structure)
  - [logic.py](#logicpy)
  - [bahdanau_attention.py](#bahdanau_attentionpy)
  - [luong_attention.py](#luong_attentionpy)
  - [decoder.py](#decoderpy)


## Introduction

Attention mechanisms have become a crucial component in sequence-to-sequence (Seq2Seq) models, enhancing the ability of models to handle longer sequences and improve the accuracy of translations or predictions. This repository provides implementations of both Bahdanau and Luong attention mechanisms in a Seq2Seq framework.

## File Structure

### logic.py

This file contains the core logic for the Seq2Seq model. It includes the following functionalities:

- **Tokenization and Vocabulary Management**: Defines special tokens and builds a vocabulary from a given sentence.
- **Tensor Conversion**: Converts sentences to tensors for input into the model.
- **Encoder**: Implements the EncoderRNN class using GRU, which processes the input sequence and outputs hidden states.

### bahdanau_attention.py

This file implements the Bahdanau attention mechanism, also known as additive attention. It includes the `BahdanauAttention` class, which calculates attention scores and context vectors based on the hidden states of the encoder.

### luong_attention.py

This file implements the Luong attention mechanism, also known as multiplicative attention. It includes the `LuongAttention` class, which calculates attention scores and context vectors differently from Bahdanau attention.

### decoder.py

This file builds the decoder part of the Seq2Seq model. It includes the `AttnDecoderRNN` class, which incorporates the attention mechanisms to generate output sequences based on the encoder's hidden states and outputs.

