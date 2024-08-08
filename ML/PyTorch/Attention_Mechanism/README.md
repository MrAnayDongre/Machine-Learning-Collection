# Self-Attention and Multi-Head Attention Mechanisms

This repository contains implementations of self-attention and multi-head attention mechanisms in PyTorch. Additionally, it provides a visualization of attention mechanisms using the BERTViz library.

## Table of Contents

- [Introduction](#introduction)
- [File Structure](#file-structure)
  - [self_attention.py](#self_attentionpy)
  - [multihead_attention.py](#multihead_attentionpy)
  - [visualizing_attention.py](#visualizing_attentionpy)
- [Requirements](#requirements)
- [Usage](#usage)
- [Examples](#examples)
- [License](#license)

## Introduction

Attention mechanisms are a fundamental component of many modern deep learning models, particularly in natural language processing tasks. This repository provides basic implementations of self-attention and multi-head attention, along with a visualization of attention using the BERTViz library.

## File Structure

### self_attention.py

This file implements the self-attention mechanism. The `Attention` class in this script calculates the attention scores and outputs the weighted sum of the values (hidden states), given a set of input queries, keys, and values.

### multihead_attention.py

This file implements the multi-head attention mechanism, which runs multiple self-attention mechanisms in parallel. The `MultiheadAttention` class combines the outputs from multiple attention heads and processes them through a linear layer.

### visualizing_attention.py

This file demonstrates how to visualize attention mechanisms using the BERTViz library. It includes examples of how to visualize attention weights at various levels (head, model, and neuron view) using pre-trained BERT models.
