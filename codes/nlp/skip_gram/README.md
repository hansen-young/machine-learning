# Skip-Gram Model

This directory contains a simple implementation of the Skip-Gram model for natural language processing tasks.

## Overview

The Skip-Gram model is used to predict the context words given a target word. It is a type of word embedding model that helps in understanding the semantic relationships between words.

### Core Logic

https://www.notion.so/drplatelets/Word2Vec-Skip-Gram-1284be4e49d98049a694e6ce63ab17d8

## Directory Structure

```
skip_gram/
│
├── dataset/
│   ├── train.txt      # The training dataset for the Skip-Gram model
│
├── src/          
│   ├── main.cpp       # Main implementation file for the Skip-Gram model
│   └── utils.cpp      # Utility functions used in the Skip-Gram model
│
├── Test.ipynb         # .ipynb notebook to test the trained Skip-Gram model
└── Makefile           # Makefile for building the project
```

## Getting Started

### Prerequisites

- C++17
- Python 3.x (for `Test.ipynb`)

### Running the Code

```sh
make run
```
