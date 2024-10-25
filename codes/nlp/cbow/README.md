# Continuous Bag of Words Model

This directory contains a simple implementation of the CBOW model for natural language processing tasks.

## Overview

The CBOW model is used generate words given the surrounding words. It is a type of word embedding model that helps in understanding the semantic relationships between words.

### Core Logic

https://www.notion.so/drplatelets/Word2Vec-1284be4e49d98049a694e6ce63ab17d8?pvs=4#1294be4e49d980469e18e8c21770a777

## Directory Structure

```
cbow/
│
├── dataset/
│   ├── train.txt      # The training dataset for the CBOW model
│
├── src/          
│   ├── main.cpp       # Main implementation file for the CBOW model
│   └── utils.cpp      # Utility functions used in the CBOW model
│
├── Test.ipynb         # .ipynb notebook to test the trained CBOW model
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
