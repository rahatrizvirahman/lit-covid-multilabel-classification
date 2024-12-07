# LitCovid Multi-label Classification

This repository provides implementations of various models for multi-label classification of COVID-19-related literature. The goal is to categorize scientific articles into relevant topics to facilitate efficient information retrieval and analysis.

## Models Implemented

- **BERT-based Model**: Utilizes the Bidirectional Encoder Representations from Transformers (BERT) architecture for classification. [Notebook: `bert-model.ipynb`]
- **Bi-LSTM with Word2Vec Embeddings**: Combines Bidirectional Long Short-Term Memory networks with Word2Vec embeddings for sequential text classification. [Notebook: `bi-lstm-word2vec.ipynb`]
- **Sentence-BERT (SBERT) Model**: Employs Sentence-BERT for generating sentence embeddings tailored for classification tasks. [Notebook: `sbert-model.ipynb`]

## Repository Structure

- `.ipynb_checkpoints/`: Contains checkpoint files for Jupyter notebooks.
- `output/`: Directory designated for storing model outputs and results.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: This document.
- `requirements.txt`: Lists the Python dependencies required to run the notebooks.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rahatrizvirahman/lit-covid-multilabel-classification.git
   cd lit-covid-multilabel-classification
