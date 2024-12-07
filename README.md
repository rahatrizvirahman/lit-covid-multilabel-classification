
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

To set up and use this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rahatrizvirahman/lit-covid-multilabel-classification.git
   cd lit-covid-multilabel-classification
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   - On Linux/Mac:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Follow these steps to use the project:

1. **Prepare the Dataset**:
   - Ensure your dataset is formatted appropriately for multi-label classification. The dataset should follow the expected structure.

2. **Run the Notebooks**:
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open any of the following notebooks to train and evaluate models:
     - `bert-model.ipynb`
     - `bi-lstm-word2vec.ipynb`
     - `sbert-model.ipynb`

3. **Model Outputs**:
   - Results and outputs will be saved in the `output/` directory for further analysis.

## Project Members
[Rahat Rizvi Rahman](https://github.com/rahatrizvirahman)
[Nafeez Fahad](https://github.com/Nafeez-f)
[Paula A. Gearon](https://github.com/quoll)
[Florence Steve-Essi]()


## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**:
   - Click the "Fork" button on the top-right of this repository.

2. **Create a Branch**:
   ```bash
   git checkout -b your-branch-name
   ```

3. **Make Your Changes**:
   - Commit your changes with a clear and concise message:
     ```bash
     git commit -m "Add your message here"
     ```

4. **Push Changes**:
   ```bash
   git push origin your-branch-name
   ```

5. **Open a Pull Request**:
   - Go to your forked repository and click "New Pull Request."


