
## Environment Setup

Follow these steps to set up your environment:

1. **Create a virtual environment** (if you haven't already):

    ```sh
    python3 -m venv venv
    ```

2. **Activate the environment**:

    ```sh
    source venv/bin/activate
    ```

3. **Install the required packages**:

    ```sh
    pip3 install -r requirements.txt
    ```

4. **Deactivate the environment** when done:

    ```sh
    deactivate
    ```

## Run Codes
To run Sbert
```
python3 sbert-model.py
```

To run Bert
```
python3 bert-model.py
```

To run Bidirectional LSTM
```
python3 bi-lstm-word2vec.py
```

The model outputs are saved in output folder