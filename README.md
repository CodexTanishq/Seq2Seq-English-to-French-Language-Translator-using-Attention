# Neural Machine Translator: English to French

A Sequence-to-Sequence (Seq2Seq) neural network built with PyTorch to translate English text into French. The model uses Bahdanau Additive Attention and includes a Streamlit web interface.

## Project Overview
This project builds the architecture behind neural machine translation from scratch using PyTorch. The model processes English sentences into context vectors using an encoder-decoder setup with attention and decodes them autoregressively into French.

## Data and Preprocessing
The dataset used is the English-French parallel corpus from Tatoeba (manythings.org).

- Dataset Size: 240,521 sentence pairs
- Text Normalization: Converted text to lowercase, removed special characters and punctuation using regex, and normalized whitespace.
- Tokenization: Utilized Byte-Pair Encoding (BPE) subword concepts.
- Vocabulary: Extracted the top 10,000 most frequent words from the cleaned data. Added four special control tokens ([PAD], [UNK], [SOS], [EOS]) to create a final vocabulary size of 10,004 tokens for both English and French.

## Model Architecture
- Encoder: Bidirectional RNN (GRU/LSTM) that processes the source sequence and generates hidden states.
- Bahdanau Attention: Additive attention mechanism that scores the relevance of source words for every decoded target word.
- Decoder: Unidirectional RNN that combines target token embeddings with the calculated context vectors to generate next-token probabilities.
- Optimization: Hyperparameters (embedding dimensions, hidden units, dropout, learning rate) were tuned using Optuna.

## Web Interface
A Streamlit web application is included for local inference:
- Caches model weights and vocabulary mappings in memory.
- Provides a simple text input box that applies the preprocessing pipeline and returns the translated French text.

## How to Run

1. Clone the repository:
git clone https://github.com/CodexTanishq/Seq2Seq-English-to-French-Language-Translator-using-Attention.git
cd Seq2Seq-English-to-French-Language-Translator-using-Attention

2. Install dependencies:
pip install torch pandas streamlit optuna nltk

3. Launch the app:
streamlit run app.py
