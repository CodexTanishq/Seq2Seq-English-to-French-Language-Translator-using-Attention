import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re

# ==========================================
# 1. MODEL ARCHITECTURE CLASSES
# ==========================================
# (These must be defined here so PyTorch knows how to construct the loaded weights)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, rnn_type='GRU', num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        else:
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
            
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, source_tokens):
        embedded = self.embedding(source_tokens) 
        encoder_outputs, hidden = self.rnn(embedded)
        
        if self.rnn_type == 'LSTM':
            h_n, c_n = hidden
            h_concat = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
            c_concat = torch.cat((c_n[-2,:,:], c_n[-1,:,:]), dim=1)
            decoder_hidden = torch.tanh(self.fc_hidden(h_concat)).unsqueeze(0).repeat(self.num_layers, 1, 1)
            decoder_cell = torch.tanh(self.fc_cell(c_concat)).unsqueeze(0).repeat(self.num_layers, 1, 1)
            return encoder_outputs, (decoder_hidden, decoder_cell)
        else:
            h_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            decoder_hidden = torch.tanh(self.fc_hidden(h_concat)).unsqueeze(0).repeat(self.num_layers, 1, 1)
            return encoder_outputs, decoder_hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[1]
        dec_h = decoder_hidden[-1].unsqueeze(1).repeat(1, seq_len, 1) 
        score = torch.tanh(self.W1(dec_h) + self.W2(encoder_outputs)) 
        attention_weights = F.softmax(self.V(score), dim=1) 
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1) 
        return context_vector, attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, attention, rnn_type='GRU', num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        rnn_input_dim = embedding_dim + (hidden_dim * 2)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(rnn_input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        else:
            self.rnn = nn.GRU(rnn_input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            
        self.fc_out = nn.Linear(hidden_dim + (hidden_dim * 2) + embedding_dim, vocab_size)

    def forward(self, target_token, decoder_hidden, encoder_outputs):
        target_token = target_token.unsqueeze(1) 
        embedded = self.embedding(target_token) 
        
        current_h = decoder_hidden[0] if self.rnn_type == 'LSTM' else decoder_hidden
        context, attention_weights = self.attention(current_h, encoder_outputs) 
        
        context_input = context.unsqueeze(1) 
        rnn_input = torch.cat((embedded, context_input), dim=2) 
        
        rnn_output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)
        prediction_input = torch.cat((rnn_output.squeeze(1), context, embedded.squeeze(1)), dim=1)
        predictions = self.fc_out(prediction_input) 
        return predictions, decoder_hidden, attention_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # Forward method is primarily for training, skipped during simple inference
        pass

# ==========================================
# 2. CACHED LOADING FUNCTIONS
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_vocabularies():
    with open("eng_w2i.json", "r") as f:
        eng_w2i = json.load(f)
    with open("fra_i2w.json", "r") as f:
        # JSON keys are always strings, we need to convert them back to integers for PyTorch
        fra_i2w = {int(k): v for k, v in json.load(f).items()}
    return eng_w2i, fra_i2w

@st.cache_resource
def load_model(_eng_w2i, _fra_i2w):
    with open("best_hyperparameters.json", "r") as f:
        best_config = json.load(f)

    encoder = Encoder(
        vocab_size=len(_eng_w2i), 
        embedding_dim=best_config['embedding_dim'], 
        hidden_dim=best_config['hidden_dim'], 
        rnn_type=best_config['rnn_type'], 
        num_layers=1, 
        dropout=best_config['dropout']
    )
    attention = BahdanauAttention(hidden_dim=best_config['hidden_dim'])
    decoder = Decoder(
        vocab_size=len(_fra_i2w), 
        embedding_dim=best_config['embedding_dim'], 
        hidden_dim=best_config['hidden_dim'], 
        attention=attention, 
        rnn_type=best_config['rnn_type'], 
        num_layers=1, 
        dropout=best_config['dropout']
    )

    model = Seq2Seq(encoder, decoder).to(device)
    model.load_state_dict(torch.load("best_translation_model.pt", map_location=device))
    model.eval()
    return model

# ==========================================
# 3. INFERENCE FUNCTIONS
# ==========================================
def clean_input_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def translate_sentence(model, source_tensor, max_len=50, sos_token=2, eos_token=3):
    with torch.no_grad():
        encoder_outputs, decoder_hidden = model.encoder(source_tensor.unsqueeze(0).to(device))
        translated_tokens = [sos_token]
        
        for _ in range(max_len):
            current_token = torch.tensor([translated_tokens[-1]], dtype=torch.long).to(device)
            output, decoder_hidden, _ = model.decoder(current_token, decoder_hidden, encoder_outputs)
            predicted_token = output.argmax(1).item()
            translated_tokens.append(predicted_token)
            if predicted_token == eos_token:
                break
    return translated_tokens

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Neural Translator Bot", page_icon="🤖")

st.title("🤖 Neural Machine Translator")
st.markdown("Custom Seq2Seq model with Bahdanau Attention (English to French)")

# Load everything 
eng_w2i, fra_i2w = load_vocabularies()
model = load_model(eng_w2i, fra_i2w)

# User Input
user_input = st.text_area("Enter English text:", placeholder="Type a short sentence here...")

if st.button("Translate", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        # Preprocess
        cleaned_text = clean_input_text(user_input)
        unk_id = eng_w2i.get('[UNK]', 1)
        tokens = [eng_w2i.get('[SOS]', 2)]
        tokens.extend([eng_w2i.get(word, unk_id) for word in cleaned_text.split()])
        tokens.append(eng_w2i.get('[EOS]', 3))
        
        # Translate
        source_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
        predicted_ids = translate_sentence(model, source_tensor)
        
        # Decode
        output_words = []
        for idx in predicted_ids:
            if idx in [0, 2]: continue
            if idx == 3: break
            output_words.append(fra_i2w.get(idx, '[UNK]'))
            
        final_translation = " ".join(output_words)
        
        st.success("Translation:")
        st.write(f"**{final_translation}**")