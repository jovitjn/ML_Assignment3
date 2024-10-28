import streamlit as st
import torch
import pickle
from torch import nn
import torch.nn.functional as F

# Load the trained model weights
class NextWordPredictor(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size, activation):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.activation = activation
        self.lin2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.size(0), -1)
        return self.lin2(self.activation(self.lin1(x)))

# Load variables
with open('variables.pkl', 'rb') as f:
    variables = pickle.load(f)

# Load the model
vocab_size = variables['vocab_size']
block_size = variables['block_size']
emb_dim = 64
hidden_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NextWordPredictor(block_size, vocab_size, emb_dim, hidden_size, activation=F.relu).to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Define the prediction function
def predict_next_word(model, input_words):
    with torch.no_grad():
        input_indices = [variables['stoi'].get(word, 0) for word in input_words.split()]
        if len(input_indices) > block_size:
            input_indices = input_indices[-block_size:]
        elif len(input_indices) < block_size:
            input_indices = [0] * (block_size - len(input_indices)) + input_indices
        
        input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
        outputs = model(input_tensor)
        predicted_index = torch.argmax(outputs, dim=1).item()
        predicted_word = variables['itos'][predicted_index]
    return predicted_word

# Streamlit app layout
st.title("Next Word Predictor")
input_text = st.text_input("Enter text:")
max_words = st.number_input("Number of words to generate:", min_value=1, max_value=20, value=5)

if st.button("Generate"):
    generated_text = input_text
    for _ in range(max_words):
        predicted_word = predict_next_word(model, ' '.join(generated_text.split()[-block_size:]))
        generated_text += ' ' + predicted_word
    st.write(f"Generated text: {generated_text}")
