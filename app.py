from pathlib import Path
import pickle

import streamlit as st
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pth"
VARIABLES_PATH = ROOT / "variables.pkl"


class NextWordPredictor(nn.Module):
    """Feed-forward next-word model used in the original assignment."""

    def __init__(self, block_size, vocab_size, emb_dim=64, hidden_size=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.reshape(x.size(0), -1)
        return self.lin2(F.relu(self.lin1(x)))


@st.cache_resource
def load_artifacts():
    """Load the vocabulary metadata and trained model once per app session."""
    with VARIABLES_PATH.open("rb") as f:
        variables = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NextWordPredictor(
        block_size=variables["block_size"],
        vocab_size=variables["vocab_size"],
    ).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, variables, device


def predict_next_word(model, variables, device, input_words):
    """Greedily predict one word from the most recent context window."""
    block_size = variables["block_size"]
    words = input_words.lower().split()
    input_indices = [variables["stoi"].get(word, 0) for word in words]

    if len(input_indices) > block_size:
        input_indices = input_indices[-block_size:]
    elif len(input_indices) < block_size:
        input_indices = [0] * (block_size - len(input_indices)) + input_indices

    input_tensor = torch.tensor(
        input_indices, dtype=torch.long, device=device
    ).unsqueeze(0)

    with torch.inference_mode():
        logits = model(input_tensor)
        predicted_index = torch.argmax(logits, dim=1).item()

    return variables["itos"][predicted_index]


st.set_page_config(page_title="Next Word Predictor", page_icon="📝")
st.title("Next Word Predictor")
st.caption("A small feed-forward neural language model trained on a fixed 5-word context.")

try:
    model, variables, device = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts are missing. Expected model.pth and variables.pkl beside app.py.")
    st.stop()

input_text = st.text_input("Enter a starting phrase")
max_words = st.number_input(
    "Number of words to generate", min_value=1, max_value=20, value=5, step=1
)

if st.button("Generate", type="primary"):
    if not input_text.strip():
        st.warning("Enter at least one word to start generation.")
    else:
        generated_words = input_text.strip().split()
        for _ in range(int(max_words)):
            context = " ".join(generated_words[-variables["block_size"] :])
            generated_words.append(
                predict_next_word(model, variables, device, context)
            )

        st.subheader("Generated text")
        st.write(" ".join(generated_words))
