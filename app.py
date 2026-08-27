from pathlib import Path

import streamlit as st
import torch

from model import LanguageModelMetadata, generate_tokens, load_model, normalize_tokens

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pth"
METADATA_PATH = ROOT / "metadata.json"


@st.cache_resource
def load_artifacts():
    """Load the vocabulary metadata and trained model once per app session."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = LanguageModelMetadata.load(METADATA_PATH)
    model = load_model(MODEL_PATH, metadata, device)
    return model, metadata, device


st.set_page_config(page_title="Next Word Predictor", page_icon="📝")
st.title("Next Word Predictor")
st.caption("A small feed-forward neural language model trained on a fixed 5-word context.")

try:
    model, metadata, device = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts are missing. Expected model.pth and metadata.json beside app.py.")
    st.stop()
except (ValueError, RuntimeError) as error:
    st.error(f"Could not load model artifacts: {error}")
    st.stop()

input_text = st.text_input("Enter a starting phrase")
max_words = st.number_input(
    "Number of words to generate", min_value=1, max_value=20, value=5, step=1
)
decoding = st.selectbox("Decoding", ("Greedy", "Top-k sampling"))
temperature = st.slider(
    "Temperature", min_value=0.2, max_value=2.0, value=1.0, step=0.1,
    disabled=decoding == "Greedy",
)
top_k = st.slider(
    "Top-k", min_value=2, max_value=100, value=20, step=1,
    disabled=decoding == "Greedy",
)

if st.button("Generate", type="primary"):
    if not input_text.strip():
        st.warning("Enter at least one word to start generation.")
    else:
        prompt_tokens = normalize_tokens(input_text)
        if not prompt_tokens:
            st.warning("Enter at least one alphanumeric word of length 3 or more.")
        else:
            generated_words = generate_tokens(
                model,
                metadata,
                prompt_tokens,
                int(max_words),
                greedy=decoding == "Greedy",
                temperature=float(temperature),
                top_k=int(top_k) if decoding != "Greedy" else None,
                device=device,
            )
            st.subheader("Generated text")
            st.write(" ".join(generated_words))
