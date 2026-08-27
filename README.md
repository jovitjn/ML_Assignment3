# Machine Learning Experiments: Language Modeling and Regularization

This repository contains coursework experiments that are worth preserving because they illustrate several core machine-learning ideas with working implementations rather than only library calls.

The strongest component is a small **feed-forward neural language model** with learned word embeddings and a Streamlit demo for next-word generation. A second notebook studies regularization and nonlinear decision boundaries on XOR-style data.

## 1. Next-word prediction

`Task1_training.ipynb` trains a PyTorch model that predicts the next word from a fixed-length context window.

### Model

The architecture is intentionally simple:

```text
5-word context
    ↓
word embeddings (64-D)
    ↓
flatten
    ↓
Linear → ReLU (128 hidden units)
    ↓
Linear → vocabulary logits
    ↓
next-word prediction
```

The original experiment used a vocabulary of roughly 8k words and a 5-word context window. It was trained with cross-entropy loss and Adam.

The notebook also visualizes learned embeddings with t-SNE to inspect whether semantically related words occupy nearby regions of the embedding space.

### Interactive demo

`app.py` provides a Streamlit frontend that loads the trained checkpoint and repeatedly predicts the next word.

Run it with:

```bash
pip install torch streamlit
streamlit run app.py
```

The current app expects the included `model.pth` and `variables.pkl` files.

## 2. Regularization on nonlinear classification

`ml_assignment3_part2.ipynb` compares:

- an MLP without regularization
- L1-regularized MLP training
- L2-regularized MLP training
- polynomial logistic regression

on a synthetic XOR-style problem. The point of this notebook is to visualize how regularization changes a nonlinear decision boundary rather than to present a production classifier.

## 3. Additional coursework

`ML3_Task3.ipynb` is retained as part of the original assignment submission. It is not the main project surface of this repository.

## Repository status

This repository is intentionally labeled as coursework/educational work. The language-model component is the part worth developing further. A stronger future version would:

- move the model into a reusable Python module
- add train/validation splits and perplexity
- support top-k / temperature sampling instead of greedy decoding only
- replace serialized vocabulary state with a safer text/JSON format
- add a reproducible corpus-download script
- include screenshots of the Streamlit interface

The current repository preserves the original implementation while making clear which parts are technically substantive and which parts are classroom exercises.
