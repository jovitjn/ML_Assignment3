# Feed-Forward Neural Language Model

A fixed-context next-word predictor implemented in PyTorch, with reusable training and inference code, safe JSON vocabulary metadata, and a Streamlit demo.

This began as coursework for ES 335: Machine Learning at IIT Gandhinagar. It is presented as an educational language-model implementation—not as an RNN, Transformer, or research contribution.

## Model

Given the five most recent words, the model embeds each token and predicts a distribution over the vocabulary:

```text
5 token IDs → 64-D embeddings → flatten → Linear(320, 128)
            → ReLU → Linear(128, vocabulary size) → logits
```

For context tokens \(x_1, \ldots, x_5\), the forward pass is

\[
p(x_6 \mid x_1, \ldots, x_5)
= \operatorname{softmax}\!\left(W_2\,\operatorname{ReLU}
\left(W_1 [E(x_1);\ldots;E(x_5)] + b_1\right)+b_2\right).
\]

The included checkpoint was trained on preprocessed Sherlock Holmes text with an 8,040-token vocabulary, Adam, and cross-entropy loss. The recorded training loss in the original notebook decreased from **6.9704 to 2.8452** over 10 epochs. That run did not record a validation metric, so the training loss should not be read as a generalization result.

## What is reusable now

- `model.py` — model, JSON metadata validation, preprocessing, and greedy/temperature/top-k decoding
- `train.py` — configurable training with a contiguous train/validation split and validation perplexity
- `generate.py` — command-line inference
- `app.py` — Streamlit interface for the included checkpoint
- `metadata.json` — non-executable vocabulary and architecture metadata
- `model.pth` — state-dict-only checkpoint from the original experiment
- `Task1_training.ipynb` — original experiment and embedding visualization

The original `variables.pkl` is retained for history, but the application and command-line tools do **not** load it. Only use checkpoints and metadata from sources you trust.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inference

Run the included model from the terminal:

```bash
python generate.py "sherlock holmes looked into" --num-words 12 --greedy
python generate.py "sherlock holmes looked into" --num-words 12 --top-k 20 --temperature 0.8
```

Or launch the interactive demo:

```bash
streamlit run app.py
```

Greedy decoding is deterministic. Top-k decoding samples only from the highest-scoring \(k\) words; temperature controls how concentrated that distribution is.

## Train on another corpus

```bash
python train.py path/to/corpus.txt \
  --context-length 5 \
  --embedding-dim 64 \
  --hidden-dim 128 \
  --epochs 10 \
  --output-dir artifacts

python generate.py "your starting phrase" \
  --checkpoint artifacts/model.pth \
  --metadata artifacts/metadata.json
```

The script builds its vocabulary from the training portion only, reserves distinct padding and unknown-word tokens, reports held-out loss and perplexity each epoch, and saves the best validation checkpoint. The split is contiguous so validation windows do not duplicate training windows.

## Reproducibility and limitations

- Seeds are applied to Python and PyTorch; exact floating-point results can still vary across hardware and PyTorch versions.
- The source text used for the included checkpoint is not stored in this repository, so that historical run cannot be reproduced exactly from the repository alone.
- A five-word feed-forward model has no recurrent state or self-attention. It cannot represent dependencies outside its fixed context window.
- The original text preprocessing removes punctuation and words shorter than three characters, which limits fluency and discards useful syntax.
- The included checkpoint is small and intended to demonstrate the mechanics of embeddings, context windows, training, and decoding—not modern language-model quality.

## Other retained coursework

- `ml_assignment3_part2.ipynb` compares unregularized, L1-regularized, and L2-regularized MLPs with polynomial logistic regression on a synthetic nonlinear classification problem.
- `ML3_Task3.ipynb` is retained as part of the original assignment submission and is not the main project surface.
