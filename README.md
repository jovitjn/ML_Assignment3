# Feed-forward neural language model

I built this fixed-context word predictor for ES 335: Machine Learning at IIT Gandhinagar. It is a small PyTorch model rather than an RNN or Transformer: the previous five words are embedded, concatenated, and passed through a two-layer MLP to predict the next word.

```text
5 token IDs -> 64-D embeddings -> flatten -> Linear(320, 128)
            -> ReLU -> Linear(128, vocabulary size)
```

The bundled checkpoint was trained on preprocessed Sherlock Holmes text with a vocabulary of 8,040 tokens. In the original ten-epoch notebook run, training loss fell from 6.9704 to 2.8452. I did not record a validation score for that checkpoint, so I treat this only as a training result.

## Try the included model

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python generate.py "sherlock holmes looked into" --num-words 12 --greedy
python generate.py "sherlock holmes looked into" \
  --num-words 12 --top-k 20 --temperature 0.8 --seed 42
```

The bundled checkpoint produces:

```text
# greedy
sherlock holmes looked into the air the gang the ceiling and was the key between the

# top-k, temperature 0.8, seed 42
sherlock holmes looked into the air post being silence bridge was the copper beeches child down
```

The rough phrasing is expected from a five-word feed-forward model trained on a small corpus. Greedy decoding is deterministic; top-k sampling is more varied and can be reproduced with `--seed`.

To use the Streamlit interface:

```bash
streamlit run app.py
```

## Train on another text file

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

The training script builds the vocabulary from the training split, keeps padding and unknown-word tokens separate, reports validation loss and perplexity, and saves the best validation checkpoint. I use a contiguous split so that validation windows are not duplicates of training windows.

## Files

- `model.py`: model definition, preprocessing, metadata checks, and decoding
- `train.py`: training and validation CLI
- `generate.py`: command-line generation
- `app.py`: Streamlit interface
- `metadata.json`: vocabulary and architecture settings for `model.pth`
- `Task1_training.ipynb`: original training and embedding-visualization notebook

`variables.pkl` is kept only with the original submission; none of the current scripts load it. The historical corpus is not in the repository, so the included checkpoint cannot be retrained exactly from these files alone.

The other two notebooks contain the remaining assignment work: regularized MLP comparisons in `ml_assignment3_part2.ipynb` and the original Task 3 submission in `ML3_Task3.ipynb`.