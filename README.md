# Transformer Encoder for AG News Classification

Built a Transformer-based text classifier from scratch in PyTorch and trained it on the AG News dataset (4-class news categorization).

## Results

| Metric        | Value  |
|---------------|--------|
| Test Accuracy | 92.45% |
| Macro F1      | 0.924  |
| Precision     | 0.924  |
| Recall        | 0.924  |

Per-class F1 on 7,600 test articles: World 0.93 · Sports 0.97 · Business 0.89 · Tech 0.90

## Dataset

- **AG News** — 120,000 training articles, 7,600 test articles
- 4 classes: World, Sports, Business, Technology

## Approach

- Text preprocessing: cleaning, lowercasing, stopword removal, NLTK tokenization
- Built vocabulary from training tokens with padding and sequence truncation
- Implemented core Transformer components:
  - Sinusoidal positional encoding
  - Multi-head self-attention (via `nn.TransformerEncoder`)
  - Padding-mask handling
  - Mean-pooled classification head
- Optimization experiments: dropout, L2 weight decay, learning rate, StepLR scheduler
- Training: 8 epochs, best-F1 checkpointing, cross-entropy loss

## Stack

PyTorch · HuggingFace Datasets · NLTK · scikit-learn · Matplotlib

## Files

- `transformer_ag_news.ipynb` — end-to-end notebook (EDA, preprocessing, model, training, evaluation, ROC curves)

## How to Run

The AG News dataset loads automatically via HuggingFace Datasets:
```python
from datasets import load_dataset
dataset = load_dataset("ag_news")
```
Then open the notebook and run all cells.
