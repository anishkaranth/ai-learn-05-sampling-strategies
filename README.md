# AI Learn 05 — Sampling Strategies

Implement **temperature**, **top-k**, and **top-p (nucleus)** sampling from scratch. Compare entropy / diversity on a toy logits distribution and a tiny bigram LM.

## Learning goals

- Softmax temperature sharpens or flattens distributions
- Top-k truncates to a fixed candidate set
- Nucleus (top-p) adapts the set to cumulative probability
- Entropy & unique-token diversity as diagnostics

## Layout

```
sampling.py
run_smoke.py
notebooks/sampling_strategies.ipynb
results/
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_smoke.py
```
