# Smoke results — ai-learn-05-sampling-strategies

**Seed:** `42` · toy logits vocab=50, samples=2000

## Headline metrics

| Strategy | Entropy | Unique | Top-1 share |
|----------|--------:|-------:|------------:|
| temp_0.5 | 1.270 | 36 | 0.688 |
| temp_1.0 | 2.991 | 50 | 0.274 |
| temp_1.5 | 3.518 | 50 | 0.136 |
| topk_5 | 1.384 | 5 | 0.472 |
| topk_20 | 2.426 | 20 | 0.312 |
| topp_0.9 | 2.670 | 29 | 0.294 |
| topp_0.95 | 2.818 | 38 | 0.284 |

Base softmax entropy (T=1): **3.031**

## Takeaway

Lower temperature / small top-k concentrates mass. Nucleus (top-p) adapts the candidate set — a good default for open-ended generation.
