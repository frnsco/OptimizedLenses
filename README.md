---
license: cc-by-4.0
base_model:
- Qwen/Qwen3-0.6B-Base
- Qwen/Qwen3-1.7B-Base
tags:
- interpretability
- jacobian-lens
- r-lens
- neuron-lens
---

# OptimizedLenses — Herded Fast R-Lenses for Qwen3-Base

Drop-in R-lens files (`r-lens/lens.pt` and `r-lens/mid.pt`) formatted identically to [`stdlogits/rs-lenses`](https://huggingface.co/stdlogits/rs-lenses), fitted on **100 greedy-herded documents** (selected from a 1,000-document DCLM pool using top-10 PCs at 1/3 model depth) instead of 1,000 documents (10× less fitting compute).

## Drop-In Loading (Exact `neuron-lens` / `rs-lenses` Format)

```python
import torch
from huggingface_hub import hf_hub_download

# Point your normal rs-lenses loader directly at epicfz/OptimizedLenses:
path = hf_hub_download('epicfz/OptimizedLenses', 'qwen3-0.6b-base/r-lens/lens.pt')
J = torch.load(path, map_location='cpu', weights_only=True, mmap=True)['J']
```

## Verification vs. `stdlogits/rs-lenses` (1,000-doc Reference)

| Model | Path | Docs | Top-10 Early | Top-10 Mid | Top-10 Late | Min Cos | Med Cos |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-0.6B-Base` | `qwen3-0.6b-base/r-lens/{lens,mid}.pt` | 100 | 87.0% | 91.5% | 97.6% | 0.9959 | 0.9987 |
| `Qwen/Qwen3-1.7B-Base` | `qwen3-1.7b-base/r-lens/{lens,mid}.pt` | 100 | 87.8% | 92.4% | 97.9% | 0.9956 | 0.9989 |
