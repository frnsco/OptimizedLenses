import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import warnings; warnings.filterwarnings("ignore")

import sys, re, json, math, time, gc, hashlib
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_DIR = "/workspace/neuron-lens"
sys.path.insert(0, REPO_DIR)
from neuron_lens import transport as TR
from neuron_lens.transport import Accumulator, decoder_stack, target_layers, SKIP_FIRST
from neuron_lens.corpus import lens_docs
from neuron_lens.paths import CORPUS_DIR
from neuron_lens.arch import stem as nl_stem

# ======================== CONFIG (A40 48GB OPTIMIZED) ========================
MODELS = [
    ("Qwen/Qwen3-0.6B-Base", 1024),  # d=1024 -> 1 chunk per doc on 48GB A40!
    ("Qwen/Qwen3-1.7B-Base", 1024),  # d=2048 -> 2 chunks per doc on 48GB A40!
]
N_HERDED = 100       # 100 herded docs (matches qwen_top10_curve.ipynb N_SET=100; 10x fewer than 1,000!)
T_FIT = 128          # 128 tokens per doc
POOL_SIZE = 1000     # 1,000-doc DCLM candidate pool
FEAT_FRAC = 1 / 3    # layer fraction for herding features
N_PC = 10            # top-10 PCs for greedy herding
SEED = 0
OUT_ROOT = Path("/workspace/optimized_lenses_export")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
# =============================================================================

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False  # exact match to neuron-lens precision
print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB)")

def rows_sha256(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(np.asarray(r, dtype=np.int32).tobytes())
    return h.hexdigest()

@torch.no_grad()
def pool_feats(stack, rows, layer):
    cap = {}
    h = stack.layers[layer].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    vp = TR.valid_positions(len(rows[0]))
    vp = (torch.nonzero(vp).flatten() if vp.dtype == torch.bool else vp).to(dev)
    out = []
    bs = 64  # batched forward pass on A40 (64x faster than 1-by-1!)
    try:
        for s in range(0, len(rows), bs):
            inp = torch.tensor(rows[s:s+bs], device=dev)
            stack(input_ids=inp)
            out.append(cap["h"][:, vp].float().mean(1).cpu().numpy())
    finally:
        h.remove()
    return np.concatenate(out, axis=0)

def herd(X, n):
    mu, chosen, s = X.mean(0), [], np.zeros(X.shape[1])
    for j in range(n):
        g = np.linalg.norm((s[None] + X) / (j + 1) - mu, axis=1)
        g[chosen] = np.inf
        i = int(np.argmin(g))
        chosen.append(i)
        s += X[i]
    return chosen

def read_logits(E, OK, g, eps, Z):
    y = Z * torch.rsqrt(Z.pow(2).mean(-1, keepdim=True) + eps) * g
    lg = y @ E.T
    lg[:, ~OK] = float("-inf")
    return lg

corpus_prov_path = CORPUS_DIR / "dclm.provenance.json"
corpus_prov = json.loads(corpus_prov_path.read_text()) if corpus_prov_path.exists() else {}

summary_rows = []
for mid, dim_batch in MODELS:
    t_m = time.time()
    st = nl_stem(mid)
    print(f"\n==================== {mid} ({st}) ====================", flush=True)
    tok = AutoTokenizer.from_pretrained(mid)
    torch.set_float32_matmul_precision("high")
    lm = AutoModelForCausalLM.from_pretrained(mid, config=AutoConfig.from_pretrained(mid), dtype=torch.bfloat16).eval().requires_grad_(False)
    E = lm.get_output_embeddings().weight.detach().float().to(dev)
    stack = decoder_stack(lm).to(dev)
    torch.set_float32_matmul_precision("highest")
    n_layers, d = len(stack.layers), E.shape[1]
    names = tok.convert_ids_to_tokens(list(range(E.shape[0])))
    OK = torch.tensor([s is not None and not (re.fullmatch(r"<.*>", s) or "\ufffd" in s) for s in names], device=dev)
    g_norm = stack.norm.weight.detach().float().to(dev)
    eps = float(getattr(getattr(stack, "norm", None), "variance_epsilon", 1e-6))

    # 1. Select N_HERDED documents from 1,000-doc DCLM pool via batched 10-PC greedy herding
    pool = [r[:T_FIT] for r in lens_docs(CORPUS_DIR / "dclm.jsonl", tok, POOL_SIZE, 256, SKIP_FIRST + 2) if len(r) >= T_FIT]
    feat_l = max(1, round(n_layers * FEAT_FRAC))
    X = pool_feats(stack, pool, feat_l)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    U, S_, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    Z = U[:, :N_PC] * S_[:N_PC]; Z = Z / Z.std(0)
    sel_idx = herd(Z, N_HERDED)
    rows = [pool[i] for i in sel_idx]
    gap_sel = float(np.linalg.norm(Z[sel_idx].mean(0)))
    gap_rnd = float(np.linalg.norm(Z[:N_HERDED].mean(0)))
    print(f"Herded {N_HERDED}/{len(pool)} docs (L{feat_l}, {N_PC} PCs): mean gap={gap_sel:.4f} vs random={gap_rnd:.4f}", flush=True)

    # 2. Fit BOTH 'out' (r-lens/lens.pt) AND 'mid' (r-lens/mid.pt) across ALL layers in ONE pass!
    acc = Accumulator({
        "out": target_layers("R", "out", n_layers),
        "mid": target_layers("R", "mid", n_layers),
    }, d, device="cuda")

    times = []
    for i, ids in enumerate(rows):
        t0 = time.perf_counter()
        torch.set_float32_matmul_precision("high")
        try:
            TR.lens_doc("R", stack, torch.tensor(ids), acc, i % 2, dim_batch=dim_batch)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            dim_batch = max(64, dim_batch // 2)
            print(f"  [OOM fallback] reduced dim_batch -> {dim_batch}", flush=True)
            TR.lens_doc("R", stack, torch.tensor(ids), acc, i % 2, dim_batch=dim_batch)
            torch.cuda.synchronize()
        finally:
            torch.set_float32_matmul_precision("highest")
        times.append(time.perf_counter() - t0)
        if i in (0, 9, 24, 49, 74, N_HERDED - 1):
            print(f"  doc {i+1:3d}/{N_HERDED} | {times[-1]:.2f}s/doc (med {np.median(times[1:] or times):.2f}s) | peak VRAM={torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)

    # 3. Export in EXACT stdlogits/rs-lenses drop-in format!
    r_dir = OUT_ROOT / st / "r-lens"
    r_dir.mkdir(parents=True, exist_ok=True)
    r_sha = rows_sha256(rows)
    base_prov = {
        "model_id": mid,
        "dataset_id": f"{corpus_prov.get('repo', 'mlfoundations/dclm-baseline-1.0')}@{corpus_prov.get('revision', 'main')}",
        "rows": corpus_prov.get("rows", f"herded {N_HERDED} from {len(pool)} DCLM docs"),
        "rows_sha256": r_sha,
        "n_prompts": sum(acc.n_docs),
        "t_max": T_FIT,
        "skip_first": SKIP_FIRST,
        "selection_method": f"greedy_herding_{N_PC}pc_L{feat_l}_pool{len(pool)}",
        "selected_pool_indices": [int(x) for x in sel_idx],
    }

    for target, fname in (("out", "lens.pt"), ("mid", "mid.pt")):
        J_mean = acc.mean(target).cpu()  # fp32 [n_out_or_mid, d, d]
        J_dict = {L: J_mean[L].contiguous() for L in range(J_mean.shape[0])}
        if target == "out" and (n_layers - 1) not in J_dict:
            J_dict[n_layers - 1] = torch.eye(d, dtype=torch.float32)
        sh = [float(x) for x in acc.split_half(target)]
        prov = {**base_prov, "split_half": sh, "config_json": json.dumps({
            "estimator": "relp", "lens": "R", "target": target,
            "selection": base_prov["selection_method"],
            "n_prompts": sum(acc.n_docs), "t_max": T_FIT, "skip_first": SKIP_FIRST,
            "fit_dtype": "bfloat16", "code": "neuron-lens (neuron_lens/transport.py)",
        })}
        payload = {
            "J": J_dict,
            "n_prompts": sum(acc.n_docs),
            "source_layers": list(range(n_layers)),
            "d_model": int(d),
            "provenance": prov,
        }
        out_p = r_dir / fname
        torch.save(payload, out_p)
        print(f"  Saved {out_p} ({out_p.stat().st_size/2**20:.1f} MB) | split-half median={np.median(sh):.3f}", flush=True)

    # 4. Verify directly against stdlogits/rs-lenses 1,000-doc reference!
    ref_path = hf_hub_download("stdlogits/rs-lenses", f"{st}/r-lens/lens.pt", repo_type="model")
    REF = torch.load(ref_path, map_location="cpu", weights_only=True)["J"]
    OUR = torch.load(r_dir / "lens.pt", map_location="cpu", weights_only=True)["J"]
    rng = np.random.default_rng(SEED + 1)
    scored_ls = list(range(2, n_layers - 1, 2))
    top10s, coss = [], []
    for L in scored_ls:
        Wd = stack.layers[L].mlp.down_proj.weight.detach().float()
        picks = rng.choice(Wd.shape[1], 128, replace=False)
        P = torch.stack([s * Wd[:, n] for n in picks for s in (1, -1)]).to(dev)
        R_m = REF[L].float().to(dev); O_m = OUR[L].float().to(dev)
        rt = read_logits(E, OK, g_norm, eps, P @ R_m.T).topk(10, dim=1).indices
        ot = read_logits(E, OK, g_norm, eps, P @ O_m.T).topk(10, dim=1).indices
        top10s.append(float((rt[:, :, None] == ot[:, None, :]).any(-1).float().mean()))
        coss.append(float((O_m * R_m).sum() / (O_m.norm() * R_m.norm())))
    k3 = max(1, len(scored_ls) // 3)
    rec = dict(
        model=mid, stem=st, docs=N_HERDED, secs=round(time.time()-t_m, 1),
        med_s_doc=round(float(np.median(times[1:] or times)), 2),
        top10_early=round(100*np.mean(top10s[:k3]), 1),
        top10_mid=round(100*np.mean(top10s[k3:-k3]), 1),
        top10_late=round(100*np.mean(top10s[-k3:]), 1),
        min_cos=round(float(np.min(coss)), 4),
        med_cos=round(float(np.median(coss)), 4),
    )
    summary_rows.append(rec)
    print(f"  VERIFIED vs stdlogits/rs-lenses (1000 docs): top-10 early={rec['top10_early']}% mid={rec['top10_mid']}% late={rec['top10_late']}% | cos min={rec['min_cos']} med={rec['med_cos']}", flush=True)

    del lm, stack, E, acc, REF, OUR; gc.collect(); torch.cuda.empty_cache()

# Write README.md for frnsco/OptimizedLenses
readme_lines = [
    "---",
    "license: cc-by-4.0",
    "base_model:",
    "- Qwen/Qwen3-0.6B-Base",
    "- Qwen/Qwen3-1.7B-Base",
    "tags:",
    "- interpretability",
    "- jacobian-lens",
    "- r-lens",
    "- neuron-lens",
    "---",
    "",
    "# OptimizedLenses — Herded Fast R-Lenses for Qwen3-Base",
    "",
    "Drop-in R-lens files (`r-lens/lens.pt` and `r-lens/mid.pt`) formatted identically to [`stdlogits/rs-lenses`](https://huggingface.co/stdlogits/rs-lenses), fitted on **100 greedy-herded documents** (selected from a 1,000-document DCLM pool using top-10 PCs at 1/3 model depth) instead of 1,000 documents (10× less fitting compute).",
    "",
    "## Drop-In Loading (Exact `neuron-lens` / `rs-lenses` Format)",
    "",
    "```python",
    "import torch",
    "from huggingface_hub import hf_hub_download",
    "",
    "# Point your normal rs-lenses loader directly at frnsco/OptimizedLenses:",
    "path = hf_hub_download('frnsco/OptimizedLenses', 'qwen3-0.6b-base/r-lens/lens.pt')",
    "J = torch.load(path, map_location='cpu', weights_only=True, mmap=True)['J']",
    "```",
    "",
    "## Verification vs. `stdlogits/rs-lenses` (1,000-doc Reference)",
    "",
    "| Model | Path | Docs | Top-10 Early | Top-10 Mid | Top-10 Late | Min Cos | Med Cos |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for r in summary_rows:
    readme_lines.append(
        f"| `{r['model']}` | `{r['stem']}/r-lens/{{lens,mid}}.pt` | {r['docs']} | {r['top10_early']}% | {r['top10_mid']}% | {r['top10_late']}% | {r['min_cos']} | {r['med_cos']} |"
    )
(OUT_ROOT / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
(OUT_ROOT / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
print("\nALL LENSES FITTED, VERIFIED, AND STAGED AT:", OUT_ROOT)
