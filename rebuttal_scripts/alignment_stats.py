"""
Tokenizer-Patch Alignment Statistics
=====================================
Computes statistics for each dataset x tokenizer combination,
using tok_grapheme as the 100% aligned reference baseline.

Metrics:
  1. Mismatch rate        -- fraction where len(tok) != len(tok_grapheme)
  2. Length ratio         -- len(tok) / len(tok_grapheme) (mean, std, min, max)
  3. Over/under-seg rate  -- fraction where tok is longer/shorter than grapheme
  4. Mismatch by quartile -- mismatch rate binned by tok_grapheme length
  5. Precision            -- fraction of tok_X tokens (decoded) found in tok_grapheme (decoded)
  6. Recall               -- fraction of tok_grapheme tokens found in tok_X
  7. F1                   -- harmonic mean of precision and recall
  8. Avg token length     -- mean character length of decoded tokens
  9. Type-token ratio     -- unique tokens / total tokens per sample (mean)
  10. UNK rate            -- fraction of decoded tokens that are <unk> or equivalent

All token comparisons use decoded strings (multiset-aware, exact match).

Output structure:
  statistic_report/
    alignment_stats.csv
    mismatch_by_length.csv
    precision_recall_f1.csv
    token_characteristics.csv
    <dataset>/
      (same 4 files)
"""

import os
from collections import Counter
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

# =============================================================================
# CONFIGURATION — fill in paths before running
# =============================================================================
_BASE = "/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/datasets"

DATASETS = {
    "java":          f"{_BASE}/Exqrch___rebuttal-javanese-pixelgpt/default/0.0.0/c5fef00960c24c242eeb63befca570f18c1b3ec7",
    "bali_java-tok": f"{_BASE}/Exqrch___rebuttal-balinese-pixelgpt/default/0.0.0/4a31992e9daac65666bc3a814085bc8e48352210",
    "bali_bali-tok": f"{_BASE}/Exqrch___rebuttal-pure_bali-grapheme_experiment_only/default/0.0.0/97c9a64ef822783fe211834f590bc8e2c3a02b08",
    "sunda":         f"{_BASE}/Exqrch___rebuttal-sundanese-pixelgpt/default/0.0.0/dcd6ec05b35e3845faf930e710924114233c6f55",
    "lampung":       f"{_BASE}/Exqrch___rebuttal-lampung-pixelgpt/default/0.0.0/efac497f7298c8234298699767f8f46db161eb73",
}

# Nested: TOKENIZER_PATHS[dataset][tokenizer] = path or HF repo
# tok_grapheme differs for bali_bali-tok; all others share the same 3 tokenizers.
_TOK_LLAMA2  = "ernie-research/DualGPT"
_TOK_KOMODO  = "Yellow-AI-NLP/komodo-7b-base"
_TOK_MT5     = "google/mt5-small"
_TOK_GR_JAVA = "izzako/javanese-llama-tokenizer"   # e.g. izzako/javanese-llama-tokenizer
_TOK_GR_BALI = "izzako/balinese-llama-tokenizer"   # e.g. izzako/balinese-llama-tokenizer
_TOK_GR_SUND = "izzako/sunda-llama-tokenizer"  # e.g. izzako/sunda-llama-tokenizer

TOKENIZER_PATHS = {
    "java": {
        "tok_grapheme": _TOK_GR_JAVA,
        "tok_llama2":   _TOK_LLAMA2,
        "tok_komodo":   _TOK_KOMODO,
        "tok_mt5":      _TOK_MT5,
    },
    "bali_java-tok": {
        "tok_grapheme": _TOK_GR_JAVA,   # Java tokenizer used for Bali data
        "tok_llama2":   _TOK_LLAMA2,
        "tok_komodo":   _TOK_KOMODO,
        "tok_mt5":      _TOK_MT5,
    },
    "bali_bali-tok": {
        "tok_grapheme": _TOK_GR_BALI,   # Native Bali tokenizer
        "tok_llama2":   _TOK_LLAMA2,
        "tok_komodo":   _TOK_KOMODO,
        "tok_mt5":      _TOK_MT5,
    },
    "sunda": {
        "tok_grapheme": _TOK_GR_SUND,
        "tok_llama2":   _TOK_LLAMA2,
        "tok_komodo":   _TOK_KOMODO,
        "tok_mt5":      _TOK_MT5,
    },
    "lampung": {
        "tok_grapheme": _TOK_GR_SUND,   # Sunda tokenizer used for Lampung (no Lampung-specific)
        "tok_llama2":   _TOK_LLAMA2,
        "tok_komodo":   _TOK_KOMODO,
        "tok_mt5":      _TOK_MT5,
    },
}

TOKENIZERS = ["tok_grapheme", "tok_llama2", "tok_komodo", "tok_mt5"]
OUTPUT_DIR = "statistic_report"

# =============================================================================
# HELPERS
# =============================================================================

def load_tokenizers(ds_name):
    """Load all 4 tokenizers for a given dataset, caching by path to avoid reloads."""
    toks = {}
    paths_seen = {}
    for tok_col, path in TOKENIZER_PATHS[ds_name].items():
        if path in paths_seen:
            toks[tok_col] = paths_seen[path]
        else:
            print(f"  Loading tokenizer [{tok_col}]: {path}")
            t = AutoTokenizer.from_pretrained(path)
            toks[tok_col] = t
            paths_seen[path] = t
    return toks


def decode_ids(token_ids, tokenizer):
    """Decode a list of token IDs to a list of token strings (one string per token)."""
    return [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]


def multiset_precision_recall(ref_tokens, hyp_tokens):
    """
    Compute multiset precision and recall between two token lists.
      precision = |hyp ∩ ref| / |hyp|   (how much of hyp is in ref)
      recall    = |hyp ∩ ref| / |ref|   (how much of ref is covered by hyp)
    Returns (precision, recall, f1). All zero if either list is empty.
    """
    if not ref_tokens or not hyp_tokens:
        return 0.0, 0.0, 0.0
    ref_counter = Counter(ref_tokens)
    hyp_counter = Counter(hyp_tokens)
    # Multiset intersection: min count for each token
    intersection = sum((ref_counter & hyp_counter).values())
    precision = intersection / len(hyp_tokens)
    recall    = intersection / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def is_unk(token_str, tokenizer):
    """Check if a decoded token string represents an unknown token."""
    unk = getattr(tokenizer, "unk_token", None)
    return unk is not None and token_str.strip() == unk.strip()


def load_columns(path, columns):
    print(f"  Loading dataset: {path}")
    ds = load_dataset(path, split="train")
    return {col: ds[col] for col in columns if col in ds.column_names}


# =============================================================================
# CORE STATS
# =============================================================================

def compute_stats(data, tokenizers, dataset_name):
    """
    Args:
        data:       dict of col -> list of token_id lists
        tokenizers: dict of tok_col -> HF tokenizer object
        dataset_name: str

    Returns four DataFrames: summary, binned, prf, tokchar
    """
    grapheme_lens = np.array([len(x) for x in data["tok_grapheme"]], dtype=np.int32)
    n = len(grapheme_lens)
    print(f"  Samples: {n:,}")

    quartile_edges = np.percentile(grapheme_lens, [0, 25, 50, 75, 100])

    summary_rows  = []
    binned_data   = {}
    prf_rows      = []
    tokchar_rows  = []
    lengths       = {}

    for tok in TOKENIZERS:
        if tok not in data:
            print(f"  WARNING: column '{tok}' not found, skipping.")
            continue

        tokenizer = tokenizers[tok]
        tok_ids_list = data[tok]
        tok_lens = np.array([len(x) for x in tok_ids_list], dtype=np.int32)
        lengths[tok] = tok_lens

        # ── Stats 1–3 ──────────────────────────────────────────────────────
        if tok == "tok_grapheme":
            mismatch   = np.zeros(n, dtype=bool)
            ratio      = np.ones(n, dtype=np.float32)
            over_rate  = 0.0
            under_rate = 0.0
        else:
            mismatch   = tok_lens != grapheme_lens
            ratio      = tok_lens / grapheme_lens.clip(min=1)
            over_rate  = float((tok_lens > grapheme_lens).mean())
            under_rate = float((tok_lens < grapheme_lens).mean())

        binned_data[tok] = mismatch

        summary_rows.append({
            "dataset":        dataset_name,
            "tokenizer":      tok,
            "n":              n,
            "mismatch_rate":  round(float(mismatch.mean()), 4),
            "ratio_mean":     round(float(ratio.mean()), 4),
            "ratio_std":      round(float(ratio.std()), 4),
            "ratio_min":      round(float(ratio.min()), 4),
            "ratio_max":      round(float(ratio.max()), 4),
            "over_seg_rate":  round(over_rate, 4),
            "under_seg_rate": round(under_rate, 4),
        })

        # ── Stats 5–7: precision / recall / F1 (decoded multiset) ──────────
        # ── Stats 8–10: avg token length, TTR, UNK rate ────────────────────
        precisions   = []
        recalls      = []
        f1s          = []
        avg_tok_lens = []
        ttrs         = []
        unk_rates    = []

        grapheme_ids_list = data["tok_grapheme"]

        print(f"  Computing decoded metrics [{tok}]...")
        for i in range(n):
            decoded_tok      = decode_ids(tok_ids_list[i],      tokenizer)
            decoded_grapheme = decode_ids(grapheme_ids_list[i], tokenizers["tok_grapheme"])

            # Strip special tokens from reference and hypothesis for fair comparison
            ref_clean = [t for t in decoded_grapheme
                         if t not in (tokenizers["tok_grapheme"].all_special_tokens or [])]
            hyp_clean = [t for t in decoded_tok
                         if t not in (tokenizer.all_special_tokens or [])]

            p, r, f = multiset_precision_recall(ref_clean, hyp_clean)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

            # Avg token length (characters), TTR, UNK rate on hyp
            if hyp_clean:
                avg_tok_lens.append(np.mean([len(t) for t in hyp_clean]))
                ttrs.append(len(set(hyp_clean)) / len(hyp_clean))
                unk_rates.append(sum(is_unk(t, tokenizer) for t in hyp_clean) / len(hyp_clean))
            else:
                avg_tok_lens.append(0.0)
                ttrs.append(0.0)
                unk_rates.append(0.0)

        prf_rows.append({
            "dataset":    dataset_name,
            "tokenizer":  tok,
            "n":          n,
            "precision":  round(float(np.mean(precisions)), 4),
            "recall":     round(float(np.mean(recalls)), 4),
            "f1":         round(float(np.mean(f1s)), 4),
        })

        tokchar_rows.append({
            "dataset":         dataset_name,
            "tokenizer":       tok,
            "n":               n,
            "avg_token_len":   round(float(np.mean(avg_tok_lens)), 4),
            "type_token_ratio":round(float(np.mean(ttrs)), 4),
            "unk_rate":        round(float(np.mean(unk_rates)), 4),
        })

    # ── Stat 4: mismatch by tok_grapheme quartile ───────────────────────────
    binned_rows = []
    bin_labels  = ["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"]
    for q_idx in range(4):
        lo = quartile_edges[q_idx]
        hi = quartile_edges[q_idx + 1]
        mask = (grapheme_lens >= lo) & (grapheme_lens <= hi if q_idx == 3 else grapheme_lens < hi)

        bin_n        = mask.sum()
        length_range = f"{int(lo)}-{int(hi)}"

        for tok in TOKENIZERS:
            if tok not in binned_data:
                continue
            bin_mismatch = binned_data[tok][mask].mean() if bin_n > 0 else float("nan")
            binned_rows.append({
                "dataset":        dataset_name,
                "tokenizer":      tok,
                "quartile":       bin_labels[q_idx],
                "grapheme_range": length_range,
                "n_samples":      int(bin_n),
                "mismatch_rate":  round(float(bin_mismatch), 4) if bin_n > 0 else None,
            })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(binned_rows),
        pd.DataFrame(prf_rows),
        pd.DataFrame(tokchar_rows),
    )


# =============================================================================
# MAIN
# =============================================================================

all_summary  = []
all_binned   = []
all_prf      = []
all_tokchar  = []

for ds_name, ds_path in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*60}")

    data       = load_columns(ds_path, TOKENIZERS)
    tokenizers = load_tokenizers(ds_name)

    summary_df, binned_df, prf_df, tokchar_df = compute_stats(data, tokenizers, ds_name)
    all_summary.append(summary_df)
    all_binned.append(binned_df)
    all_prf.append(prf_df)
    all_tokchar.append(tokchar_df)

    ds_out = os.path.join(OUTPUT_DIR, ds_name)
    os.makedirs(ds_out, exist_ok=True)
    summary_df.to_csv(os.path.join(ds_out, "alignment_stats.csv"),      index=False)
    binned_df.to_csv(os.path.join(ds_out,  "mismatch_by_length.csv"),   index=False)
    prf_df.to_csv(os.path.join(ds_out,     "precision_recall_f1.csv"),  index=False)
    tokchar_df.to_csv(os.path.join(ds_out, "token_characteristics.csv"),index=False)
    print(f"  Saved to: {ds_out}/")

os.makedirs(OUTPUT_DIR, exist_ok=True)
summary_all  = pd.concat(all_summary,  ignore_index=True)
binned_all   = pd.concat(all_binned,   ignore_index=True)
prf_all      = pd.concat(all_prf,      ignore_index=True)
tokchar_all  = pd.concat(all_tokchar,  ignore_index=True)

summary_all.to_csv( os.path.join(OUTPUT_DIR, "alignment_stats.csv"),      index=False)
binned_all.to_csv(  os.path.join(OUTPUT_DIR, "mismatch_by_length.csv"),   index=False)
prf_all.to_csv(     os.path.join(OUTPUT_DIR, "precision_recall_f1.csv"),  index=False)
tokchar_all.to_csv( os.path.join(OUTPUT_DIR, "token_characteristics.csv"),index=False)

# =============================================================================
# PRETTY PRINT
# =============================================================================
SEP = "=" * 80

print(f"\n\n{SEP}")
print("STATS 1-3: MISMATCH RATE / RATIO / OVER-UNDER SEGMENTATION")
print(SEP)
print(summary_all.to_string(index=False))

print(f"\n\n{SEP}")
print("STAT 4: MISMATCH RATE BY GRAPHEME LENGTH QUARTILE")
print(SEP)
print(binned_all.to_string(index=False))

print(f"\n\n{SEP}")
print("STATS 5-7: PRECISION / RECALL / F1 (decoded multiset)")
print(SEP)
print(prf_all.to_string(index=False))

print(f"\n\n{SEP}")
print("STATS 8-10: AVG TOKEN LENGTH / TYPE-TOKEN RATIO / UNK RATE")
print(SEP)
print(tokchar_all.to_string(index=False))

print(f"\n\nSaved to: {OUTPUT_DIR}/")