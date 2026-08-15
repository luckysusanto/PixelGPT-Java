"""
Tokenizer statistics over a HuggingFace dataset's `train` split.

Section 1 (Compression statistics, per tokenizer):
    - vocab_size_total : len(tokenizer.get_vocab()), i.e. the number of
                         distinct token ids the tokenizer can emit (base
                         vocab + added/special tokens). This is the
                         "effective" vocab size that matters for the model.
    - vocab_size_base  : tokenizer.vocab_size, the base vocabulary learned
                         during tokenizer training (e.g. 32000 for Llama-2).
    - added_tokens     : vocab_size_total - vocab_size_base.
    - oov_rate         : fraction of emitted tokens that equal the tokenizer's
                         unk_token_id (count of unk tokens / total tokens).
    - fertility        : total_tokens / total_words, where words are obtained by
                         splitting the raw `text` column on whitespace
                         (Latin-script corpus, whitespace is adequate).

Section 2 (Distribution statistics, per tokenizer):
    Following Lotz et al. (2025), "Beyond Text Compression: Evaluating
    Tokenizers Across Scales" (Sec. 3.4):
    - cardinality      : number of unique token ids observed in the corpus.
    - rank_freq_auc    : area under the log(rank) -> log(frequency) curve,
                         computed using Simpson's rule.
    - slope            : slope beta_1 of the linear fit
                         f(x) = beta_0 + beta_1 * x in log-log space,
                         approximating Zipf's law.
    - power_law_dev    : mean absolute error of the linear fit, i.e.
                         (1/n) * sum_i |beta_0 + beta_1 * x_i - y_i|.
    Per footnote 7 of the paper, AUC / slope / power_law_dev are estimated
    restricted to tokens with log(rank) <= 6 (i.e. rank <= ~403), because power
    laws only apply above some minimum. `cardinality` is computed on the full
    distribution (no restriction).

The tokenizer columns processed are: tok_grapheme, tok_llama2, tok_komodo, tok_mt5.
Each row in the dataset stores these as lists of token ids (already encoded by
your data_renderer.py with add_eos_token=True). We do not re-tokenize; we
consume the stored ids directly. This requires the corresponding tokenizers to
be loadable (for vocab_size and unk_token_id lookups).

USAGE
-----
python tokenizer_stats.py \
    --dataset_name Exqrch/Rebuttal-javanese-pixelgpt \
    --cache_dir /path/to/hf_cache_updated \
    --train_split train \
    --tok_grapheme_path izzako/javanese-llama-tokenizer \
    --tok_llama2_path  ernie-research/DualGPT \
    --tok_komodo_path  Yellow-AI-NLP/komodo-7b-base \
    --tok_mt5_path     google/mt5-small \
    --output_path /path/to/save/stats.json \
    --lang_label javanese
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from datasets import load_dataset, Dataset
from scipy.integrate import simpson
from transformers import AutoTokenizer, LlamaTokenizerFast

# Restriction from Lotz et al. (2025), footnote 7.
LOG_RANK_CUTOFF = 6.0


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------
def load_tokenizers(args):
    """Load the four tokenizers the same way data_renderer.py does."""
    print("Loading tokenizers...")
    tokenizers = {}
    # grapheme: language-specific Llama BPE (fast)
    tokenizers["tok_grapheme"] = LlamaTokenizerFast.from_pretrained(
        args.tok_grapheme_path, add_eos_token=True
    )
    # llama2: DualGPT Llama BPE (fast)
    tokenizers["tok_llama2"] = LlamaTokenizerFast.from_pretrained(
        args.tok_llama2_path, add_eos_token=True
    )
    # komodo: SEA-optimized BPE
    tokenizers["tok_komodo"] = AutoTokenizer.from_pretrained(
        args.tok_komodo_path, add_eos_token=True
    )
    # mt5: multilingual unigram
    tokenizers["tok_mt5"] = AutoTokenizer.from_pretrained(
        args.tok_mt5_path, add_eos_token=True
    )
    return tokenizers


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_train_split(dataset_name: str, cache_dir: str = None,
                     split: str = "train") -> Dataset:
    """Load the train split of a HuggingFace dataset by repo name.

    Uses datasets.load_dataset(name, split=split, cache_dir=cache_dir).
    The repo name is e.g. "Exqrch/Rebuttal-javanese-pixelgpt"; load_dataset
    will reuse the local cache under cache_dir if it's already there, and
    download only if missing.
    """
    print(f"Loading dataset: {dataset_name} (split={split})")
    if cache_dir:
        print(f"  cache_dir: {cache_dir}")
    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    return ds


# ---------------------------------------------------------------------------
# Streaming token counter
# ---------------------------------------------------------------------------
def collect_corpus_stats(ds: Dataset, tok_columns, text_column="text"):
    """Single pass over the dataset; collect per-tokenizer counters and per-row
    stats needed for fertility / OOV.

    Returns a dict:
        {
          tok_col: {
            "counter": Counter(token_id -> count),
            "total_tokens": int,
            "unk_token_count": int (filled later, since unk_id is from tokenizer),
            "id_to_unk_marker": placeholder, filled later,
          },
          ...
        }
        plus "total_words": int (whitespace-split words across the train split).
    """
    per_tok = {c: {"counter": Counter(), "total_tokens": 0} for c in tok_columns}
    total_words = 0

    # Iterate row-wise with a progress print
    n = len(ds)
    print_every = max(1, n // 20)
    for i, row in enumerate(ds):
        text = row[text_column]
        if text is not None:
            total_words += len(text.split())
        for c in tok_columns:
            ids = row[c]
            if ids is None:
                continue
            per_tok[c]["counter"].update(ids)
            per_tok[c]["total_tokens"] += len(ids)
        if (i + 1) % print_every == 0 or (i + 1) == n:
            print(f"  scanned {i + 1}/{n} rows")
    return per_tok, total_words


# ---------------------------------------------------------------------------
# Compression-section metrics
# ---------------------------------------------------------------------------
def compute_compression_metrics(counter, total_tokens, total_words, tokenizer):
    """vocab sizes, oov_rate (unk-emission rate), fertility.

    Vocab size is reported in three forms:
      - vocab_size_total : len(tokenizer.get_vocab()), the number of distinct
                           token ids the tokenizer can emit. Includes the base
                           vocabulary AND any added tokens (special tokens or
                           language-extension tokens). This is the right number
                           to report for the model's actual output space.
      - vocab_size_base  : tokenizer.vocab_size, the size of the base vocab
                           learned during tokenizer training (e.g. 32000 for
                           Llama-2-based tokenizers, before any extension).
      - added_tokens     : difference between the two, i.e. the number of
                           special / added tokens layered on top of the base.

    For tokenizers like Komodo that extend Llama-2's base vocab with extra
    SEA-language tokens, vocab_size_total and vocab_size_base will differ; for
    most others they will be equal (or differ only by a few special tokens).
    """
    vocab_size_base = int(tokenizer.vocab_size)
    try:
        vocab_size_total = int(len(tokenizer.get_vocab()))
    except Exception:
        # Fall back to len(tokenizer) if get_vocab() isn't available.
        vocab_size_total = int(len(tokenizer))
    added_tokens = vocab_size_total - vocab_size_base

    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        # Some byte-level / BPE tokenizers (e.g. modern Llama) have no <unk>.
        unk_count = 0
        oov_rate = 0.0
    else:
        unk_count = int(counter.get(unk_id, 0))
        oov_rate = unk_count / total_tokens if total_tokens > 0 else 0.0

    fertility = total_tokens / total_words if total_words > 0 else float("nan")

    return {
        "vocab_size_total": vocab_size_total,
        "vocab_size_base": vocab_size_base,
        "added_tokens": added_tokens,
        "unk_token_id": (None if unk_id is None else int(unk_id)),
        "unk_token_count": unk_count,
        "total_tokens": int(total_tokens),
        "total_words": int(total_words),
        "oov_rate": float(oov_rate),
        "fertility": float(fertility),
    }


# ---------------------------------------------------------------------------
# Distribution-section metrics (Lotz et al. 2025, Sec. 3.4 + footnote 7)
# ---------------------------------------------------------------------------
def compute_distribution_metrics(counter):
    """Compute cardinality, AUC, slope, power-law deviation.

    - cardinality: |support of counter|, using full distribution.
    - For AUC / slope / power_law_dev: sort frequencies descending, take ranks
      starting at 1; restrict to log(rank) <= 6 per footnote 7.
    """
    if len(counter) == 0:
        return {
            "cardinality": 0,
            "rank_freq_auc": float("nan"),
            "slope": float("nan"),
            "power_law_dev": float("nan"),
            "n_points_fit": 0,
        }

    cardinality = len(counter)

    # Sort frequencies in descending order; rank = 1, 2, 3, ...
    freqs = np.array(sorted(counter.values(), reverse=True), dtype=np.float64)
    ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)

    log_rank = np.log(ranks)
    log_freq = np.log(freqs)

    # Footnote 7: restrict to log(rank) <= 6
    mask = log_rank <= LOG_RANK_CUTOFF
    x = log_rank[mask]
    y = log_freq[mask]
    n_points = int(x.shape[0])

    if n_points < 2:
        # Degenerate vocabulary; can't fit a line or integrate meaningfully
        return {
            "cardinality": int(cardinality),
            "rank_freq_auc": float("nan"),
            "slope": float("nan"),
            "power_law_dev": float("nan"),
            "n_points_fit": n_points,
        }

    # AUC by Simpson's rule over (x, y) = (log rank, log freq)
    # scipy.integrate.simpson handles non-uniform x.
    auc = float(simpson(y=y, x=x))

    # Linear fit y ~ beta_0 + beta_1 * x  (least squares)
    beta1, beta0 = np.polyfit(x, y, 1)  # returns highest-degree first
    y_hat = beta0 + beta1 * x
    power_law_dev = float(np.mean(np.abs(y_hat - y)))

    return {
        "cardinality": int(cardinality),
        "rank_freq_auc": auc,
        "slope": float(beta1),
        "intercept": float(beta0),
        "power_law_dev": power_law_dev,
        "n_points_fit": n_points,
        "log_rank_cutoff": LOG_RANK_CUTOFF,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute tokenizer compression + distribution statistics "
                    "on a dataset's train split."
    )
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="HuggingFace dataset repo name, e.g. "
                             "'Exqrch/Rebuttal-javanese-pixelgpt'. "
                             "Will use the local HF cache if present.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF datasets cache directory (passed to "
                             "load_dataset). Defaults to HF_HOME if unset.")
    parser.add_argument("--train_split", type=str, default="train",
                        help="Which split to evaluate on (default: train).")
    parser.add_argument("--tok_grapheme_path", type=str, required=True)
    parser.add_argument("--tok_llama2_path", type=str,
                        default="ernie-research/DualGPT")
    parser.add_argument("--tok_komodo_path", type=str,
                        default="Yellow-AI-NLP/komodo-7b-base")
    parser.add_argument("--tok_mt5_path", type=str,
                        default="google/mt5-small")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to write the JSON results.")
    parser.add_argument("--lang_label", type=str, default="unknown",
                        help="Human-readable language label written into the "
                             "output JSON.")
    args = parser.parse_args()

    # Safety check on HF_HOME (mirrors data_renderer.py)
    hf_home = os.environ.get("HF_HOME")
    if hf_home is None:
        print("WARNING: HF_HOME not set. Tokenizer/dataset caches will go to "
              "the default location.")
    else:
        print(f"HF_HOME = {hf_home}")

    # Resolve cache_dir: explicit flag wins, else HF_HOME, else None (default).
    cache_dir = args.cache_dir or hf_home

    # 1) Load dataset
    train_ds = load_train_split(
        args.dataset_name, cache_dir=cache_dir, split=args.train_split
    )
    print(f"Train split size: {len(train_ds)} rows")
    print(f"Columns: {train_ds.column_names}")

    tok_columns = ["tok_grapheme", "tok_llama2", "tok_komodo", "tok_mt5"]
    for c in tok_columns:
        if c not in train_ds.column_names:
            raise ValueError(
                f"Column {c!r} not found in dataset. Available: "
                f"{train_ds.column_names}"
            )
    if args.text_column not in train_ds.column_names:
        raise ValueError(
            f"Text column {args.text_column!r} not found. Available: "
            f"{train_ds.column_names}"
        )

    # 2) Load tokenizers
    tokenizers = load_tokenizers(args)

    # 3) One streaming pass over the dataset to build token-id counters and a
    #    total whitespace-word count.
    print("Counting tokens and words over the train split...")
    per_tok, total_words = collect_corpus_stats(
        train_ds, tok_columns=tok_columns, text_column=args.text_column
    )
    print(f"Total whitespace words across train split: {total_words}")

    # 4) Compute metrics per tokenizer
    results = {
        "lang_label": args.lang_label,
        "dataset_name": args.dataset_name,
        "split": args.train_split,
        "cache_dir": cache_dir,
        "n_train_rows": len(train_ds),
        "total_words": int(total_words),
        "tokenizers": {},
    }

    for col in tok_columns:
        counter = per_tok[col]["counter"]
        total_tokens = per_tok[col]["total_tokens"]
        tok = tokenizers[col]

        comp = compute_compression_metrics(counter, total_tokens,
                                           total_words, tok)
        dist = compute_distribution_metrics(counter)
        results["tokenizers"][col] = {
            "tokenizer_path": getattr(
                tok, "name_or_path",
                {
                    "tok_grapheme": args.tok_grapheme_path,
                    "tok_llama2":  args.tok_llama2_path,
                    "tok_komodo":  args.tok_komodo_path,
                    "tok_mt5":     args.tok_mt5_path,
                }[col],
            ),
            "compression": comp,
            "distribution": dist,
        }

        # Pretty per-tokenizer printout
        print("\n" + "=" * 70)
        print(f"[{col}]  path = {results['tokenizers'][col]['tokenizer_path']}")
        print("-" * 70)
        print(f"  vocab_size_total: {comp['vocab_size_total']}")
        print(f"  vocab_size_base : {comp['vocab_size_base']}")
        print(f"  added_tokens    : {comp['added_tokens']}")
        print(f"  total_tokens    : {comp['total_tokens']}")
        print(f"  unk_token_id    : {comp['unk_token_id']}")
        print(f"  unk_token_count : {comp['unk_token_count']}")
        print(f"  oov_rate        : {comp['oov_rate']:.6e}")
        print(f"  fertility       : {comp['fertility']:.6f}")
        print(f"  cardinality     : {dist['cardinality']}")
        print(f"  rank_freq_auc   : {dist['rank_freq_auc']}")
        print(f"  slope           : {dist['slope']}")
        print(f"  power_law_dev   : {dist['power_law_dev']}")
        print(f"  (n points used  : {dist['n_points_fit']}, "
              f"log_rank_cutoff = {LOG_RANK_CUTOFF})")

    # 5) Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {args.output_path}")


if __name__ == "__main__":
    main()