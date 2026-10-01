"""
remove_watermark.py

Removes LLM watermarks (KGW, Unigram, SIR) from text with small,
context-aware word substitutions, while keeping the meaning intact.

Pipeline for each text:
  1. Pick ~12% of the alphabetic words longer than 3 characters.
  2. Mask each one and let BERT (masked LM) propose replacements that fit
     the context. No external word lists: BERT itself rarely proposes
     alternatives for function words, because the context fixes them.
  3. Score every candidate by the cosine similarity between the edited
     and the original text (all-mpnet-base-v2) and keep the best one only
     if the similarity stays >= 0.85.

Why it works against each scheme:
  - KGW:     changing a token changes the hash of the next window, so the
             green/red lists of the following tokens are re-drawn.
  - Unigram: green tokens are replaced by tokens that are in the red list
             about half of the time, diluting the green-token excess.
  - SIR:     the meaning (and so the semantic embedding) is preserved, but
             the token IDs change, which disrupts the watermark logits.

Runs fully offline once the two models are cached (no internet needed on
the compute node).
"""
import argparse
import json
import random
import sys
import zlib
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHARS = 3990  # character limit of the challenge format

print(f"Device: {DEVICE}\nLoading models...", flush=True)
mlm_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
mlm_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(DEVICE).eval()
sem_model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)


def get_bert_candidates(words, word_idx, top_k=15):
    """Mask one word and return BERT's top-k whole-word replacements."""
    masked = words.copy()
    masked[word_idx] = mlm_tokenizer.mask_token
    inputs = mlm_tokenizer(" ".join(masked), return_tensors="pt",
                           truncation=True, max_length=512).to(DEVICE)

    mask_positions = (inputs["input_ids"][0] == mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:  # the mask fell beyond BERT's 512-token window
        return []

    with torch.no_grad():
        logits = mlm_model(**inputs).logits
    top_ids = logits[0, mask_positions[0]].topk(top_k).indices.tolist()

    candidates = []
    for token_id in top_ids:
        token_str = mlm_tokenizer.decode([token_id]).strip()
        if token_str.isalpha() and len(token_str) > 1:  # no subwords or punctuation
            candidates.append(token_str)
    return candidates


def attack_text(text, replace_rate=0.12, sem_threshold=0.85, n_trials=5):
    """Return a watermark-free version of `text` with the same meaning."""
    text = text[:MAX_CHARS]
    words = text.split()
    if not words:
        return text

    positions = [i for i, w in enumerate(words) if w.isalpha() and len(w) > 3]
    if not positions:
        return text

    # Deterministic per text (Python's built-in hash() changes between runs)
    rng = random.Random(zlib.crc32(text.encode("utf-8")))
    n_replace = max(1, int(len(positions) * replace_rate))
    chosen = rng.sample(positions, min(n_replace, len(positions)))

    orig_emb = sem_model.encode(text, convert_to_tensor=True)
    modified = words.copy()

    for idx in chosen:
        original = words[idx]
        candidates = [c for c in get_bert_candidates(words, idx) if c.lower() != original.lower()]
        if not candidates:
            continue

        trial_texts, trial_words = [], []
        for cand in candidates[:n_trials]:
            if original[0].isupper():  # keep capitalisation
                cand = cand[0].upper() + cand[1:]
            trial = modified.copy()
            trial[idx] = cand
            trial_texts.append(" ".join(trial))
            trial_words.append(cand)

        # Encode all trials in one batch
        embs = sem_model.encode(trial_texts, convert_to_tensor=True, batch_size=len(trial_texts))
        sims = util.cos_sim(orig_emb, embs)[0]
        best = int(sims.argmax())
        if float(sims[best]) >= sem_threshold:
            modified[idx] = trial_words[best]

    return " ".join(modified)[:MAX_CHARS]


def main():
    p = argparse.ArgumentParser(description="Remove LLM watermarks from a JSONL file of texts.")
    p.add_argument("--input", type=Path, default=Path("test.jsonl"),
                   help='JSONL with one {"id": ..., "text": ...} per line')
    p.add_argument("--output", type=Path, default=Path("submission.jsonl"))
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"File not found: {args.input}")

    with args.input.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(rows)} texts.", flush=True)

    with args.output.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="Attacking", unit="text"):
            out = {"id": str(row["id"]), "text": attack_text(row["text"])}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
