# LLM Watermark Removal

Removing LLM watermarks (**KGW**, **Unigram** and **SIR**) from generated text with small, context-aware word substitutions, while keeping the meaning intact.

Built for the *CISPA Hackathon Barcelona 2026*, where our team placed **2nd**.

## The problem

Watermarked LLMs bias their sampling so that generated text carries a hidden statistical signal. The goal is to rewrite watermarked texts so that detectors no longer find the signal, while scoring well on two quality checks: semantic similarity to the original (sentence embeddings) and token-level similarity (so a full rewrite is penalised).

That makes it a trade-off: every edit weakens the watermark but also costs similarity.

## Approach

A targeted paraphrasing attack that edits only about **12% of the words**:

1. **Pick positions.** A random sample of alphabetic words longer than 3 characters, seeded per text so results are reproducible.
2. **Propose replacements with BERT.** Each chosen word is masked, and `bert-base-uncased` suggests whole words that fit the context. No synonym lists are needed: for function words like *the* or *is*, the context leaves no room, so BERT simply repeats the original.
3. **Keep the meaning.** Every candidate sentence is embedded with `all-mpnet-base-v2`, and the best one is kept only if its cosine similarity with the original stays ≥ 0.85. Candidates are encoded in a single batch for speed.

### Why it breaks each scheme

| Scheme | What it relies on | Effect of a substitution |
|---|---|---|
| KGW | Green list seeded by the previous token(s) | Re-draws the green lists of the following tokens too |
| Unigram | One fixed green list for the whole vocabulary | The new token is red about half of the time |
| SIR | Semantic-invariant watermark logits | Meaning is kept, but token IDs change |

## Design notes

Earlier versions used a hand-written synonym table, then spaCy and NLTK part-of-speech tagging to choose only content words. The final version drops all of them: the compute node had no internet access to download NLTK data, and BERT turned out to filter function words on its own. The result is simpler, fully offline once the two models are cached, and depends only on `transformers` and `sentence-transformers`.

## Usage

```bash
pip install -r requirements.txt
python remove_watermark.py --input test.jsonl --output submission.jsonl
```

The input is a JSONL file with one `{"id": ..., "text": ...}` per line. A GPU makes it much faster.

## References

- Kirchenbauer et al., *A Watermark for Large Language Models*, ICML 2023 (KGW)
- Zhao et al., *Provable Robust Watermarking for AI-Generated Text*, ICLR 2024 (Unigram)
- Liu et al., *A Semantic Invariant Robust Watermark for Large Language Models*, ICLR 2024 (SIR)
