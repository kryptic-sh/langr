# langr-v1 — model card

A production language-detection model for `langr`. 84 languages, ISO 639-3
labels, ~3.4 MB. Built from clean formal + informal corpora so it stays reliable
on both proper grammar and casual/subtitle-style text.

The model binary is **not** committed (it is a derived artifact). Rebuild it
exactly with the commands below, or fetch it from the release that matches
`models/manifest.json` (which pins the required tokenizer by SHA-256).

## What it is

- **Input:** any UTF-8 text. **Output:** language mixture + confidence (see the
  README's output schema).
- **Tokenizer:** Qwen3 `tokenizer.json` (Apache-2.0). The model is bound to it;
  a different tokenizer produces garbage. `manifest.json` pins its SHA-256.
- **Languages (84):** the "solid tier" — every language with ≥5k Tatoeba
  sentences. Includes all major internet languages plus well-represented
  constructed/minority languages (see `manifest.json` for the full list).

## Evaluation (top-1, single sentence)

| Test set                                                | Register | Accuracy             |
| ------------------------------------------------------- | -------- | -------------------- |
| **FLORES-200 devtest** (neutral, not a training source) | formal   | **92.6%** (65 langs) |
| Tatoeba held-out                                        | formal   | 89.1% (53 langs)     |
| OpenSubtitles held-out                                  | informal | 79.5% (46 langs)     |

Single short sentences are the hard case; longer real-world text scores higher.
Speed: ~12–35 µs/detect single-threaded.

## Confidence calibration

`confidence` (the top language's share) is well-calibrated and monotonic on a
mixed formal+informal held-out set (156 k samples):

| Keep predictions with `confidence` ≥ | Coverage | Precision |
| ------------------------------------ | -------- | --------- |
| 0.30                                 | 62%      | **95.0%** |
| 0.40                                 | 44%      | 96.5%     |
| 0.50                                 | 29%      | 98.2%     |
| 0.90                                 | 8%       | 99.9%     |

**Recommended policy:** return `und` (undetermined) when `confidence < 0.30`;
above it, expect ~95% precision. Raise the threshold for higher precision at the
cost of coverage. Coverage looks low only because the calibration set is single
sentences — real longer inputs sit far higher on this curve.

## Training data

| Source                   | Register                 | Langs | License                                  |
| ------------------------ | ------------------------ | ----- | ---------------------------------------- |
| Tatoeba                  | formal (clean sentences) | 84    | CC BY 2.0 FR                             |
| OPUS OpenSubtitles v2018 | informal (subtitles)     | 47    | OPUS terms — **gray for commercial use** |

Balanced ~18k sentences per source per language. Two sources were tried and
**rejected** (both regressed accuracy): raw CC-100 (dirty web crawl) and
OPUS-Wikipedia (only ~20 langs + proper-noun noise). See the README.

> **Licensing:** the trained model is derived frequency statistics, not the
> source text, which is lower-risk to redistribute. Still, OpenSubtitles is
> legally gray for commercial use — settle that before shipping commercially, or
> rebuild from `--source tatoeba` only (drops informal accuracy).

## Reproduce

Every step is a Rust bin/example — no ad-hoc scripts.

```sh
# 1. Tokenizer (verify its sha256 against models/manifest.json)
./scripts/fetch-tokenizer.sh

# 2. Corpora. The "solid tier" is languages with >= 5k Tatoeba sentences, which
#    --min selects automatically. For a byte-exact rebuild, instead pass
#    --langs with the list from models/manifest.json.
cargo run --release --features corpus --bin langr-corpus -- \
  --source tatoeba,opensubtitles --out corpus --test-out test --min 5000 --jobs 10

# 3. Neutral eval set (FLORES-200 -> test/flores/<code>.txt)
cargo run --release --features corpus --bin langr-corpus -- --source flores --test-out test

# 4. Train
cargo run --release --bin langr-train -- \
  --corpus corpus --tokenizer tokenizer.json --out models/langr-v1.bin

# 5. Evaluate (neutral) and calibrate (formal + informal at once)
cargo run --release --example eval      -- -t tokenizer.json -m models/langr-v1.bin -d test/flores
cargo run --release --example calibrate -- -t tokenizer.json -m models/langr-v1.bin \
  -d test/flores -d test/opensubtitles

# 6. Manifest (hashes + language list)
cargo run --release --features pack --bin langr-pack -- \
  --model models/langr-v1.bin --tokenizer tokenizer.json --out models/manifest.json
```

## Known limitations

- Confusable pairs still err on short text: sr/hr/bs, ru/uk, cs/sk.
- The 84-language set is the reliable tier; adding thin-corpus languages lowers
  overall accuracy (measured). Widen only with more clean data.
- Not tuned for hard netspeak (`lol wyd fr fr`); subtitles are the informal
  proxy. Add real social-media text for that register.
