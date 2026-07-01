# langr

Fast, low-memory, single-purpose **language detector** for Rust.

`langr` borrows an LLM's subword vocabulary (a HuggingFace `tokenizer.json`,
e.g. Qwen3's Apache-2.0 byte-level BPE) to turn text into token ids, then
aggregates a trained `token -> P(language)` table into a language mixture. No
neural net runs at inference time — just tokenize and sum.

- **Fast** — tokenize + a handful of adds per token. Microsecond range.
- **Low memory** — the model stores only the top-K languages per token, a few
  MB instead of a dense `vocab x languages` matrix.
- **Mixture-aware** — returns the full language breakdown with shares, so
  "80% French, 20% English" falls out directly.

## How it works

Two artifacts, both built offline, loaded at runtime:

1. **Tokenizer** — a pretrained subword vocab. `text -> [token ids]`.
2. **Model** — `token_id -> top-K {language: P(lang | token)}` plus a
   discriminative weight per token.

Detection is one pass:

```text
tokens = tokenize(text)
for t in tokens:
    for (lang, p) in model[t]:      # precomputed top-K
        score[lang] += weight[t] * p
normalize score by sum of weights
-> { fr: 0.80, en: 0.20, ... }
```

Averaging the per-token posteriors yields the mixture. Function-subwords shared
across languages get a low weight (from posterior entropy), so they add little
noise.

## Output schema

```json
{
  "language": "fr",
  "confidence": 0.80,
  "margin": 0.60,
  "is_multilingual": true,
  "languages": [
    { "lang": "fr", "score": 0.80 },
    { "lang": "en", "score": 0.20 }
  ],
  "scored_tokens": 10
}
```

## Quick start

```sh
# 1. fetch a permissive tokenizer vocab (Qwen3, Apache-2.0)
./scripts/fetch-tokenizer.sh

# 2. train a model from the bundled sample corpus (4 languages)
cargo run --release --bin langr-train -- \
  --corpus sample-corpus --tokenizer tokenizer.json --out model.bin

# 3. detect
cargo run --release --bin langr-detect -- \
  --tokenizer tokenizer.json --model model.bin \
  "Bonjour le monde, this is a small test."
```

The bundled `sample-corpus/` is only a smoke test. For real accuracy, train on
a proper multilingual corpus — see below.

## Training corpus

Layout: one subdirectory per language code, holding UTF-8 text files (any
extension), read line by line.

```text
corpus/
  en/  news.txt  wiki.txt
  fr/  news.txt
  ja/  wiki.txt
```

Recommended sources (permissive licenses):

- **Leipzig Corpora Collection** — per-language sentence packs, 250+ langs.
- **CC-100 / OSCAR** — bulk CommonCrawl text, 100+ langs.
- **Wikipedia dumps** — good tail-language coverage.
- **Tatoeba** — short sentences, 400+ langs.

Validate on held-out **FLORES-200**. Match your training domain to your input
domain (e.g. informal/social text) for best accuracy.

## Library use

```rust
use langr::Detector;

let detector = Detector::load("tokenizer.json", "model.bin")?;
let result = detector.detect("Bonjour le monde, this is a test.")?;
println!("{} ({:.0}%)", result.language, result.confidence * 100.0);
# Ok::<(), anyhow::Error>(())
```

## License

MIT. The tokenizer vocab you fetch has its own license — Qwen3's is Apache-2.0.
`langr` does not bundle any vocab.
