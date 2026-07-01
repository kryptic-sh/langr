# langr

Fast, low-memory, single-purpose **language detector** for Rust.

`langr` borrows an LLM's subword vocabulary (a HuggingFace `tokenizer.json`,
e.g. Qwen3's Apache-2.0 byte-level BPE) to turn text into token ids, then
aggregates a trained `token -> P(language)` table into a language mixture. No
neural net runs at inference time — just tokenize and sum.

- **Fast** — tokenize + a handful of adds per token. Microsecond range.
- **Low memory** — the model stores only the top-K languages per token, a few MB
  instead of a dense `vocab x languages` matrix.
- **Mixture-aware** — returns the full language breakdown with shares, so "80%
  French, 20% English" falls out directly.

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
  "confidence": 0.8,
  "margin": 0.6,
  "is_multilingual": true,
  "languages": [
    { "lang": "fr", "score": 0.8 },
    { "lang": "en", "score": 0.2 }
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

The bundled `sample-corpus/` is only a smoke test. For real accuracy, train on a
proper multilingual corpus — see below.

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

## Licensing & data

`langr` ships **code only**. It bundles no vocab, no corpus, and no trained
model — you fetch a tokenizer and bring your own training data. This keeps the
repository cleanly MIT with no third-party data-license entanglement.

- **Code + Rust dependencies** — all permissive (MIT / Apache-2.0 / BSD / Zlib /
  Unlicense / BSL / Unicode). No copyleft. The source is MIT.
- **Tokenizer vocab** — a _runtime input_ you fetch, not a bundled asset.
  Qwen3's is Apache-2.0. Used with MIT code that is fine; just don't commit or
  relicense it (the repo `.gitignore`s `tokenizer.json`).
- **Training corpora** — bring your own. Note many common corpora are **not**
  freely redistributable: Leipzig is CC BY-NC (non-commercial), Wikipedia and
  FLORES-200 are CC BY-SA (share-alike), CC-100 / OSCAR carry CommonCrawl
  third-party copyright. Tatoeba (CC BY) and UDHR (public domain) are the
  redistributable ones. The repo `.gitignore`s `/corpus/`.
- **Trained `model.bin`** — a derivative statistical table. Whether it counts as
  a derivative of the corpus is legally unsettled; the repo `.gitignore`s
  `*.bin`. If you publish a pretrained model, do it as a **separate release
  artifact** (not in this MIT source tree), trained only on permissive data,
  with its own notice.

### Third-party notices

Binaries statically compile the dependencies. Generate the bundled-license
notice file before distributing binaries:

```sh
cargo install cargo-about   # once
./scripts/gen-notices.sh    # writes THIRD-PARTY-LICENSES.md
```

The accepted-license allowlist lives in `about.toml`; `cargo about` fails if a
dependency ever introduces a license not on it, so copyleft can't slip in
unnoticed.

## License

MIT — see [`LICENSE`](LICENSE).
