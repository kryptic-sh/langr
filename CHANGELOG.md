# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-01

Initial release.

### Added

- **Detection library.** `Detector` turns text into subword token ids (via a
  pretrained LLM `tokenizer.json`) and aggregates a trained
  `token -> P(language)` table into a language mixture. Returns a `Detection`
  with the top language, confidence, top1–top2 margin, an `is_multilingual`
  flag, the full sorted mixture, and the scored-token count.
- **Model format.** `LangModel` stores a compact top-K-per-token posterior table
  (a few MB, not a dense `vocab × languages` matrix), with a version guard and
  `validate()` of structural invariants at load.
- **Ergonomics.** `Detector::detect`, `detect_batch` (rayon fan-out),
  `detect_ids`, `with_multilingual_threshold`, `with_max_input_bytes` (prefix
  cap for long text), and an `encoder()` accessor. `Detector` is `Sync`.
- **Training.** `train()` builds a model from a labeled corpus (Naive-Bayes
  bag-of-subwords with entropy-weighted tokens), parallelized across languages,
  with a progress callback instead of library-side I/O.
- **Binaries.** `langr-train`, `langr-detect`, plus feature-gated `langr-corpus`
  (Tatoeba / OpenSubtitles / Wikipedia / CC-100 / FLORES sources, 639-3 labels)
  and `langr-pack` (release manifest with model + tokenizer hashes).
- **Examples.** `eval` (per-language accuracy), `bench` (tokenize-vs-aggregate
  split and throughput), and `calibrate` (confidence → empirical-accuracy
  operating points).
- **Performance.** `target-cpu=native` build config; offset-free `encode_fast`
  and a thread-local accumulator on the detection hot path.
- **Docs & CI.** README, `MODEL_CARD.md` for the langr-v1 model, third-party
  license notices, and GitHub Actions for fmt/clippy/machete/test plus a weekly
  `cargo-deny` cron.

[Unreleased]: https://github.com/kryptic-sh/langr/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kryptic-sh/langr/releases/tag/v0.1.0
