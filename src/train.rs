//! Offline training: turn a labeled corpus into a [`LangModel`].
//!
//! Corpus layout is one subdirectory per language, named by its code, holding
//! one or more UTF-8 text files (any extension). Every file is read line by
//! line, so sentence-per-line corpora (Leipzig, Tatoeba) and free text both
//! work.
//!
//! ```text
//! corpus/
//!   en/  news.txt  wiki.txt
//!   fr/  news.txt
//!   ja/  wiki.txt
//! ```
//!
//! Algorithm (Naive-Bayes bag-of-subwords):
//!   1. Count token frequencies per language, capped for balance.
//!   2. `P(token | lang)` via add-k smoothing.
//!   3. `P(lang | token) ∝ P(token | lang)` (uniform language prior).
//!   4. Keep the top-K languages per token; store a discriminative weight
//!      (`1 - normalized entropy`) so shared function-subwords count for less.

use crate::model::{LangId, LangModel, Posterior, MODEL_VERSION};
use crate::tokenizer::Encoder;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Knobs for [`train`].
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Languages kept per token in the final table.
    pub top_k: usize,
    /// Per-language cap on counted tokens, to balance corpus sizes.
    pub max_tokens_per_lang: u64,
    /// Add-k Laplace smoothing applied to `P(token | lang)`.
    pub add_k: f64,
    /// Drop tokens seen fewer than this many times across all languages.
    pub min_token_count: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            top_k: 4,
            max_tokens_per_lang: 5_000_000,
            add_k: 0.5,
            min_token_count: 2,
        }
    }
}

/// Per-language accumulation during counting.
struct LangCounts {
    code: String,
    counts: HashMap<u32, u64>,
    total: u64,
}

/// Train a [`LangModel`] from `corpus_root` using `encoder`.
pub fn train(
    corpus_root: impl AsRef<Path>,
    encoder: &Encoder,
    cfg: &TrainConfig,
) -> Result<LangModel> {
    let vocab_size = encoder.vocab_size();
    let mut langs = count_corpus(corpus_root.as_ref(), encoder, cfg)?;

    // Drop empty languages so ids stay dense and valid.
    langs.retain(|l| l.total > 0);
    if langs.is_empty() {
        bail!("no tokens counted; is the corpus empty?");
    }
    langs.sort_by(|a, b| a.code.cmp(&b.code));
    let num_langs = langs.len();

    // Which tokens are worth modeling.
    let mut token_totals: HashMap<u32, u64> = HashMap::new();
    for l in &langs {
        for (&tok, &c) in &l.counts {
            *token_totals.entry(tok).or_default() += c;
        }
    }

    let mut token_post: Vec<Posterior> = vec![Vec::new(); vocab_size];
    let mut token_weight: Vec<f32> = vec![0.0; vocab_size];
    let denom: Vec<f64> = langs
        .iter()
        .map(|l| l.total as f64 + cfg.add_k * vocab_size as f64)
        .collect();
    let h_max = (num_langs as f64).ln();

    // Build each token's posterior independently across rayon workers, then
    // scatter the results into the dense tables.
    let built: Vec<(u32, Posterior, f32)> = token_totals
        .par_iter()
        .filter_map(|(&tok, &total_c)| {
            if total_c < cfg.min_token_count || (tok as usize) >= vocab_size {
                return None;
            }

            // P(token | lang) for every language, then normalize to P(lang | token).
            let mut posterior: Vec<(usize, f64)> = Vec::with_capacity(num_langs);
            let mut sum = 0.0f64;
            for (li, l) in langs.iter().enumerate() {
                let c = l.counts.get(&tok).copied().unwrap_or(0);
                let p = (c as f64 + cfg.add_k) / denom[li];
                posterior.push((li, p));
                sum += p;
            }
            for entry in &mut posterior {
                entry.1 /= sum;
            }

            // Discriminative weight from entropy of the full posterior.
            let entropy: f64 = posterior
                .iter()
                .map(|&(_, p)| if p > 0.0 { -p * p.ln() } else { 0.0 })
                .sum();
            let weight = if h_max > 0.0 {
                (1.0 - entropy / h_max) as f32
            } else {
                0.0
            };

            // Keep top-K languages and renormalize their share.
            posterior.sort_by(|a, b| b.1.total_cmp(&a.1));
            posterior.truncate(cfg.top_k);
            let top_sum: f64 = posterior.iter().map(|&(_, p)| p).sum();
            let entry: Posterior = posterior
                .iter()
                .map(|&(li, p)| (li as LangId, (p / top_sum) as f32))
                .collect();

            Some((tok, entry, weight))
        })
        .collect();

    for (tok, entry, weight) in built {
        token_post[tok as usize] = entry;
        token_weight[tok as usize] = weight;
    }

    Ok(LangModel {
        version: MODEL_VERSION,
        langs: langs.into_iter().map(|l| l.code).collect(),
        top_k: cfg.top_k,
        vocab_size,
        token_post,
        token_weight,
    })
}

/// Walk the corpus and count token frequencies per language, one language per
/// rayon worker. Tokenization dominates training cost, so this scales roughly
/// linearly with cores while the number of languages exceeds the core count.
fn count_corpus(root: &Path, encoder: &Encoder, cfg: &TrainConfig) -> Result<Vec<LangCounts>> {
    let mut dirs: Vec<(String, PathBuf)> = Vec::new();
    for entry in
        fs::read_dir(root).with_context(|| format!("read corpus dir {}", root.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let code = entry.file_name().to_string_lossy().into_owned();
            dirs.push((code, path));
        }
    }

    dirs.par_iter()
        .map(|(code, path)| {
            let mut lang = LangCounts {
                code: code.clone(),
                counts: HashMap::new(),
                total: 0,
            };
            count_lang_dir(path, encoder, cfg, &mut lang)?;
            eprintln!("  {}: {} tokens", lang.code, lang.total);
            Ok(lang)
        })
        .collect()
}

fn count_lang_dir(
    dir: &Path,
    encoder: &Encoder,
    cfg: &TrainConfig,
    lang: &mut LangCounts,
) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        if lang.total >= cfg.max_tokens_per_lang {
            break;
        }
        let path = entry?.path();
        if path.is_dir() {
            count_lang_dir(&path, encoder, cfg, lang)?;
            continue;
        }
        let file = fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
        for line in BufReader::new(file).lines() {
            let line = line?;
            let text = line.trim();
            if text.is_empty() {
                continue;
            }
            for id in encoder.encode(text)? {
                *lang.counts.entry(id).or_default() += 1;
                lang.total += 1;
            }
            if lang.total >= cfg.max_tokens_per_lang {
                break;
            }
        }
    }
    Ok(())
}
