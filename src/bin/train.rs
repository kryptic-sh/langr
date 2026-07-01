//! `langr-train` — build a detector model from a labeled corpus.

use anyhow::Result;
use clap::Parser;
use langr::train::{train, TrainConfig};
use langr::Encoder;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Train a langr language-detection model from a labeled corpus")]
struct Args {
    /// Corpus root: one subdirectory per language code, holding text files.
    #[arg(short, long)]
    corpus: PathBuf,

    /// HuggingFace tokenizer.json (e.g. Qwen3's vocab).
    #[arg(short, long)]
    tokenizer: PathBuf,

    /// Output model path.
    #[arg(short, long, default_value = "model.bin")]
    out: PathBuf,

    /// Languages kept per token.
    #[arg(long, default_value_t = 4)]
    top_k: usize,

    /// Per-language token cap (corpus balancing).
    #[arg(long, default_value_t = 5_000_000)]
    max_tokens: u64,

    /// Add-k smoothing.
    #[arg(long, default_value_t = 0.5)]
    add_k: f64,

    /// Drop tokens seen fewer than this many times.
    #[arg(long, default_value_t = 2)]
    min_count: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("loading tokenizer {}", args.tokenizer.display());
    let encoder = Encoder::from_file(&args.tokenizer)?;
    eprintln!("vocab size: {}", encoder.vocab_size());

    let cfg = TrainConfig {
        top_k: args.top_k,
        max_tokens_per_lang: args.max_tokens,
        add_k: args.add_k,
        min_token_count: args.min_count,
    };

    eprintln!("counting corpus {}", args.corpus.display());
    let model = train(&args.corpus, &encoder, &cfg, |code, total| {
        eprintln!("  {code}: {total} tokens");
    })?;

    eprintln!(
        "languages: {}, modeled tokens: {}/{}",
        model.languages().len(),
        model.modeled_tokens(),
        model.vocab_size()
    );

    model.save(&args.out)?;
    eprintln!("wrote {}", args.out.display());
    Ok(())
}
