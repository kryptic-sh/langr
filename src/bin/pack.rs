//! `langr-pack` — emit a release manifest for a trained model.
//!
//! Reads the language list from the model, hashes the model and the tokenizer
//! it is bound to, and writes a JSON manifest. Pinning the tokenizer's SHA-256
//! matters: the model is meaningless with a different tokenizer.
//!
//! ```sh
//! cargo run --release --features pack --bin langr-pack -- \
//!   --model models/langr-v1.bin --tokenizer tokenizer.json --out models/manifest.json
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use langr::model::MODEL_VERSION;
use langr::LangModel;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(about = "Write a release manifest (hashes + language list) for a model")]
struct Args {
    /// Trained model file.
    #[arg(short, long)]
    model: PathBuf,
    /// Tokenizer the model is bound to.
    #[arg(short, long)]
    tokenizer: PathBuf,
    /// Output manifest path.
    #[arg(short, long, default_value = "manifest.json")]
    out: PathBuf,
    /// Model/release name.
    #[arg(long, default_value = "langr-v1")]
    name: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model = LangModel::load(&args.model)?;

    let manifest = serde_json::json!({
        "name": args.name,
        "code_scheme": "ISO 639-3",
        "language_count": model.languages().len(),
        "languages": model.languages(),
        "tokenizer": {
            "file": file_name(&args.tokenizer),
            "sha256": sha256(&args.tokenizer)?,
        },
        "model": {
            "file": file_name(&args.model),
            "sha256": sha256(&args.model)?,
            "bytes": std::fs::metadata(&args.model)?.len(),
            "format_version": MODEL_VERSION,
        },
    });

    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&args.out, json + "\n")
        .with_context(|| format!("write {}", args.out.display()))?;
    eprintln!(
        "wrote {} ({} languages)",
        args.out.display(),
        model.languages().len()
    );
    Ok(())
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn sha256(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let digest = Sha256::digest(&bytes);
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}
