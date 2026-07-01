//! `langr-detect` — detect the language of text from args or stdin.

use anyhow::Result;
use clap::Parser;
use langr::Detector;
use std::io::Read;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Detect the language mixture of text (args or stdin)")]
struct Args {
    /// HuggingFace tokenizer.json used to train the model.
    #[arg(short, long)]
    tokenizer: PathBuf,

    /// Trained model file.
    #[arg(short, long, default_value = "model.bin")]
    model: PathBuf,

    /// Second-language share needed to flag multilingual.
    #[arg(long, default_value_t = 0.15)]
    multilingual_threshold: f32,

    /// Text to classify; if omitted, read from stdin.
    text: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let detector = Detector::load(&args.tokenizer, &args.model)?
        .with_multilingual_threshold(args.multilingual_threshold);

    let text = if args.text.is_empty() {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf
    } else {
        args.text.join(" ")
    };

    let result = detector.detect(&text)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
