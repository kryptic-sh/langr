//! Confidence calibration: over a labeled test set, report how the model's
//! `confidence` relates to empirical accuracy, and a threshold/operating-point
//! table (coverage vs precision) so a service can pick a "return `und` below X"
//! cutoff with a known accuracy promise.
//!
//! ```sh
//! cargo run --release --example calibrate -- \
//!   --tokenizer tokenizer.json --model model.bin --test-dir calib
//! ```

use anyhow::Result;
use clap::Parser;
use langr::Detector;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    tokenizer: PathBuf,
    #[arg(short, long)]
    model: PathBuf,
    /// One or more test dirs (e.g. a formal and an informal set); repeat `-d`.
    #[arg(short = 'd', long)]
    test_dir: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let detector = Detector::load(&args.tokenizer, &args.model)?;

    // (confidence, correct) for every sample across all test dirs.
    let mut samples: Vec<(f32, bool)> = Vec::new();
    let mut files: Vec<PathBuf> = Vec::new();
    for dir in &args.test_dir {
        for e in fs::read_dir(dir)? {
            let p = e?.path();
            if p.extension().is_some_and(|x| x == "txt") {
                files.push(p);
            }
        }
    }
    files.sort();

    for file in &files {
        let lang = file.file_stem().unwrap().to_string_lossy().into_owned();
        for line in fs::read_to_string(file)?.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let d = detector.detect(line)?;
            samples.push((d.confidence, d.language == lang));
        }
    }
    if samples.is_empty() {
        eprintln!("no samples");
        return Ok(());
    }
    let total = samples.len();

    // Reliability: empirical accuracy within each confidence decile.
    println!("confidence bucket      n     acc%");
    println!("{}", "-".repeat(38));
    for b in 0..10 {
        let lo = b as f32 / 10.0;
        let hi = lo + 0.1;
        let in_bucket: Vec<bool> = samples
            .iter()
            .filter(|(c, _)| *c >= lo && (*c < hi || (b == 9 && *c <= 1.0)))
            .map(|(_, ok)| *ok)
            .collect();
        let n = in_bucket.len();
        let acc = pct(in_bucket.iter().filter(|&&ok| ok).count(), n);
        println!("[{lo:.1},{hi:.1})       {n:>7}   {acc:>5.1}");
    }

    // Operating points: keep predictions with confidence >= t, else return und.
    println!("\nthreshold   coverage%   precision%   (abstain below t -> und)");
    println!("{}", "-".repeat(60));
    for i in 3..=9 {
        let t = i as f32 / 10.0;
        let kept: Vec<bool> = samples
            .iter()
            .filter(|(c, _)| *c >= t)
            .map(|(_, ok)| *ok)
            .collect();
        let coverage = pct(kept.len(), total);
        let precision = pct(kept.iter().filter(|&&ok| ok).count(), kept.len());
        println!("  >= {t:.1}      {coverage:>7.1}     {precision:>7.1}");
    }
    Ok(())
}

fn pct(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}
