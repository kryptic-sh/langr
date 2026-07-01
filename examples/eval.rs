//! Evaluate a trained model against a labeled test set.
//!
//! Each `*.txt` file in `--test-dir` is one language; the file stem is the true
//! label and each line is one test sample. Reports per-language top-1 accuracy,
//! the overall accuracy, and detection throughput.
//!
//! ```sh
//! cargo run --release --example eval -- \
//!   --tokenizer tokenizer.json --model model.bin --test-dir test
//! ```

use anyhow::Result;
use clap::Parser;
use langr::Detector;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    tokenizer: PathBuf,
    #[arg(short, long)]
    model: PathBuf,
    #[arg(short = 'd', long)]
    test_dir: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let detector = Detector::load(&args.tokenizer, &args.model)?;

    let mut rows: Vec<(String, usize, usize)> = Vec::new(); // (lang, correct, total)
    let mut grand_correct = 0usize;
    let mut grand_total = 0usize;
    let mut elapsed = std::time::Duration::ZERO;

    let mut files: Vec<PathBuf> = fs::read_dir(&args.test_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    files.sort();

    for file in files {
        let lang = file.file_stem().unwrap().to_string_lossy().into_owned();
        let text = fs::read_to_string(&file)?;
        let mut correct = 0;
        let mut total = 0;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let start = Instant::now();
            let d = detector.detect(line)?;
            elapsed += start.elapsed();
            total += 1;
            if d.language == lang {
                correct += 1;
            }
        }
        grand_correct += correct;
        grand_total += total;
        rows.push((lang, correct, total));
    }

    println!(
        "{:<6} {:>8} {:>8} {:>8}",
        "lang", "correct", "total", "acc%"
    );
    println!("{}", "-".repeat(34));
    for (lang, correct, total) in &rows {
        let acc = if *total > 0 {
            100.0 * *correct as f64 / *total as f64
        } else {
            0.0
        };
        println!("{lang:<6} {correct:>8} {total:>8} {acc:>7.1}%");
    }
    println!("{}", "-".repeat(34));
    let overall = if grand_total > 0 {
        100.0 * grand_correct as f64 / grand_total as f64
    } else {
        0.0
    };
    println!(
        "{:<6} {grand_correct:>8} {grand_total:>8} {overall:>7.1}%",
        "ALL"
    );

    if grand_total > 0 {
        let us = elapsed.as_micros() as f64 / grand_total as f64;
        println!(
            "\n{grand_total} samples in {:.2}s  ->  {us:.1} us/detect  ({:.0}/s)",
            elapsed.as_secs_f64(),
            grand_total as f64 / elapsed.as_secs_f64()
        );
    }
    Ok(())
}
