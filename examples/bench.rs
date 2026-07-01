//! Split detection latency into tokenize vs aggregate to see where time goes.
//!
//! ```sh
//! cargo run --release --example bench -- \
//!   --tokenizer tokenizer.json --model model.bin --input test/eng.txt
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
    #[arg(short, long)]
    input: PathBuf,
    /// Cap each input to this many bytes before tokenizing.
    #[arg(long)]
    max_bytes: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut detector = Detector::load(&args.tokenizer, &args.model)?;
    if let Some(n) = args.max_bytes {
        detector = detector.with_max_input_bytes(n);
        println!("(capping inputs to {n} bytes)");
    }
    let enc = detector.encoder();

    let text = fs::read_to_string(&args.input)?;
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    let n = lines.len();
    if n == 0 {
        eprintln!("no non-empty lines in {}", args.input.display());
        return Ok(());
    }

    // Phase 1: tokenization only.
    let t0 = Instant::now();
    let mut all_ids: Vec<Vec<u32>> = Vec::with_capacity(n);
    for l in &lines {
        all_ids.push(enc.encode(l)?);
    }
    let tok = t0.elapsed();

    // Phase 2: aggregation only, over pre-tokenized ids.
    let t1 = Instant::now();
    let mut sink = 0usize;
    for ids in &all_ids {
        sink += detector.detect_ids(ids).scored_tokens;
    }
    let agg = t1.elapsed();
    std::hint::black_box(sink);

    let total_tokens: usize = all_ids.iter().map(|v| v.len()).sum();
    println!("{n} samples, {total_tokens} tokens");
    println!(
        "tokenize : {:>8.2}s  {:>7.2} us/sample",
        tok.as_secs_f64(),
        tok.as_micros() as f64 / n as f64
    );
    println!(
        "aggregate: {:>8.3}s  {:>7.3} us/sample  {:>6.1} ns/token",
        agg.as_secs_f64(),
        agg.as_micros() as f64 / n as f64,
        agg.as_nanos() as f64 / total_tokens as f64
    );
    let split = 100.0 * tok.as_secs_f64() / (tok.as_secs_f64() + agg.as_secs_f64());
    println!("tokenize is {split:.1}% of total");

    // Phase 3: full detect (tokenize + aggregate) single-threaded vs batched.
    let owned: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
    let refs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();

    let t2 = Instant::now();
    for l in &refs {
        std::hint::black_box(detector.detect(l)?);
    }
    let seq = t2.elapsed();

    let t3 = Instant::now();
    let out = detector.detect_batch(&refs)?;
    let par = t3.elapsed();
    std::hint::black_box(out.len());

    println!(
        "\ndetect seq   : {:>7.0}/s  ({:.2} us/sample)",
        n as f64 / seq.as_secs_f64(),
        seq.as_micros() as f64 / n as f64
    );
    println!(
        "detect batch : {:>7.0}/s  ({:.2} us/sample)  {:.1}x on {} cores",
        n as f64 / par.as_secs_f64(),
        par.as_micros() as f64 / n as f64,
        seq.as_secs_f64() / par.as_secs_f64(),
        rayon::current_num_threads(),
    );
    Ok(())
}
