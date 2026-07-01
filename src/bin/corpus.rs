//! `langr-corpus` — download and prepare a training corpus from Tatoeba's
//! per-language sentence exports, labeled with uniform 3-char ISO 639-3 codes.
//!
//! Streams each language's `*.tsv.bz2`, extracts the sentence column, caps and
//! splits into train/test, and writes the [`langr-train`]-compatible layout:
//!
//! ```text
//! <out>/<code>/train.txt      <test-out>/<code>.txt
//! ```
//!
//! Downloads run in parallel with retries; failures are reported, never
//! silently dropped. Enable with `--features corpus`.

use anyhow::{anyhow, Context, Result};
use bzip2_rs::DecoderReader;
use clap::Parser;
use rayon::prelude::*;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Parser)]
#[command(about = "Download and prepare a Tatoeba training corpus (639-3 labels)")]
struct Args {
    /// Base URL of the Tatoeba per-language exports.
    #[arg(
        long,
        default_value = "https://downloads.tatoeba.org/exports/per_language"
    )]
    base_url: String,
    /// Output corpus root (one subdir per language).
    #[arg(short, long, default_value = "corpus")]
    out: PathBuf,
    /// Output directory for held-out test files (`<code>.txt`).
    #[arg(long, default_value = "test")]
    test_out: PathBuf,
    /// Max sentences to keep per language.
    #[arg(long, default_value_t = 20_000)]
    max_sentences: usize,
    /// First N sentences go to train; the remainder (if any) to test.
    #[arg(long, default_value_t = 18_000)]
    train: usize,
    /// Skip languages with fewer than this many sentences.
    #[arg(long, default_value_t = 200)]
    min: usize,
    /// Concurrent downloads.
    #[arg(short, long, default_value_t = 8)]
    jobs: usize,
    /// Attempts per language before giving up.
    #[arg(long, default_value_t = 3)]
    retries: usize,
    /// Explicit language codes to fetch; if empty, discover from the index.
    #[arg(long, value_delimiter = ',')]
    langs: Vec<String>,
}

enum Status {
    Added(usize),
    TooFew,
    Errored(String),
}

fn main() -> Result<()> {
    let args = Args::parse();
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(180))
        .build();

    let codes = if args.langs.is_empty() {
        let codes = discover_languages(&agent, &args.base_url)?;
        eprintln!("discovered {} languages from index", codes.len());
        codes
    } else {
        args.langs.clone()
    };

    std::fs::create_dir_all(&args.out)?;
    std::fs::create_dir_all(&args.test_out)?;

    let done = AtomicUsize::new(0);
    let total = codes.len();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs)
        .build()?;

    let results: Vec<(String, Status)> = pool.install(|| {
        codes
            .par_iter()
            .map(|code| {
                let status = fetch_language(&agent, &args, code);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                match &status {
                    Status::Added(c) => eprintln!("[{n}/{total}] + {code}: {c}"),
                    Status::Errored(e) => eprintln!("[{n}/{total}] ! {code}: {e}"),
                    Status::TooFew => {}
                }
                (code.clone(), status)
            })
            .collect()
    });

    let mut added = 0;
    let mut too_few = Vec::new();
    let mut errored = Vec::new();
    for (code, status) in &results {
        match status {
            Status::Added(_) => added += 1,
            Status::TooFew => too_few.push(code.as_str()),
            Status::Errored(_) => errored.push(code.as_str()),
        }
    }

    eprintln!(
        "\nADDED {added}   too_few {}   errored {}",
        too_few.len(),
        errored.len()
    );
    if !errored.is_empty() {
        errored.sort_unstable();
        eprintln!("errored: {}", errored.join(" "));
    }
    Ok(())
}

/// Parse the Tatoeba autoindex for 3-letter language directory names.
fn discover_languages(agent: &ureq::Agent, base_url: &str) -> Result<Vec<String>> {
    let url = format!("{}/", base_url.trim_end_matches('/'));
    let html = agent
        .get(&url)
        .call()
        .with_context(|| format!("fetch index {url}"))?
        .into_string()?;

    let mut codes = std::collections::BTreeSet::new();
    for part in html.split("href=\"").skip(1) {
        // Each href looks like `abc/"...`; take up to the closing quote.
        if let Some(end) = part.find('"') {
            let href = &part[..end];
            let code = href.trim_end_matches('/');
            if code.len() == 3 && code.bytes().all(|b| b.is_ascii_lowercase()) {
                codes.insert(code.to_string());
            }
        }
    }
    Ok(codes.into_iter().collect())
}

/// Download one language with retries, then write train/test files.
fn fetch_language(agent: &ureq::Agent, args: &Args, code: &str) -> Status {
    let url = format!(
        "{}/{code}/{code}_sentences.tsv.bz2",
        args.base_url.trim_end_matches('/')
    );

    let mut last_err = String::new();
    for attempt in 0..args.retries {
        match download_sentences(agent, &url, args.max_sentences) {
            Ok(sents) => {
                if sents.len() < args.min {
                    return Status::TooFew;
                }
                return match write_split(args, code, &sents) {
                    Ok(()) => Status::Added(sents.len()),
                    Err(e) => Status::Errored(e.to_string()),
                };
            }
            Err(e) => {
                last_err = e.to_string();
                if attempt + 1 < args.retries {
                    std::thread::sleep(Duration::from_millis(1500 * (attempt as u64 + 1)));
                }
            }
        }
    }
    Status::Errored(last_err)
}

/// Stream-decompress the bz2 export and collect the sentence column (3rd TSV
/// field), stopping once `max` sentences are gathered.
fn download_sentences(agent: &ureq::Agent, url: &str, max: usize) -> Result<Vec<String>> {
    let resp = agent.get(url).call().map_err(|e| anyhow!("{e}"))?;
    let reader = BufReader::new(DecoderReader::new(resp.into_reader()));

    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        // id \t lang \t text
        if let Some(text) = line.split('\t').nth(2) {
            let text = text.trim();
            if !text.is_empty() {
                out.push(text.to_string());
                if out.len() >= max {
                    break;
                }
            }
        }
    }
    Ok(out)
}

fn write_split(args: &Args, code: &str, sents: &[String]) -> Result<()> {
    let dir = args.out.join(code);
    std::fs::create_dir_all(&dir)?;
    let split = args.train.min(sents.len());
    write_lines(&dir.join("train.txt"), &sents[..split])?;
    if sents.len() > args.train {
        write_lines(
            &args.test_out.join(format!("{code}.txt")),
            &sents[args.train..],
        )?;
    }
    Ok(())
}

fn write_lines(path: &Path, lines: &[String]) -> Result<()> {
    let mut body = lines.join("\n");
    body.push('\n');
    std::fs::write(path, body).with_context(|| format!("write {}", path.display()))
}
