//! `langr-corpus` — download and prepare a training corpus, labeled with
//! uniform 3-char ISO 639-3 codes, in the [`langr-train`] layout:
//!
//! ```text
//! <out>/<code>/<source>.txt      <test-out>/<code>.txt
//! ```
//!
//! Sources (mix registers for robustness — formal + conversational):
//! - **tatoeba** — clean per-language sentences (bz2 TSV), discovered from the
//!   site index; codes are already 639-3.
//! - **opensubtitles** — conversational movie/TV subtitles (gzip plaintext,
//!   OPUS), the informal/slang-adjacent register; mapped to 639-3.
//! - **cc100** — CommonCrawl-derived monolingual text (xz plaintext); raw web
//!   crawl, noisy — clean before use.
//!
//! Downloads run in parallel with retries; failures are reported, never
//! silently dropped. Enable with `--features corpus`.

use anyhow::{anyhow, Context, Result};
use bzip2_rs::DecoderReader;
use clap::Parser;
use rayon::prelude::*;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Parser)]
#[command(about = "Download and prepare a training corpus (639-3 labels)")]
struct Args {
    /// Sources to pull: `tatoeba`, `opensubtitles`, `cc100` (comma-separated).
    #[arg(long, value_delimiter = ',', default_value = "tatoeba")]
    source: Vec<String>,
    /// Base URL for Tatoeba per-language exports.
    #[arg(
        long,
        default_value = "https://downloads.tatoeba.org/exports/per_language"
    )]
    tatoeba_url: String,
    /// Base URL for CC-100 monolingual files.
    #[arg(long, default_value = "https://data.statmt.org/cc-100")]
    cc100_url: String,
    /// Base URL for OPUS OpenSubtitles monolingual files.
    #[arg(
        long,
        default_value = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono"
    )]
    opensubtitles_url: String,
    /// Output corpus root (one subdir per language).
    #[arg(short, long, default_value = "corpus")]
    out: PathBuf,
    /// Output directory for held-out test files (`<code>.txt`).
    #[arg(long, default_value = "test")]
    test_out: PathBuf,
    /// Max sentences to keep per language per source.
    #[arg(long, default_value_t = 20_000)]
    max_sentences: usize,
    /// First N sentences go to train; the remainder to test (unless --no-test).
    #[arg(long, default_value_t = 18_000)]
    train: usize,
    /// Skip languages with fewer than this many sentences.
    #[arg(long, default_value_t = 200)]
    min: usize,
    /// Put everything into train; don't create test files (for augmentation).
    #[arg(long)]
    no_test: bool,
    /// Concurrent downloads.
    #[arg(short, long, default_value_t = 8)]
    jobs: usize,
    /// Attempts per language before giving up.
    #[arg(long, default_value_t = 3)]
    retries: usize,
    /// Explicit 639-3 codes to fetch; if empty, use each source's full set.
    #[arg(long, value_delimiter = ',')]
    langs: Vec<String>,
}

#[derive(Clone, Copy)]
enum Decomp {
    Bzip2,
    Xz,
    Gzip,
}

/// One download unit: a source URL producing sentences for one 639-3 code.
struct Job {
    code: String,
    url: String,
    source: &'static str,
    decomp: Decomp,
    /// TSV column holding the text (Tatoeba); `None` = whole line (CC-100).
    tsv_col: Option<usize>,
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
        .timeout_read(Duration::from_secs(300))
        .build();

    let mut jobs = Vec::new();
    for source in &args.source {
        match source.as_str() {
            "tatoeba" => jobs.extend(tatoeba_jobs(&agent, &args)?),
            "cc100" => jobs.extend(cc100_jobs(&args)),
            "opensubtitles" => jobs.extend(opensubtitles_jobs(&args)),
            other => {
                return Err(anyhow!(
                    "unknown source '{other}' (want tatoeba|opensubtitles|cc100)"
                ))
            }
        }
    }
    eprintln!(
        "{} download jobs across {} source(s)",
        jobs.len(),
        args.source.len()
    );

    std::fs::create_dir_all(&args.out)?;
    if !args.no_test {
        std::fs::create_dir_all(&args.test_out)?;
    }

    let done = AtomicUsize::new(0);
    let total = jobs.len();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs)
        .build()?;

    let results: Vec<(String, &str, Status)> = pool.install(|| {
        jobs.par_iter()
            .map(|job| {
                let status = run_job(&agent, &args, job);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                match &status {
                    Status::Added(c) => {
                        eprintln!("[{n}/{total}] + {}:{} {c}", job.source, job.code)
                    }
                    Status::Errored(e) => {
                        eprintln!("[{n}/{total}] ! {}:{} {e}", job.source, job.code)
                    }
                    Status::TooFew => {}
                }
                (job.code.clone(), job.source, status)
            })
            .collect()
    });

    let mut added = 0;
    let mut too_few = 0;
    let mut errored = Vec::new();
    for (code, source, status) in &results {
        match status {
            Status::Added(_) => added += 1,
            Status::TooFew => too_few += 1,
            Status::Errored(_) => errored.push(format!("{source}:{code}")),
        }
    }
    eprintln!(
        "\nADDED {added}   too_few {too_few}   errored {}",
        errored.len()
    );
    if !errored.is_empty() {
        errored.sort();
        eprintln!("errored: {}", errored.join(" "));
    }
    Ok(())
}

/// Build Tatoeba jobs by discovering 3-letter codes from the site index.
fn tatoeba_jobs(agent: &ureq::Agent, args: &Args) -> Result<Vec<Job>> {
    let codes = if args.langs.is_empty() {
        let codes = discover_tatoeba(agent, &args.tatoeba_url)?;
        eprintln!("discovered {} tatoeba languages", codes.len());
        codes
    } else {
        args.langs.clone()
    };
    let base = args.tatoeba_url.trim_end_matches('/');
    Ok(codes
        .into_iter()
        .map(|code| Job {
            url: format!("{base}/{code}/{code}_sentences.tsv.bz2"),
            source: "tatoeba",
            decomp: Decomp::Bzip2,
            tsv_col: Some(2),
            code,
        })
        .collect())
}

/// Build CC-100 jobs from the fixed language list, mapped to 639-3.
fn cc100_jobs(args: &Args) -> Vec<Job> {
    let base = args.cc100_url.trim_end_matches('/');
    CC100_LANGS
        .iter()
        .filter(|(_, iso)| args.langs.is_empty() || args.langs.iter().any(|l| l == iso))
        .map(|(cc, iso)| Job {
            code: (*iso).to_string(),
            url: format!("{base}/{cc}.txt.xz"),
            source: "cc100",
            decomp: Decomp::Xz,
            tsv_col: None,
        })
        .collect()
}

/// Build OpenSubtitles jobs from the fixed OPUS language list, mapped to 639-3.
fn opensubtitles_jobs(args: &Args) -> Vec<Job> {
    let base = args.opensubtitles_url.trim_end_matches('/');
    OPENSUB_LANGS
        .iter()
        .filter(|(_, iso)| args.langs.is_empty() || args.langs.iter().any(|l| l == iso))
        .map(|(os, iso)| Job {
            code: (*iso).to_string(),
            url: format!("{base}/{os}.txt.gz"),
            source: "opensubtitles",
            decomp: Decomp::Gzip,
            tsv_col: None,
        })
        .collect()
}

/// Parse the Tatoeba autoindex for 3-letter language directory names.
fn discover_tatoeba(agent: &ureq::Agent, base_url: &str) -> Result<Vec<String>> {
    let url = format!("{}/", base_url.trim_end_matches('/'));
    let html = agent
        .get(&url)
        .call()
        .with_context(|| format!("fetch index {url}"))?
        .into_string()?;

    let mut codes = std::collections::BTreeSet::new();
    for part in html.split("href=\"").skip(1) {
        if let Some(end) = part.find('"') {
            let code = part[..end].trim_end_matches('/');
            if code.len() == 3 && code.bytes().all(|b| b.is_ascii_lowercase()) {
                codes.insert(code.to_string());
            }
        }
    }
    Ok(codes.into_iter().collect())
}

/// Run one job with retries, then write train/test files.
fn run_job(agent: &ureq::Agent, args: &Args, job: &Job) -> Status {
    let mut last_err = String::new();
    for attempt in 0..args.retries {
        match download(agent, job, args.max_sentences) {
            Ok(sents) => {
                if sents.len() < args.min {
                    return Status::TooFew;
                }
                return match write_split(args, job, &sents) {
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

/// Stream-decompress a source and collect sentences, stopping at `max`.
fn download(agent: &ureq::Agent, job: &Job, max: usize) -> Result<Vec<String>> {
    let resp = agent.get(&job.url).call().map_err(|e| anyhow!("{e}"))?;
    let raw = resp.into_reader();
    let decoded: Box<dyn Read> = match job.decomp {
        Decomp::Bzip2 => Box::new(DecoderReader::new(raw)),
        Decomp::Xz => Box::new(xz2::read::XzDecoder::new(raw)),
        Decomp::Gzip => Box::new(flate2::read::GzDecoder::new(raw)),
    };
    let reader = BufReader::new(decoded);

    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let text = match job.tsv_col {
            Some(col) => line.split('\t').nth(col),
            None => Some(line.as_str()),
        };
        if let Some(text) = text {
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

fn write_split(args: &Args, job: &Job, sents: &[String]) -> Result<()> {
    let dir = args.out.join(&job.code);
    std::fs::create_dir_all(&dir)?;
    let file = dir.join(format!("{}.txt", job.source));
    if args.no_test {
        return write_lines(&file, sents);
    }
    let split = args.train.min(sents.len());
    write_lines(&file, &sents[..split])?;
    if sents.len() > args.train {
        write_lines(
            &args.test_out.join(format!("{}.txt", job.code)),
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

/// CC-100 language file code -> ISO 639-3 output code. Specific codes match
/// Tatoeba's (pes, kmr, lvs, zsm, nob, cmn) so the two sources merge.
#[rustfmt::skip]
const CC100_LANGS: &[(&str, &str)] = &[
    ("af", "afr"), ("am", "amh"), ("ar", "ara"), ("as", "asm"), ("az", "aze"),
    ("be", "bel"), ("bg", "bul"), ("bn", "ben"), ("br", "bre"), ("bs", "bos"),
    ("ca", "cat"), ("cs", "ces"), ("cy", "cym"), ("da", "dan"), ("de", "deu"),
    ("el", "ell"), ("en", "eng"), ("eo", "epo"), ("es", "spa"), ("et", "est"),
    ("eu", "eus"), ("fa", "pes"), ("ff", "ful"), ("fi", "fin"), ("fr", "fra"),
    ("fy", "fry"), ("ga", "gle"), ("gd", "gla"), ("gl", "glg"), ("gn", "grn"),
    ("gu", "guj"), ("ha", "hau"), ("he", "heb"), ("hi", "hin"), ("hr", "hrv"),
    ("ht", "hat"), ("hu", "hun"), ("hy", "hye"), ("id", "ind"), ("ig", "ibo"),
    ("is", "isl"), ("it", "ita"), ("ja", "jpn"), ("jv", "jav"), ("ka", "kat"),
    ("kk", "kaz"), ("km", "khm"), ("kn", "kan"), ("ko", "kor"), ("ku", "kmr"),
    ("ky", "kir"), ("la", "lat"), ("lg", "lug"), ("li", "lim"), ("ln", "lin"),
    ("lo", "lao"), ("lt", "lit"), ("lv", "lvs"), ("mg", "mlg"), ("mk", "mkd"),
    ("ml", "mal"), ("mn", "mon"), ("mr", "mar"), ("ms", "zsm"), ("my", "mya"),
    ("ne", "npi"), ("nl", "nld"), ("no", "nob"), ("ns", "nso"), ("om", "orm"),
    ("or", "ori"), ("pa", "pan"), ("pl", "pol"), ("ps", "pus"), ("pt", "por"),
    ("qu", "que"), ("rm", "roh"), ("ro", "ron"), ("ru", "rus"), ("sa", "san"),
    ("sc", "srd"), ("sd", "snd"), ("si", "sin"), ("sk", "slk"), ("sl", "slv"),
    ("so", "som"), ("sq", "sqi"), ("sr", "srp"), ("ss", "ssw"), ("su", "sun"),
    ("sv", "swe"), ("sw", "swh"), ("ta", "tam"), ("te", "tel"), ("th", "tha"),
    ("tl", "tgl"), ("tn", "tsn"), ("tr", "tur"), ("ug", "uig"), ("uk", "ukr"),
    ("ur", "urd"), ("uz", "uzb"), ("vi", "vie"), ("wo", "wol"), ("xh", "xho"),
    ("yi", "yid"), ("yo", "yor"), ("zh-Hans", "cmn"), ("zh-Hant", "cmn"),
    ("zu", "zul"),
];

/// OPUS OpenSubtitles file code -> ISO 639-3 output code. Regional variants
/// (pt/pt_br, zh_cn/zh_tw) fold into one 639-3 label; specifics match Tatoeba's
/// (pes, nob, zsm, cmn, lvs) so sources merge.
#[rustfmt::skip]
const OPENSUB_LANGS: &[(&str, &str)] = &[
    ("af", "afr"), ("ar", "ara"), ("bg", "bul"), ("bn", "ben"), ("br", "bre"),
    ("bs", "bos"), ("ca", "cat"), ("cs", "ces"), ("da", "dan"), ("de", "deu"),
    ("el", "ell"), ("en", "eng"), ("eo", "epo"), ("es", "spa"), ("et", "est"),
    ("eu", "eus"), ("fa", "pes"), ("fi", "fin"), ("fr", "fra"), ("gl", "glg"),
    ("he", "heb"), ("hi", "hin"), ("hr", "hrv"), ("hu", "hun"), ("hy", "hye"),
    ("id", "ind"), ("is", "isl"), ("it", "ita"), ("ja", "jpn"), ("ka", "kat"),
    ("kk", "kaz"), ("ko", "kor"), ("lt", "lit"), ("lv", "lvs"), ("mk", "mkd"),
    ("ml", "mal"), ("ms", "zsm"), ("nl", "nld"), ("no", "nob"), ("pl", "pol"),
    ("pt", "por"), ("pt_br", "por"), ("ro", "ron"), ("ru", "rus"), ("si", "sin"),
    ("sk", "slk"), ("sl", "slv"), ("sq", "sqi"), ("sr", "srp"), ("sv", "swe"),
    ("ta", "tam"), ("te", "tel"), ("th", "tha"), ("tl", "tgl"), ("tr", "tur"),
    ("uk", "ukr"), ("ur", "urd"), ("vi", "vie"), ("zh_cn", "cmn"), ("zh_tw", "cmn"),
];
