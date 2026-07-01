//! `langr` — a fast, low-memory, single-purpose language detector.
//!
//! It borrows an LLM's subword vocabulary (a HuggingFace `tokenizer.json`,
//! e.g. Qwen3's byte-level BPE) to turn text into token ids, then aggregates a
//! trained `token -> P(language)` table into a language mixture. Long text at
//! high speed with a few MB of model — no neural net at inference time.
//!
//! ```no_run
//! use langr::Detector;
//!
//! let detector = Detector::load("tokenizer.json", "model.bin")?;
//! let result = detector.detect("Bonjour le monde, this is a test.")?;
//! println!("{} ({:.0}%)", result.language, result.confidence * 100.0);
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod detect;
pub mod model;
pub mod schema;
pub mod tokenizer;
pub mod train;

pub use detect::Detector;
pub use model::LangModel;
pub use schema::{Detection, LangScore};
pub use tokenizer::Encoder;
pub use train::{train, TrainConfig};
