//! The trained model: a compact token -> language posterior table.
//!
//! For each vocabulary token we store, at most, its top-K most likely
//! languages (`P(lang | token)`) plus a scalar discriminative weight. Unseen
//! or non-discriminative tokens carry an empty posterior and zero weight and
//! are skipped at detection time. Storing only top-K keeps the table at a few
//! MB instead of `vocab_size * num_langs` dense floats.

use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

/// Index into [`LangModel::langs`].
pub type LangId = u16;

/// Bump when the on-disk layout changes so stale models fail loudly.
pub const MODEL_VERSION: u32 = 1;

/// A token's top-K posterior: `(language index, P(lang | token))`.
pub type Posterior = Vec<(LangId, f32)>;

/// The serialized detector model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangModel {
    /// Format version guard.
    pub version: u32,
    /// Language codes; a `LangId` indexes into this.
    pub langs: Vec<String>,
    /// How many languages are kept per token.
    pub top_k: usize,
    /// Tokenizer vocab size the table was built against.
    pub vocab_size: usize,
    /// `token_id -> top-K posterior`. Length == `vocab_size`.
    pub token_post: Vec<Posterior>,
    /// `token_id -> discriminative weight` in `[0, 1]`. Length == `vocab_size`.
    pub token_weight: Vec<f32>,
}

impl LangModel {
    /// Serialize to a compact binary file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = bincode::serialize(self).map_err(to_io)?;
        std::fs::write(path, bytes)
    }

    /// Load a model written by [`LangModel::save`].
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let model: LangModel = bincode::deserialize(&bytes).map_err(to_io)?;
        if model.version != MODEL_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "model version {} != expected {MODEL_VERSION}",
                    model.version
                ),
            ));
        }
        Ok(model)
    }

    /// Look up a language code by id.
    pub fn lang(&self, id: LangId) -> &str {
        &self.langs[id as usize]
    }
}

fn to_io<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}
