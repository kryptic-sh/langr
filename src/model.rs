//! The trained model: a compact token -> language posterior table.
//!
//! For each vocabulary token we store, at most, its top-K most likely
//! languages (`P(lang | token)`) plus a scalar discriminative weight. Unseen
//! or non-discriminative tokens carry an empty posterior and zero weight and
//! are skipped at detection time. Storing only top-K keeps the table at a few
//! MB instead of `vocab_size * num_langs` dense floats.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
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
    /// How many languages were kept per token at training time. Stored as
    /// metadata only — detection reads whatever each posterior slice contains.
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
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let bytes = bincode::serialize(self).context("serialize model")?;
        std::fs::write(path, bytes).with_context(|| format!("write model {}", path.display()))
    }

    /// Load a model written by [`LangModel::save`].
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).with_context(|| format!("read model {}", path.display()))?;
        let model: LangModel = bincode::deserialize(&bytes)
            .with_context(|| format!("deserialize model {}", path.display()))?;
        if model.version != MODEL_VERSION {
            bail!(
                "model {} version {} != expected {MODEL_VERSION}",
                path.display(),
                model.version
            );
        }
        Ok(model)
    }
}
