//! The trained model: a compact token -> language posterior table.
//!
//! For each vocabulary token we store, at most, its top-K most likely
//! languages (`P(lang | token)`) plus a scalar discriminative weight. Unseen
//! or non-discriminative tokens carry an empty posterior and zero weight and
//! are skipped at detection time. Storing only top-K keeps the table at a few
//! MB instead of `vocab_size * num_langs` dense floats.

use anyhow::{bail, ensure, Context, Result};
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
    pub(crate) version: u32,
    /// Language codes; a `LangId` indexes into this.
    pub(crate) langs: Vec<String>,
    /// How many languages were kept per token at training time. Stored as
    /// metadata only — detection reads whatever each posterior slice contains.
    pub(crate) top_k: usize,
    /// Tokenizer vocab size the table was built against.
    pub(crate) vocab_size: usize,
    /// `token_id -> top-K posterior`. Length == `vocab_size`.
    pub(crate) token_post: Vec<Posterior>,
    /// `token_id -> discriminative weight` in `[0, 1]`. Length == `vocab_size`.
    pub(crate) token_weight: Vec<f32>,
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
        model
            .validate()
            .with_context(|| format!("invalid model {}", path.display()))?;
        Ok(model)
    }

    /// Check the structural invariants that detection relies on, so a truncated
    /// or malformed model fails at load instead of panicking mid-detection.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.token_post.len() == self.vocab_size,
            "token_post len {} != vocab_size {}",
            self.token_post.len(),
            self.vocab_size
        );
        ensure!(
            self.token_weight.len() == self.vocab_size,
            "token_weight len {} != vocab_size {}",
            self.token_weight.len(),
            self.vocab_size
        );
        let n = self.langs.len();
        for (t, post) in self.token_post.iter().enumerate() {
            for &(lang, _) in post {
                ensure!(
                    (lang as usize) < n,
                    "token {t}: LangId {lang} out of range (langs.len() = {n})"
                );
            }
        }
        Ok(())
    }

    /// Language codes the model can emit, indexed by `LangId`.
    pub fn languages(&self) -> &[String] {
        &self.langs
    }

    /// Tokenizer vocabulary size the table was built against.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Number of vocabulary tokens that carry any language signal.
    pub fn modeled_tokens(&self) -> usize {
        self.token_weight.iter().filter(|&&w| w > 0.0).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> LangModel {
        LangModel {
            version: MODEL_VERSION,
            langs: vec!["en".into(), "fr".into()],
            top_k: 1,
            vocab_size: 2,
            token_post: vec![vec![(0, 1.0)], vec![(1, 1.0)]],
            token_weight: vec![1.0, 1.0],
        }
    }

    #[test]
    fn validate_accepts_consistent_model() {
        assert!(base().validate().is_ok());
    }

    #[test]
    fn validate_rejects_out_of_range_langid() {
        let mut m = base();
        m.token_post[0] = vec![(5, 1.0)];
        assert!(m.validate().is_err());
    }

    #[test]
    fn validate_rejects_length_mismatch() {
        let mut m = base();
        m.token_weight.pop();
        assert!(m.validate().is_err());
    }
}
