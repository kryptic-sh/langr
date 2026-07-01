//! Thin wrapper over a HuggingFace `tokenizer.json` (e.g. Qwen3's byte-level
//! BPE vocab). We only ever load and encode — never train — so the heavy
//! trainer dependencies of the `tokenizers` crate are turned off.

use anyhow::{anyhow, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// Encodes text into subword token ids using a pretrained LLM vocab.
pub struct Encoder {
    inner: Tokenizer,
}

impl Encoder {
    /// Load a tokenizer from a HuggingFace `tokenizer.json`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = Tokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow!("load tokenizer {}: {e}", path.as_ref().display()))?;
        Ok(Self { inner })
    }

    /// Vocabulary size (including added/special tokens).
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Encode `text` into token ids, without special tokens.
    ///
    /// Uses `encode_fast`, which skips character-offset computation — we only
    /// need the ids, and offset mapping is a large part of `encode`'s cost.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode_fast(text, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }
}
