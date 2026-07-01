//! Runtime detector: tokenize -> aggregate per-token posteriors -> mixture.

use crate::model::LangModel;
use crate::schema::{Detection, LangScore};
use crate::tokenizer::Encoder;
use anyhow::Result;
use std::path::Path;

/// Default second-language share needed to flag input as multilingual.
pub const DEFAULT_MULTILINGUAL_THRESHOLD: f32 = 0.15;

/// A loaded detector: a tokenizer plus its trained posterior table.
pub struct Detector {
    encoder: Encoder,
    model: LangModel,
    multilingual_threshold: f32,
}

impl Detector {
    /// Build from an already-loaded encoder and model.
    pub fn new(encoder: Encoder, model: LangModel) -> Self {
        Self {
            encoder,
            model,
            multilingual_threshold: DEFAULT_MULTILINGUAL_THRESHOLD,
        }
    }

    /// Load a tokenizer and model from disk.
    pub fn load(tokenizer_path: impl AsRef<Path>, model_path: impl AsRef<Path>) -> Result<Self> {
        let encoder = Encoder::from_file(tokenizer_path)?;
        let model = LangModel::load(model_path)?;
        Ok(Self::new(encoder, model))
    }

    /// Override the multilingual flag threshold (second-language share).
    pub fn with_multilingual_threshold(mut self, threshold: f32) -> Self {
        self.multilingual_threshold = threshold;
        self
    }

    /// Detect the language mixture of `text`.
    pub fn detect(&self, text: &str) -> Result<Detection> {
        let ids = self.encoder.encode(text)?;
        Ok(self.detect_ids(&ids))
    }

    /// Score pre-tokenized ids. Exposed for callers that tokenize once and
    /// reuse the ids, and for testing.
    pub fn detect_ids(&self, ids: &[u32]) -> Detection {
        score(&self.model, self.multilingual_threshold, ids)
    }
}

/// Aggregate per-token posteriors into a language mixture. Kept free-standing
/// so it can be tested without a real tokenizer.
fn score(model: &LangModel, multilingual_threshold: f32, ids: &[u32]) -> Detection {
    let mut acc = vec![0.0f32; model.langs.len()];
    let mut weight_sum = 0.0f32;
    let mut scored = 0usize;

    for &id in ids {
        let t = id as usize;
        // Guard: model may have been trained against a smaller vocab.
        if t >= model.token_weight.len() {
            continue;
        }
        let w = model.token_weight[t];
        let post = &model.token_post[t];
        if w <= 0.0 || post.is_empty() {
            continue;
        }
        weight_sum += w;
        scored += 1;
        for &(lang, p) in post {
            acc[lang as usize] += w * p;
        }
    }

    if weight_sum <= 0.0 {
        return Detection::undetermined();
    }

    let mut languages: Vec<LangScore> = acc
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, &v)| LangScore {
            lang: model.langs[i].clone(),
            score: v / weight_sum,
        })
        .collect();
    languages.sort_by(|a, b| b.score.total_cmp(&a.score));

    let top = languages[0].score;
    let second = languages.get(1).map_or(0.0, |s| s.score);

    Detection {
        language: languages[0].lang.clone(),
        confidence: top,
        margin: top - second,
        is_multilingual: second >= multilingual_threshold,
        languages,
        scored_tokens: scored,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LangModel, MODEL_VERSION};

    /// Toy model over vocab {1 => fr, 2 => en}; token 0 is noise.
    fn toy_model() -> LangModel {
        LangModel {
            version: MODEL_VERSION,
            langs: vec!["en".into(), "fr".into()],
            top_k: 1,
            vocab_size: 3,
            token_post: vec![
                vec![],         // 0: no signal
                vec![(1, 1.0)], // 1: pure fr
                vec![(0, 1.0)], // 2: pure en
            ],
            token_weight: vec![0.0, 1.0, 1.0],
        }
    }

    #[test]
    fn mixture_is_weighted_share() {
        // 8 fr tokens + 2 en tokens => 80/20.
        let ids = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2];
        let d = score(&toy_model(), DEFAULT_MULTILINGUAL_THRESHOLD, &ids);
        assert_eq!(d.language, "fr");
        assert!((d.confidence - 0.8).abs() < 1e-6, "conf {}", d.confidence);
        assert_eq!(d.scored_tokens, 10);
        assert!(d.is_multilingual);
        let en = d.languages.iter().find(|l| l.lang == "en").unwrap();
        assert!((en.score - 0.2).abs() < 1e-6, "en {}", en.score);
    }

    #[test]
    fn noise_only_is_undetermined() {
        let d = score(&toy_model(), DEFAULT_MULTILINGUAL_THRESHOLD, &[0, 0, 0]);
        assert_eq!(d.language, crate::schema::UNDETERMINED);
        assert_eq!(d.scored_tokens, 0);
        assert!(!d.is_multilingual);
    }

    #[test]
    fn monolingual_not_flagged() {
        let d = score(&toy_model(), DEFAULT_MULTILINGUAL_THRESHOLD, &[1, 1, 1, 1]);
        assert_eq!(d.language, "fr");
        assert!((d.confidence - 1.0).abs() < 1e-6);
        assert!(!d.is_multilingual);
    }
}
