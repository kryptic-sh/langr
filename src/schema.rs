//! Public output schema returned by the detector.

use serde::{Deserialize, Serialize};

/// Language code used when nothing scores (empty / all-unknown input).
pub const UNDETERMINED: &str = "und";

/// Result of detecting the language(s) of an input string.
///
/// `languages` is the full sorted mixture; the top-level fields are
/// convenience projections of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    /// Best-guess language code (`"und"` when nothing scored).
    pub language: String,
    /// Weighted share of the winning language, in `[0, 1]`.
    pub confidence: f32,
    /// Gap between the top and second language shares; a reliability signal
    /// (large margin = confident single language).
    pub margin: f32,
    /// True when a second language holds a meaningful share
    /// (see [`crate::Detector::with_multilingual_threshold`]).
    pub is_multilingual: bool,
    /// Full language mixture, sorted by score descending.
    pub languages: Vec<LangScore>,
    /// Number of input tokens that carried model signal.
    pub scored_tokens: usize,
}

impl Detection {
    /// The empty/unknown result.
    pub fn undetermined() -> Self {
        Self {
            language: UNDETERMINED.to_string(),
            confidence: 0.0,
            margin: 0.0,
            is_multilingual: false,
            languages: Vec::new(),
            scored_tokens: 0,
        }
    }
}

/// A single language and its share of the input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LangScore {
    /// Language code.
    pub lang: String,
    /// Share of the weighted tokens attributed to this language, in `[0, 1]`.
    pub score: f32,
}
