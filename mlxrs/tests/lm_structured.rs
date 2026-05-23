//! Integration tests for `mlxrs::lm::structured` — port of
//! `mlx_vlm/structured.py`'s `LLGuidanceLogitsProcessor` +
//! `build_json_schema_logits_processor` (V6 / issue #180).
//!
//! Uses an inline byte-level HF tokenizer fixture (single-token ASCII
//! vocab: every printable byte gets its own id + a special token) so the
//! `toktrie_hf_tokenizers` `ByteLevel` decoder-detection path accepts the
//! tokenizer without needing the full WordLevel/SPM fixture (which the
//! crate rejects: it requires `ByteLevel` or `ByteFallback`).
//!
//! Test scope (per V6 spec):
//!
//! - `build_json_schema_logits_processor_constructs` — sanity-check
//!   construction with a simple object schema.
//! - `json_schema_processor_masks_invalid_first_tokens` — apply the
//!   processor on a fixture logits row; assert tokens that DON'T lead to
//!   a valid JSON start (an alphabetic char before `{`) are masked to
//!   `-inf`, while `{` (the only valid first character for the
//!   `{"type":"object"}` schema) remains finite.
//! - `llguidance_regex_grammar_constructs` —
//!   `GrammarSpec::Regex(r"^[0-9]+$")` constructor succeeds.
//! - `llguidance_lark_grammar_constructs` — minimal Lark grammar
//!   constructor succeeds.
//! - `llguidance_processor_implements_logits_processor_trait` — compile-
//!   check that `into_logits_processor` plugs into the
//!   `make_logits_processors` trait alias.

#![cfg(all(feature = "lm", feature = "llguidance"))]

use std::{fs, io::Write, path::PathBuf, process};

use mlxrs::{
  Array,
  lm::{generate::LogitsProcessor, structured},
};
use serde_json::json;

/// A minimal byte-level HF tokenizer (vocab = printable ASCII + a few
/// specials) accepted by `toktrie_hf_tokenizers::ByteTokenizer`:
/// `ByteLevel` pre-tokenizer + `ByteLevel` decoder (the two decoders the
/// crate's `check_decoder` recognizes), one-to-one byte→id model.
///
/// The vocab covers the 95 printable ASCII bytes (`0x20..=0x7E`), each
/// mapped via the `tokenizers::ByteLevel` byte→unicode char convention,
/// plus three special tokens (`<unk>`, `<s>`, `</s>`). All printable
/// bytes appear as their byte-level glyph (`tokenizers` automatically
/// maps non-printable + space using its standard byte→char table —
/// `Ġ` for `0x20` etc.).
fn build_byte_level_tokenizer_json() -> String {
  // The `tokenizers::ByteLevel` byte→unicode map: bytes `0x21..=0x7E`
  // map to themselves, byte `0x20` (space) maps to `Ġ` (0x120). Build
  // a minimal vocab covering exactly those + the 3 special tokens.
  let mut vocab_entries: Vec<String> = Vec::new();
  // Special tokens (ids 0..3); the byte-level adapter recognizes the
  // `<...>`-bracketed form as specials.
  vocab_entries.push("\"<unk>\": 0".to_string());
  vocab_entries.push("\"<s>\": 1".to_string());
  vocab_entries.push("\"</s>\": 2".to_string());
  // Printable ASCII via ByteLevel byte→char map.
  let mut next_id: u32 = 3;
  // Build the byte→char table the same way `tokenizers::ByteLevel`
  // does — `is_self_mapped` for `!..~` + `00A1..00AC` + `00AE..00FF`,
  // remap others starting at U+0100.
  let mut k: u32 = 0x100;
  let mut char_map: Vec<char> = Vec::with_capacity(256);
  for byte in 0..=255u8 {
    let c = byte as char;
    let mapped = match c {
      '!'..='~' => c,
      '\u{00A1}'..='\u{00AC}' => c,
      '\u{00AE}'..='\u{00FF}' => c,
      _ => {
        let m = char::from_u32(k).unwrap();
        k += 1;
        m
      }
    };
    char_map.push(mapped);
  }
  for byte in 0x20u8..=0x7Eu8 {
    let glyph = char_map[byte as usize];
    // JSON-escape the glyph as a Rust char literal.
    let escaped = match glyph {
      '"' => "\\\"".to_string(),
      '\\' => "\\\\".to_string(),
      c => c.to_string(),
    };
    vocab_entries.push(format!("\"{}\": {}", escaped, next_id));
    next_id += 1;
  }

  format!(
    r#"{{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        {{"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}},
        {{"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}},
        {{"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}}
      ],
      "normalizer": null,
      "pre_tokenizer": {{
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true
      }},
      "post_processor": null,
      "decoder": {{
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true
      }},
      "model": {{
        "type": "BPE",
        "dropout": null,
        "unk_token": "<unk>",
        "continuing_subword_prefix": null,
        "end_of_word_suffix": null,
        "fuse_unk": false,
        "vocab": {{
          {}
        }},
        "merges": []
      }}
    }}"#,
    vocab_entries.join(",\n      ")
  )
}

const TOKENIZER_CONFIG_JSON: &str = r#"{
  "bos_token": "<s>",
  "eos_token": "</s>",
  "unk_token": "<unk>",
  "model_max_length": 2048
}"#;

fn temp_dir(name: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!("mlxrs_lm_structured_{}_{}", process::id(), name));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

fn fixture_tokenizer(name: &str) -> mlxrs::tokenizer::Tokenizer {
  let dir = temp_dir(name);
  let tj_path = dir.join("tokenizer.json");
  let mut tj = fs::File::create(&tj_path).unwrap();
  tj.write_all(build_byte_level_tokenizer_json().as_bytes())
    .unwrap();
  let mut tc = fs::File::create(dir.join("tokenizer_config.json")).unwrap();
  tc.write_all(TOKENIZER_CONFIG_JSON.as_bytes()).unwrap();
  mlxrs::tokenizer::Tokenizer::from_path(&dir, None)
    .unwrap_or_else(|e| panic!("fixture tokenizer load failed: {e}"))
}

/// Look up the id for a single-byte token in our fixture vocab. The
/// printable-ASCII region starts at id 3.
fn id_for_byte(byte: u8) -> u32 {
  assert!(
    (0x20..=0x7E).contains(&byte),
    "byte {byte:#x} not in fixture vocab"
  );
  3 + (byte - 0x20) as u32
}

#[test]
fn build_json_schema_logits_processor_constructs() {
  let tok = fixture_tokenizer("build_json_schema_constructs");
  let schema = json!({
    "type": "object",
    "properties": {
      "name": { "type": "string" }
    }
  });
  let _proc = structured::build_json_schema_logits_processor(schema, &tok)
    .expect("processor construction should succeed for a simple schema");
}

#[test]
fn json_schema_processor_masks_invalid_first_tokens() {
  let tok = fixture_tokenizer("masks_invalid_first");
  // The simplest schema: any JSON object — only `{` (and optional
  // leading whitespace per the JSON grammar) can be the first token.
  let schema = json!({ "type": "object" });
  let proc = structured::build_json_schema_logits_processor(schema, &tok)
    .expect("processor construction should succeed");

  // Build a `[1, V]` logits row of zeros; the processor should mask
  // every token that can't start a valid JSON object to `-inf`.
  // Use the actual vocab size from the matcher's mask (after the
  // first apply call returns).
  let vocab = tok.hf().get_vocab_size(true);
  let zeros = vec![0.0f32; vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, vocab)).unwrap();

  let mut out = proc.apply(&[], &logits).expect("apply should succeed");
  let out_v = out.to_vec::<f32>().unwrap();
  assert_eq!(out_v.len(), vocab);

  // Token for `{` must be finite (allowed as the JSON object start).
  let open_brace = id_for_byte(b'{') as usize;
  assert!(
    out_v[open_brace].is_finite(),
    "`{{` token (id {open_brace}) must remain finite, got {}",
    out_v[open_brace]
  );

  // Token for `a` must be `-inf` (an alphabetic char cannot start a
  // JSON object — JSON grammar requires `{` or whitespace).
  let a = id_for_byte(b'a') as usize;
  assert!(
    out_v[a].is_infinite() && out_v[a] < 0.0,
    "`a` token (id {a}) must be masked to -inf, got {}",
    out_v[a]
  );

  // Token for `}` must also be masked (it can't be the FIRST character
  // of an object, only the LAST).
  let close_brace = id_for_byte(b'}') as usize;
  assert!(
    out_v[close_brace].is_infinite() && out_v[close_brace] < 0.0,
    "`}}` token (id {close_brace}) must be masked to -inf, got {}",
    out_v[close_brace]
  );
}

#[test]
fn llguidance_regex_grammar_constructs() {
  let tok = fixture_tokenizer("regex_constructs");
  let grammar = structured::GrammarSpec::Regex(r"[0-9]+".to_string());
  let _proc = structured::LLGuidanceLogitsProcessor::new(grammar, &tok)
    .expect("regex grammar processor construction should succeed");
}

#[test]
fn llguidance_lark_grammar_constructs() {
  let tok = fixture_tokenizer("lark_constructs");
  // A minimal Lark grammar: a single string of digits.
  let lark = r#"start: DIGITS
DIGITS: /[0-9]+/
"#;
  let grammar = structured::GrammarSpec::Lark(lark.to_string());
  let _proc = structured::LLGuidanceLogitsProcessor::new(grammar, &tok)
    .expect("lark grammar processor construction should succeed");
}

#[test]
fn llguidance_processor_implements_logits_processor_trait() {
  // Compile-time check: `into_logits_processor` returns the
  // `make_logits_processors` trait alias `Box<dyn Fn(&[u32], &Array)
  // -> Result<Array>>`; the binding's type pin enforces it.
  let tok = fixture_tokenizer("plug_into_chain");
  let proc = structured::build_json_schema_logits_processor(json!({"type": "object"}), &tok)
    .expect("processor construction should succeed");

  let boxed: LogitsProcessor = proc.into_logits_processor();
  // Exercise the boxed closure once to confirm the wiring round-trips.
  let vocab = tok.hf().get_vocab_size(true);
  let zeros = vec![0.0f32; vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, vocab)).unwrap();
  let _out = boxed(&[], &logits).expect("boxed closure call should succeed");
}
