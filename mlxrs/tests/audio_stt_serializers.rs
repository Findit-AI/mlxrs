//! Integration tests for the [`mlxrs::audio::stt::serializers`] surface
//! (issue #176, AUDIO-A13): transcript file writers (TXT / SRT / WebVTT /
//! JSON) and the `format_timestamp` / `format_vtt_timestamp` helpers.
//!
//! Hand-traced fixtures + byte-exact file-content asserts so a regression in
//! the python-port shape (1-based index, `,` vs `.` separator, trailing
//! blank line, JSON key order) is caught here rather than at downstream
//! tooling that string-matches the SRT/VTT/JSON output.
#![cfg(feature = "audio")]

use std::{collections::BTreeMap, fs, path::PathBuf, process};

use mlxrs::audio::stt::serializers::{
  Segment, Sentence, SentenceToken, Transcript, Word, save_as_json, save_as_srt, save_as_txt,
  save_as_vtt,
};

/// Process-scoped + named tempfile so parallel test binaries / cases never
/// collide. Mirrors the convention `tests/audio_stt.rs` uses.
fn temp_base(name: &str) -> PathBuf {
  let mut p = std::env::temp_dir();
  p.push(format!(
    "mlxrs_audio_stt_serializers_{}_{}",
    process::id(),
    name
  ));
  p
}

/// A 3-segment Whisper-style transcript fixture — three contiguous segments,
/// no word-level alignment, no speaker_id. Hand-traced timestamps so the
/// SRT/VTT/TXT byte-exact asserts are unambiguous.
fn fixture_3_segments() -> Transcript {
  Transcript::Segments {
    text: "hello world foo".into(),
    segments: vec![
      Segment {
        start: 0.0,
        end: 1.234,
        text: "hello".into(),
        words: None,
        speaker_id: None,
      },
      Segment {
        start: 1.234,
        end: 2.500,
        text: "world".into(),
        words: None,
        speaker_id: None,
      },
      Segment {
        start: 2.500,
        end: 4.000,
        text: "foo".into(),
        words: None,
        speaker_id: None,
      },
    ],
  }
}

#[test]
fn save_as_txt_writes_plain_lines() {
  // python `save_as_txt` (stt/generate.py:135-141) writes `segments.text`
  // verbatim to `f"{output_path}.txt"` — no per-segment newlines, no
  // trailing newline, no transformations. mlxrs mirrors that exactly.
  let base = temp_base("txt_plain");
  let t = fixture_3_segments();
  save_as_txt(&t, &base).unwrap();
  let txt_path = base.with_file_name(format!(
    "{}.txt",
    base.file_name().unwrap().to_string_lossy()
  ));
  let contents = fs::read_to_string(&txt_path).unwrap();
  assert_eq!(
    contents, "hello world foo",
    "save_as_txt writes `segments.text` verbatim (no per-segment lines, no trailing \\n)"
  );
  let _ = fs::remove_file(&txt_path);
}

#[test]
fn save_as_srt_writes_subrip_format() {
  // python `save_as_srt` per-cue format:
  //   {idx}\n{HH:MM:SS,mmm} --> {HH:MM:SS,mmm}\n{text}\n\n
  // Indexing 1-based; `,` separator (SRT spec).
  let base = temp_base("srt_format");
  let t = fixture_3_segments();
  save_as_srt(&t, &base).unwrap();
  let srt_path = base.with_file_name(format!(
    "{}.srt",
    base.file_name().unwrap().to_string_lossy()
  ));
  let contents = fs::read_to_string(&srt_path).unwrap();
  let expected = "1\n00:00:00,000 --> 00:00:01,234\nhello\n\n\
                  2\n00:00:01,234 --> 00:00:02,500\nworld\n\n\
                  3\n00:00:02,500 --> 00:00:04,000\nfoo\n\n";
  assert_eq!(
    contents, expected,
    "save_as_srt: 1-based index, `,` separator, double-newline cue separator"
  );
  let _ = fs::remove_file(&srt_path);
}

#[test]
fn save_as_vtt_writes_webvtt_format() {
  // python `save_as_vtt` per-cue format:
  //   WEBVTT\n\n{idx}\n{HH:MM:SS.mmm} --> {HH:MM:SS.mmm}\n{text}\n\n
  // Indexing 1-based; `.` separator (WebVTT spec); WEBVTT header required.
  let base = temp_base("vtt_format");
  let t = fixture_3_segments();
  save_as_vtt(&t, &base).unwrap();
  let vtt_path = base.with_file_name(format!(
    "{}.vtt",
    base.file_name().unwrap().to_string_lossy()
  ));
  let contents = fs::read_to_string(&vtt_path).unwrap();
  let expected = "WEBVTT\n\n\
                  1\n00:00:00.000 --> 00:00:01.234\nhello\n\n\
                  2\n00:00:01.234 --> 00:00:02.500\nworld\n\n\
                  3\n00:00:02.500 --> 00:00:04.000\nfoo\n\n";
  assert_eq!(
    contents, expected,
    "save_as_vtt: WEBVTT header, 1-based index, `.` separator, double-newline cue separator"
  );
  let _ = fs::remove_file(&vtt_path);
}

#[test]
fn save_as_json_round_trips_transcript() {
  // python `save_as_json` (stt/generate.py:173-225) writes a 2-space-indent
  // JSON tree we can round-trip via serde_json. The python Whisper shape is:
  //   {"text": ..., "segments": [{"text", "start", "end", "duration",
  //                               [optional "words"], [optional "speaker_id"]}, ...]}
  // duration is `end - start` (computed, not carried on Segment).
  let base = temp_base("json_roundtrip");
  let t = fixture_3_segments();
  save_as_json(&t, &base).unwrap();
  let json_path = base.with_file_name(format!(
    "{}.json",
    base.file_name().unwrap().to_string_lossy()
  ));
  let raw = fs::read_to_string(&json_path).unwrap();
  // Sanity: file is 2-space-indented (python `indent=2` matches
  // `serde_json::to_writer_pretty`'s 2-space default).
  assert!(raw.contains("\n  \"text\":"), "JSON uses 2-space indent");
  assert!(
    raw.contains("\"segments\""),
    "JSON top-level has `segments` key"
  );
  // Round-trip into a `Transcript` and assert structural equality. Note:
  // the JSON shape mlxrs writes includes a computed `duration` field per
  // segment that `Segment` doesn't carry — `Transcript::Deserialize` will
  // see + ignore it (serde_json default behavior for unknown fields).
  let parsed: Transcript = serde_json::from_str(&raw).expect("JSON parses back into Transcript");
  // Equality holds because `Segment` doesn't carry `duration` (and the
  // input had no `words` / `speaker_id`), so the round-trip is lossless
  // for the fields `Transcript` declares.
  assert_eq!(
    parsed, t,
    "JSON round-trips back to the original Transcript"
  );
  let _ = fs::remove_file(&json_path);
}

#[test]
fn save_as_json_sentence_shape_includes_tokens() {
  // python sentence branch (`hasattr(segments, "sentences")`) — output JSON
  // is `{"text": ..., "sentences": [{"text", "start", "end", "duration",
  //                                  "tokens": [{"text", "start", "end", "duration"}, ...],
  //                                  [optional "speaker_id"]}, ...]}`.
  let base = temp_base("json_sentence");
  let t = Transcript::Sentences {
    text: "hi".into(),
    sentences: vec![Sentence {
      text: "hi".into(),
      start: 0.0,
      end: 0.5,
      duration: 0.5,
      tokens: vec![SentenceToken {
        text: "h".into(),
        start: 0.0,
        end: 0.25,
        duration: 0.25,
      }],
      speaker_id: Some("spk_0".into()),
    }],
  };
  save_as_json(&t, &base).unwrap();
  let json_path = base.with_file_name(format!(
    "{}.json",
    base.file_name().unwrap().to_string_lossy()
  ));
  let raw = fs::read_to_string(&json_path).unwrap();
  // Sanity: the `sentences` top-level key + per-sentence `tokens` array +
  // `speaker_id` are all present.
  assert!(raw.contains("\"sentences\""));
  assert!(raw.contains("\"tokens\""));
  assert!(raw.contains("\"speaker_id\": \"spk_0\""));
  // Round-trip.
  let parsed: Transcript = serde_json::from_str(&raw).unwrap();
  assert_eq!(parsed, t);
  let _ = fs::remove_file(&json_path);
}

#[test]
fn save_as_json_segments_with_words_emits_words_array() {
  // python `seg["words"] = s["words"]` pass-through when the segment has
  // word-level alignment. mlxrs writes `{start, end, word}` per word; we
  // assert the JSON contains the words array exactly.
  let base = temp_base("json_words");
  let t = Transcript::Segments {
    text: "hi".into(),
    segments: vec![Segment {
      start: 0.0,
      end: 1.0,
      text: "hi".into(),
      words: Some(vec![Word {
        start: 0.0,
        end: 0.5,
        word: "hi".into(),
        extra: BTreeMap::new(),
      }]),
      speaker_id: None,
    }],
  };
  save_as_json(&t, &base).unwrap();
  let json_path = base.with_file_name(format!(
    "{}.json",
    base.file_name().unwrap().to_string_lossy()
  ));
  let raw = fs::read_to_string(&json_path).unwrap();
  assert!(raw.contains("\"words\""), "JSON includes words array");
  assert!(
    raw.contains("\"word\": \"hi\""),
    "JSON includes per-word entry"
  );
  // Round-trip.
  let parsed: Transcript = serde_json::from_str(&raw).unwrap();
  assert_eq!(parsed, t);
  let _ = fs::remove_file(&json_path);
}

#[test]
fn save_as_srt_emits_per_word_cues_when_words_present() {
  // python `_get_cues` emits one cue per segment THEN one cue per word —
  // mlxrs preserves that exactly. Verifies the segment-level cue comes
  // BEFORE the word-level cues (and indices are 1-based and contiguous).
  let base = temp_base("srt_with_words");
  let t = Transcript::Segments {
    text: "hi".into(),
    segments: vec![Segment {
      start: 0.0,
      end: 1.0,
      text: "hi there".into(),
      words: Some(vec![
        Word {
          start: 0.0,
          end: 0.5,
          word: "hi".into(),
          extra: BTreeMap::new(),
        },
        Word {
          start: 0.5,
          end: 1.0,
          word: "there".into(),
          extra: BTreeMap::new(),
        },
      ]),
      speaker_id: None,
    }],
  };
  save_as_srt(&t, &base).unwrap();
  let srt_path = base.with_file_name(format!(
    "{}.srt",
    base.file_name().unwrap().to_string_lossy()
  ));
  let contents = fs::read_to_string(&srt_path).unwrap();
  let expected = "1\n00:00:00,000 --> 00:00:01,000\nhi there\n\n\
                  2\n00:00:00,000 --> 00:00:00,500\nhi\n\n\
                  3\n00:00:00,500 --> 00:00:01,000\nthere\n\n";
  assert_eq!(contents, expected);
  let _ = fs::remove_file(&srt_path);
}

#[test]
fn save_as_txt_appends_extension_does_not_replace() {
  // Sanity: mlxrs faithful-port uses python's `f"{output_path}.txt"`
  // convention (append, not replace). `out.draft` becomes `out.draft.txt`,
  // NOT `out.txt`.
  let base = temp_base("ext_append.draft");
  let t = Transcript::Segments {
    text: "x".into(),
    segments: vec![],
  };
  save_as_txt(&t, &base).unwrap();
  let appended = base.with_file_name(format!(
    "{}.txt",
    base.file_name().unwrap().to_string_lossy()
  ));
  assert!(appended.exists(), "extension was appended, not replaced");
  let _ = fs::remove_file(&appended);
}
