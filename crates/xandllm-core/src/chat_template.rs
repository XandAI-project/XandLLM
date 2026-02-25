//! Chat prompt formatting for different model instruction formats.
//!
//! | `fmt`    | Template                                               |
//! |----------|--------------------------------------------------------|
//! | `chatml` | ChatML (`<\|im_start\|>` / `<\|im_end\|>`)             |
//! | `qwen2`  | Alias for `chatml`                                     |
//! | `llama3` | LLaMA-3 instruct (`<\|begin_of_text\|>` / `<\|eot_id\|>`) |
//! | `llama2` | LLaMA-2 / Mistral `[INST]` format                      |
//! | `llama`  | Alias for `llama2`                                     |
//! | _other_  | Plain-text `System/User/Assistant` fallback            |

// ── Public enum ──────────────────────────────────────────────────────────────

/// Typed chat template selector.
///
/// Convert a raw format string (as returned by `AnyModel::chat_format()`) with
/// [`ChatFormat::from_str`].  All public formatting functions accept `&str` and
/// resolve to this enum internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    /// ChatML — used by Qwen2, OpenHermes, Nanbeige, etc.
    ChatML,
    /// LLaMA-3 instruct format.
    LLaMA3,
    /// LLaMA-2 / Mistral `[INST]` format.
    LLaMA2,
    /// Unknown / unsupported — plain-text `System/User/Assistant` fallback.
    Unknown,
}

impl std::str::FromStr for ChatFormat {
    // Parsing never fails — unknown strings map to `Unknown`.
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "chatml" | "qwen2" => Self::ChatML,
            "llama3" => Self::LLaMA3,
            "llama2" | "llama" => Self::LLaMA2,
            _ => Self::Unknown,
        })
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Format a single-turn prompt using the chat template for `fmt`.
///
/// This is a thin wrapper around [`build_chat_prompt`] with an empty history.
/// Use `build_chat_prompt` directly when you need multi-turn formatting.
pub fn format_prompt(fmt: &str, prompt: &str, system: &str) -> String {
    build_chat_prompt(fmt, system, &[], prompt)
}

/// Build a full multi-turn prompt from conversation history plus the current
/// user message, ready for direct tokenisation (`add_special_tokens: false`).
///
/// `fmt` is the value from `AnyModel::chat_format()` — see [`ChatFormat`] for
/// supported values.
///
/// `history` contains `(user_message, assistant_response)` pairs in
/// chronological order.  The returned string ends with the assistant's opening
/// tag so the model continues directly into its reply.
///
/// When `system` is empty the system header block is omitted entirely (no blank
/// `<|im_start|>system<|im_end|>` fragments, etc.).
pub fn build_chat_prompt(
    fmt: &str,
    system: &str,
    history: &[(String, String)],
    user_msg: &str,
) -> String {
    // `FromStr` for `ChatFormat` is infallible — unknown strings map to `Unknown`.
    let format: ChatFormat = fmt.parse().unwrap_or(ChatFormat::Unknown);
    build_chat_prompt_typed(format, system, history, user_msg)
}

/// Typed variant of [`build_chat_prompt`].
pub fn build_chat_prompt_typed(
    format: ChatFormat,
    system: &str,
    history: &[(String, String)],
    user_msg: &str,
) -> String {
    match format {
        ChatFormat::ChatML => build_chatml(system, history, user_msg),
        ChatFormat::LLaMA3 => build_llama3(system, history, user_msg),
        ChatFormat::LLaMA2 => build_llama2(system, history, user_msg),
        ChatFormat::Unknown => build_plain(system, history, user_msg),
    }
}

/// Return the correct end-of-turn stop token strings for a given chat format.
///
/// Using format-specific tokens is important: LLaMA-derived models (e.g.
/// Nanbeige) have `</s>` in their vocabulary as a sentence separator.  If
/// `</s>` is registered as a stop token for a ChatML model, generation will
/// terminate inside `<think>` blocks before `</think>` is ever emitted.
///
/// The caller should call `tokenizer.token_id(s)` for each returned string and
/// collect the `Some` results into `SamplingParams::stop_token_ids`.
pub fn stop_token_strings_for_format(fmt: &str) -> &'static [&'static str] {
    stop_token_strings(fmt.parse().unwrap_or(ChatFormat::Unknown))
}

/// Typed variant of [`stop_token_strings_for_format`].
pub fn stop_token_strings(format: ChatFormat) -> &'static [&'static str] {
    match format {
        // <|im_start|> is included because a well-trained ChatML model may
        // emit it to signal the beginning of the next turn; we must stop
        // there rather than letting it leak into the response.
        ChatFormat::ChatML => &["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        ChatFormat::LLaMA3 => &["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"],
        ChatFormat::LLaMA2 => &["</s>", "[INST]"],
        ChatFormat::Unknown => &["</s>", "<|endoftext|>"],
    }
}

// ── Private format builders ───────────────────────────────────────────────────

fn build_chatml(system: &str, history: &[(String, String)], user_msg: &str) -> String {
    let mut s = String::new();
    if !system.is_empty() {
        s.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
    }
    for (user, asst) in history {
        s.push_str(&format!(
            "<|im_start|>user\n{user}<|im_end|>\n\
             <|im_start|>assistant\n{asst}<|im_end|>\n"
        ));
    }
    s.push_str(&format!(
        "<|im_start|>user\n{user_msg}<|im_end|>\n\
         <|im_start|>assistant\n"
    ));
    s
}

fn build_llama3(system: &str, history: &[(String, String)], user_msg: &str) -> String {
    let mut s = String::from("<|begin_of_text|>");
    if !system.is_empty() {
        s.push_str(&format!(
            "<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
        ));
    }
    for (user, asst) in history {
        s.push_str(&format!(
            "<|start_header_id|>user<|end_header_id|>\n\
             {user}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\
             {asst}<|eot_id|>"
        ));
    }
    s.push_str(&format!(
        "<|start_header_id|>user<|end_header_id|>\n\
         {user_msg}<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n"
    ));
    s
}

/// LLaMA-2 / Mistral `[INST]` format.
///
/// Correct structure:
/// ```text
/// <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {asst} </s>
/// <s>[INST] {user} [/INST] {asst} </s>
/// ...
/// <s>[INST] {user_msg} [/INST]
/// ```
/// The current (incomplete) turn does **not** receive a closing `</s>`.
/// When `system` is empty the `<<SYS>>` block is omitted entirely.
fn build_llama2(system: &str, history: &[(String, String)], user_msg: &str) -> String {
    let mut s = String::new();
    for (i, (user, asst)) in history.iter().enumerate() {
        if i == 0 && !system.is_empty() {
            s.push_str(&format!(
                "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {asst} </s>\n"
            ));
        } else {
            s.push_str(&format!("<s>[INST] {user} [/INST] {asst} </s>\n"));
        }
    }
    // Current (unanswered) turn — no trailing </s>
    if history.is_empty() && !system.is_empty() {
        s.push_str(&format!(
            "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_msg} [/INST]"
        ));
    } else {
        s.push_str(&format!("<s>[INST] {user_msg} [/INST]"));
    }
    s
}

fn build_plain(system: &str, history: &[(String, String)], user_msg: &str) -> String {
    let mut s = String::new();
    if !system.is_empty() {
        s.push_str(&format!("System: {system}\n\n"));
    }
    for (user, asst) in history {
        s.push_str(&format!("User: {user}\nAssistant: {asst}\n\n"));
    }
    s.push_str(&format!("User: {user_msg}\nAssistant:"));
    s
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── original tests (updated for new behaviour where noted) ────────────────

    #[test]
    fn chatml_format() {
        let out = format_prompt("chatml", "hello", "You are a helpful assistant.");
        assert!(out.contains("<|im_start|>system"));
        assert!(out.contains("You are a helpful assistant."));
        assert!(out.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn qwen2_alias_for_chatml() {
        let a = format_prompt("chatml", "hello", "sys");
        let b = format_prompt("qwen2", "hello", "sys");
        assert_eq!(a, b);
    }

    #[test]
    fn llama2_wraps_in_inst() {
        let out = format_prompt("llama2", "hello", "Be helpful.");
        assert!(out.contains("[INST]"));
        assert!(out.contains("<<SYS>>"));
        assert!(out.contains("hello [/INST]"));
    }

    #[test]
    fn llama_alias_for_llama2() {
        let a = format_prompt("llama2", "hello", "sys");
        let b = format_prompt("llama", "hello", "sys");
        assert_eq!(a, b);
    }

    #[test]
    fn llama3_uses_header_ids() {
        let out = format_prompt("llama3", "hello", "Be helpful.");
        assert!(out.contains("<|begin_of_text|>"));
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(out.contains("hello<|eot_id|>"));
        assert!(out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n"));
    }

    #[test]
    fn unknown_format_plain_text_with_system() {
        // Unknown format now wraps in a plain System/User/Assistant block.
        let out = format_prompt("unknown_model", "hello", "Be helpful.");
        assert!(out.contains("System: Be helpful."));
        assert!(out.contains("User: hello"));
        assert!(out.ends_with("Assistant:"));
    }

    // ── new tests ─────────────────────────────────────────────────────────────

    #[test]
    fn build_chatml_no_history() {
        let via_build = build_chat_prompt("chatml", "sys", &[], "hello");
        let via_format = format_prompt("chatml", "hello", "sys");
        assert_eq!(via_build, via_format);
    }

    #[test]
    fn build_chatml_with_history() {
        let history = vec![("hi".to_string(), "hello there".to_string())];
        let out = build_chat_prompt("chatml", "sys", &history, "how are you?");
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>"));
        assert!(out.contains("<|im_start|>assistant\nhello there<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn build_llama2_first_turn() {
        let out = build_chat_prompt("llama2", "Be helpful.", &[], "hello");
        assert!(out.contains("<s>[INST]"));
        assert!(out.contains("<<SYS>>"));
        assert!(out.ends_with("[/INST]"));
        // Current turn must NOT have a trailing </s>
        assert!(!out.ends_with("</s>"));
    }

    #[test]
    fn build_llama2_multi_turn() {
        let history = vec![
            ("first".to_string(), "first reply".to_string()),
            ("second".to_string(), "second reply".to_string()),
        ];
        let out = build_chat_prompt("llama2", "sys", &history, "third");
        // Completed turns must end with </s>
        assert!(out.contains("first reply </s>"));
        assert!(out.contains("second reply </s>"));
        // Current (unanswered) turn must NOT end with </s>
        assert!(out.ends_with("[/INST]"));
        assert!(!out.ends_with("</s>"));
        // The old ~~ bug must be absent
        assert!(!out.contains("~~"));
    }

    #[test]
    fn build_llama3_with_history() {
        let history = vec![("q".to_string(), "a".to_string())];
        let out = build_chat_prompt("llama3", "sys", &history, "q2");
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>\nq<|eot_id|>"));
        assert!(out.contains(
            "<|start_header_id|>assistant<|end_header_id|>\na<|eot_id|>"
        ));
        assert!(out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n"));
    }

    #[test]
    fn empty_system_chatml() {
        let out = build_chat_prompt("chatml", "", &[], "hello");
        assert!(!out.contains("<|im_start|>system"), "system block must be absent");
        assert!(out.contains("<|im_start|>user\nhello"));
    }

    #[test]
    fn empty_system_llama2() {
        let out = build_chat_prompt("llama2", "", &[], "hello");
        assert!(!out.contains("<<SYS>>"), "SYS block must be absent");
        assert!(out.contains("[INST]"));
        assert!(out.contains("hello [/INST]"));
    }

    #[test]
    fn format_prompt_delegates() {
        // format_prompt must be a pure delegation — no match arms of its own.
        for fmt in ["chatml", "qwen2", "llama3", "llama2", "llama"] {
            let via_format = format_prompt(fmt, "prompt", "sys");
            let via_build = build_chat_prompt(fmt, "sys", &[], "prompt");
            assert_eq!(
                via_format, via_build,
                "format_prompt vs build_chat_prompt mismatch for fmt={fmt}"
            );
        }
    }

    #[test]
    fn stop_tokens_no_overlap() {
        let chatml = stop_token_strings_for_format("chatml");
        let llama2 = stop_token_strings_for_format("llama2");
        let llama3 = stop_token_strings_for_format("llama3");

        // ChatML-specific tokens must not bleed into the LLaMA lists.
        for tok in chatml {
            if *tok == "<|endoftext|>" {
                continue; // shared across families — overlap is intentional
            }
            assert!(
                !llama2.contains(tok) && !llama3.contains(tok),
                "ChatML token {tok:?} leaked into a LLaMA stop list"
            );
        }
        // LLaMA-2 </s> must never appear in the ChatML stop list.
        assert!(
            !chatml.contains(&"</s>"),
            "</s> must not be a ChatML stop token"
        );
    }
}
