//! Chat prompt formatting for different model instruction formats.
//!
//! | `fmt`             | Template                                                                  |
//! |-------------------|---------------------------------------------------------------------------|
//! | `chatml`          | ChatML (`<\|im_start\|>` / `<\|im_end\|>`)                                |
//! | `qwen2`           | Alias for `chatml`                                                        |
//! | `qwen3`           | Alias for `chatml`                                                        |
//! | `chatml-thinking` | ChatML + `<think>` prefix on assistant turn (Qwen3-Thinking variants)     |
//! | `llama3`          | LLaMA-3 instruct (`<\|begin_of_text\|>` / `<\|eot_id\|>`)                |
//! | `llama2`          | LLaMA-2 / Mistral `[INST]` format                                         |
//! | `llama`           | Alias for `llama2`                                                        |
//! | `phi3`            | Phi-3 instruct (`<\|user\|>` / `<\|assistant\|>` / `<\|end\|>`)          |
//! | `gemma`           | Gemma instruct (`<start_of_turn>` / `<end_of_turn>`)                     |
//! | `gemma2`          | Alias for `gemma`                                                         |
//! | `gemma3`          | Alias for `gemma`                                                         |
//! | _other_           | Plain-text `System/User/Assistant` fallback                               |

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
    /// ChatML with `<think>` appended to the assistant prefix.
    /// Used by Qwen3-Thinking variants whose chat template expects the model
    /// to start generating reasoning content immediately after `<think>`.
    ChatMLThinking,
    /// LLaMA-3 instruct format.
    LLaMA3,
    /// LLaMA-2 / Mistral `[INST]` format.
    LLaMA2,
    /// Phi-3 instruct format (`<|user|>` / `<|assistant|>` / `<|end|>`).
    Phi3,
    /// Gemma instruct format (`<start_of_turn>` / `<end_of_turn>`).
    /// Covers Gemma 1, 2, and 3; system prompt is folded into the first user turn.
    Gemma,
    /// Unknown / unsupported — plain-text `System/User/Assistant` fallback.
    Unknown,
}

impl std::str::FromStr for ChatFormat {
    // Parsing never fails — unknown strings map to `Unknown`.
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "chatml" | "qwen2" | "qwen3" => Self::ChatML,
            "chatml-thinking"            => Self::ChatMLThinking,
            "llama3"                     => Self::LLaMA3,
            "llama2" | "llama"           => Self::LLaMA2,
            "phi3"                       => Self::Phi3,
            "gemma" | "gemma2" | "gemma3" | "gemma3n" => Self::Gemma,
            _                            => Self::Unknown,
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
        ChatFormat::ChatML         => build_chatml(system, history, user_msg),
        ChatFormat::ChatMLThinking => build_chatml_thinking(system, history, user_msg),
        ChatFormat::LLaMA3         => build_llama3(system, history, user_msg),
        ChatFormat::LLaMA2         => build_llama2(system, history, user_msg),
        ChatFormat::Phi3           => build_phi3(system, history, user_msg),
        ChatFormat::Gemma          => build_gemma(system, history, user_msg),
        ChatFormat::Unknown        => build_plain(system, history, user_msg),
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

/// Return multi-token text-based stop strings for a given chat format.
///
/// These complement the token-ID stop list with pattern matching against the
/// accumulated response text.  They catch degenerate role-reversal loops that
/// poorly-aligned models may produce when they fail to emit their proper
/// end-of-turn token (e.g. `<|im_end|>`).
///
/// Only formats where models are known to produce plain-text role labels
/// (typically ChatML models that also have a LLaMA/SentencePiece tokenizer)
/// need non-empty lists here.  Models with robust control-token training
/// (LLaMA-3, Phi-3, Gemma) get an empty list — their token-ID stops suffice.
pub fn stop_text_strings_for_format(fmt: &str) -> &'static [&'static str] {
    let format: ChatFormat = fmt.parse().unwrap_or(ChatFormat::Unknown);
    match format {
        // ChatML models sometimes fall through to plain "User:" text when
        // the model hasn't learned to use <|im_end|> reliably.
        // For ChatMLThinking the same patterns apply, but the generation loop
        // suppresses them until </think> is seen (via SamplingParams::thinking_mode).
        //
        // We include both the colon-suffix form ("\nUser:") AND the bare form
        // with a trailing newline ("\nUser\n").  Qwen3 and other ChatML models
        // may emit the next turn without a colon (e.g. "\n\nUser\n") when
        // <|im_end|> is not correctly caught as a stop token.
        // The trailing-newline form avoids false-positives inside sentences
        // like "A user reported…".
        ChatFormat::ChatML | ChatFormat::ChatMLThinking | ChatFormat::Unknown => {
            &[
                "\n<|im_start|>",
                "\nUser:", "\nUser\n",
                "\nHuman:", "\nHuman\n",
                "\nAssistant:", "\nAssistant\n",
            ]
        }
        // LLaMA-2 uses [INST] delimiters; guard against the plain-text version too.
        ChatFormat::LLaMA2 => &["\nUser:", "\nUser\n", "\nHuman:", "\nHuman\n"],
        // LLaMA-3 and Phi-3 have robust single-token control stops — not needed.
        ChatFormat::LLaMA3 | ChatFormat::Phi3 => &[],
        // Gemma uses <end_of_turn> / <eos> as token stops.
        // Guard against role-reversal loops where a poorly-aligned or low-quant
        // Gemma emits "\n<start_of_turn>user" as plain text instead of the proper
        // control token.  "\nuser\n" and "\nmodel\n" were removed because they are
        // too broad — they match inside normal sentences ("The user reported...",
        // "The model architecture...") and cause premature truncation.
        ChatFormat::Gemma => &[
            "\n<start_of_turn>",
        ],
    }
}

/// Typed variant of [`stop_token_strings_for_format`].
pub fn stop_token_strings(format: ChatFormat) -> &'static [&'static str] {
    match format {
        // <|im_start|> is included because a well-trained ChatML model may
        // emit it to signal the beginning of the next turn; we must stop
        // there rather than letting it leak into the response.
        // ChatMLThinking uses the same stop tokens — </think> is NOT a stop
        // token because generation must continue after the think block closes.
        ChatFormat::ChatML | ChatFormat::ChatMLThinking => &["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        ChatFormat::LLaMA3  => &["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"],
        ChatFormat::LLaMA2  => &["</s>", "[INST]"],
        // <|end|> is the Phi-3 end-of-turn token; <|endoftext|> is the global EOS.
        ChatFormat::Phi3    => &["<|end|>", "<|endoftext|>"],
        // <end_of_turn> (token 107) marks the end of every Gemma turn.
        // <eos> / <end_of_text> (token 1) is the global EOS.
        // NOTE: <start_of_turn> (token 106) is intentionally NOT a stop token ID
        // during generation — adding it causes truncation after tool calls when
        // the model naturally wants to continue the conversation. It remains in
        // stop_text_strings_for_format as a text-level guard against role reversal.
        ChatFormat::Gemma   => &["<end_of_turn>", "<eos>", "<end_of_text>"],
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

/// ChatML with `<think>` appended to the assistant prefix.
///
/// Qwen3-Thinking models are trained to expect the prompt to end with
/// `<|im_start|>assistant\n<think>\n`.  They generate reasoning content
/// immediately (without emitting `<think>` themselves), then close with
/// `</think>` and produce the final response.
fn build_chatml_thinking(system: &str, history: &[(String, String)], user_msg: &str) -> String {
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
         <|im_start|>assistant\n<think>\n"
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

/// Phi-3 instruct format.
///
/// Correct structure:
/// ```text
/// <|system|>
/// {system}<|end|>
/// <|user|>
/// {user}<|end|>
/// <|assistant|>
/// {asst}<|end|>
/// ...
/// <|user|>
/// {user_msg}<|end|>
/// <|assistant|>
/// ```
/// When `system` is empty the system block is omitted entirely.
fn build_phi3(system: &str, history: &[(String, String)], user_msg: &str) -> String {
    let mut s = String::new();
    if !system.is_empty() {
        s.push_str(&format!("<|system|>\n{system}<|end|>\n"));
    }
    for (user, asst) in history {
        s.push_str(&format!(
            "<|user|>\n{user}<|end|>\n\
             <|assistant|>\n{asst}<|end|>\n"
        ));
    }
    s.push_str(&format!("<|user|>\n{user_msg}<|end|>\n<|assistant|>\n"));
    s
}

/// Gemma instruct format (covers Gemma 1, 2, and 3).
///
/// Correct structure:
/// ```text
/// <bos><start_of_turn>user
/// {system}
///
/// {user}<end_of_turn>
/// <start_of_turn>model
/// {asst}<end_of_turn>
/// ...
/// <start_of_turn>user
/// {user_msg}<end_of_turn>
/// <start_of_turn>model
/// ```
/// The system prompt (if any) is prepended to the first user message, separated
/// by a blank line, because Gemma's template does not have a dedicated system
/// role.  When `system` is empty the first user turn is emitted as-is.
fn build_gemma(system: &str, history: &[(String, String)], user_msg: &str) -> String {
    let mut s = String::from("<bos>");
    for (i, (user, asst)) in history.iter().enumerate() {
        // Fold system prompt into the very first user turn only
        let user_content = if i == 0 && !system.is_empty() {
            format!("{system}\n\n{user}")
        } else {
            user.clone()
        };
        s.push_str(&format!(
            "<start_of_turn>user\n{user_content}<end_of_turn>\n\
             <start_of_turn>model\n{asst}<end_of_turn>\n"
        ));
    }
    // Current (unanswered) turn — fold system only when there is no history
    let current_user = if history.is_empty() && !system.is_empty() {
        format!("{system}\n\n{user_msg}")
    } else {
        user_msg.to_string()
    };
    s.push_str(&format!(
        "<start_of_turn>user\n{current_user}<end_of_turn>\n\
         <start_of_turn>model\n"
    ));
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

    // ── Phi-3 tests ───────────────────────────────────────────────────────────

    #[test]
    fn phi3_format_no_history() {
        let out = build_chat_prompt("phi3", "Be helpful.", &[], "hello");
        assert!(out.contains("<|system|>\nBe helpful.<|end|>"));
        assert!(out.contains("<|user|>\nhello<|end|>"));
        assert!(out.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn phi3_format_no_system() {
        let out = build_chat_prompt("phi3", "", &[], "hi");
        assert!(!out.contains("<|system|>"), "system block must be absent");
        assert!(out.contains("<|user|>\nhi<|end|>"));
        assert!(out.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn phi3_format_with_history() {
        let history = vec![("first".to_string(), "reply".to_string())];
        let out = build_chat_prompt("phi3", "sys", &history, "second");
        assert!(out.contains("<|user|>\nfirst<|end|>"));
        assert!(out.contains("<|assistant|>\nreply<|end|>"));
        assert!(out.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn phi3_stop_tokens() {
        let stops = stop_token_strings_for_format("phi3");
        assert!(stops.contains(&"<|end|>"));
        assert!(stops.contains(&"<|endoftext|>"));
    }

    // ── Gemma tests ───────────────────────────────────────────────────────────

    #[test]
    fn gemma_format_no_history() {
        let out = build_chat_prompt("gemma", "Be helpful.", &[], "hello");
        assert!(out.starts_with("<bos>"));
        // System folded into first user turn
        assert!(out.contains("<start_of_turn>user\nBe helpful.\n\nhello<end_of_turn>"));
        assert!(out.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn gemma_format_no_system() {
        let out = build_chat_prompt("gemma", "", &[], "hi");
        assert!(out.starts_with("<bos>"));
        assert!(out.contains("<start_of_turn>user\nhi<end_of_turn>"));
        assert!(out.ends_with("<start_of_turn>model\n"));
        // No stray blank line when system is empty
        assert!(!out.contains("\n\nhi"));
    }

    #[test]
    fn gemma_format_with_history() {
        let history = vec![("q1".to_string(), "a1".to_string())];
        let out = build_chat_prompt("gemma", "sys", &history, "q2");
        // System folded into history's first user turn
        assert!(out.contains("<start_of_turn>user\nsys\n\nq1<end_of_turn>"));
        assert!(out.contains("<start_of_turn>model\na1<end_of_turn>"));
        // Current turn has no system prefix (history is non-empty)
        assert!(out.contains("<start_of_turn>user\nq2<end_of_turn>"));
        assert!(out.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn gemma_aliases() {
        let g  = build_chat_prompt("gemma",  "sys", &[], "hi");
        let g2 = build_chat_prompt("gemma2", "sys", &[], "hi");
        let g3 = build_chat_prompt("gemma3", "sys", &[], "hi");
        assert_eq!(g, g2);
        assert_eq!(g, g3);
    }

    #[test]
    fn gemma_stop_tokens() {
        let stops = stop_token_strings_for_format("gemma");
        assert!(stops.contains(&"<end_of_turn>"));
        assert!(stops.contains(&"<eos>"));
    }

    #[test]
    fn gemma_text_stop_strings_present() {
        let text_stops = stop_text_strings_for_format("gemma");
        // Must contain the role-reversal sentinel so degenerate loops are caught.
        assert!(
            text_stops.contains(&"\n<start_of_turn>"),
            "Gemma must stop on \\n<start_of_turn> to prevent role-reversal loops"
        );
        // "\nuser\n" and "\nmodel\n" were intentionally removed — they caused
        // false-positive truncation on normal text (e.g. "The user reported...",
        // "The model architecture...").  Only the control-token form is needed.
        assert!(!text_stops.contains(&"\nuser\n"),
            "\\nuser\\n must not be a stop string — too broad, causes false positives");
        assert!(!text_stops.contains(&"\nmodel\n"),
            "\\nmodel\\n must not be a stop string — too broad, causes false positives");
    }

    #[test]
    fn gemma_text_stops_aliases() {
        // gemma2 / gemma3 share the same text stops as gemma
        let g  = stop_text_strings_for_format("gemma");
        let g2 = stop_text_strings_for_format("gemma2");
        let g3 = stop_text_strings_for_format("gemma3");
        assert_eq!(g, g2);
        assert_eq!(g, g3);
    }

    // ── ChatMLThinking tests ──────────────────────────────────────────────────

    #[test]
    fn chatml_thinking_prompt_ends_with_think_tag() {
        let out = build_chat_prompt("chatml-thinking", "Be helpful.", &[], "hello");
        assert!(out.contains("<|im_start|>system\nBe helpful.<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn chatml_thinking_history_uses_plain_chatml_for_past_turns() {
        let history = vec![("hi".to_string(), "hello there".to_string())];
        let out = build_chat_prompt("chatml-thinking", "sys", &history, "how are you?");
        // Past assistant turns are stored without <think> prefix
        assert!(out.contains("<|im_start|>assistant\nhello there<|im_end|>"));
        // Only the new (unanswered) assistant turn gets <think>
        assert!(out.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn chatml_thinking_stop_tokens_same_as_chatml() {
        let thinking = stop_token_strings_for_format("chatml-thinking");
        let chatml = stop_token_strings_for_format("chatml");
        assert_eq!(thinking, chatml);
    }

    #[test]
    fn chatml_thinking_stop_text_strings_same_as_chatml() {
        let thinking = stop_text_strings_for_format("chatml-thinking");
        let chatml = stop_text_strings_for_format("chatml");
        assert_eq!(thinking, chatml);
    }
}
