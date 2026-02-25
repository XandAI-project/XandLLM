/// Streaming filter that suppresses `<think>…</think>` reasoning blocks.
///
/// Reasoning models (e.g. Nanbeige4.1, DeepSeek-R1) prefix their answer with
/// an internal chain-of-thought wrapped in `<think>` tags.  This filter sits
/// between the token stream and the terminal, buffering partial text to detect
/// the tags and only forwarding the text that lies outside them.
///
/// # Example
/// ```
/// let mut f = ThinkFilter::new();
/// assert_eq!(f.push("<think>internal</think> Hello!"), Some(" Hello!".into()));
/// ```
pub struct ThinkFilter {
    /// Text received but not yet emitted or discarded.
    buffer: String,
    /// Whether we are currently inside a `<think>` block.
    in_think: bool,
    /// Whether at least one `</think>` has been seen (used to trim leading
    /// whitespace/newlines from the answer that follows the reasoning block).
    seen_end: bool,
}

const OPEN_TAG: &str = "<think>";
const CLOSE_TAG: &str = "</think>";

impl ThinkFilter {
    pub fn new() -> Self {
        Self { buffer: String::new(), in_think: false, seen_end: false }
    }

    /// Push a new text fragment.  Returns any text that should be displayed,
    /// or `None` if the fragment was fully absorbed into a think block.
    pub fn push(&mut self, text: &str) -> Option<String> {
        self.buffer.push_str(text);
        self.process()
    }

    /// Flush remaining buffered text at end-of-stream.
    ///
    /// Text held back as a potential partial tag match is emitted when outside
    /// a think block.
    ///
    /// If the model was cut off mid-think (the `</think>` closing tag never
    /// arrived — e.g. because the token budget was exhausted), the partial
    /// thinking content is emitted rather than being silently discarded, so
    /// the user sees something rather than a blank response.
    pub fn flush(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        let s = std::mem::take(&mut self.buffer);
        if self.in_think {
            // Model ran out of tokens inside an unclosed think block.
            // Emit whatever reasoning was produced as a fallback.
            let s = s.trim_start_matches(['\n', '\r', ' ']).to_string();
            if s.is_empty() { None } else { Some(s) }
        } else {
            let s = self.trim_leading_if_needed(s);
            if s.is_empty() { None } else { Some(s) }
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn process(&mut self) -> Option<String> {
        let mut output = String::new();

        loop {
            if !self.in_think {
                if let Some(pos) = self.buffer.find(OPEN_TAG) {
                    // Emit everything before the opening tag.
                    output.push_str(&self.buffer[..pos]);
                    self.buffer = self.buffer[pos + OPEN_TAG.len()..].to_string();
                    self.in_think = true;
                } else {
                    // No opening tag yet.  Hold back enough bytes so a partial
                    // tag (split across push calls) can still be detected.
                    let hold = OPEN_TAG.len() - 1;
                    if self.buffer.len() > hold {
                        let safe = self.buffer.len() - hold;
                        output.push_str(&self.buffer[..safe]);
                        self.buffer = self.buffer[safe..].to_string();
                    }
                    break;
                }
            } else {
                // Inside a think block — scan for the closing tag.
                if let Some(pos) = self.buffer.find(CLOSE_TAG) {
                    self.buffer = self.buffer[pos + CLOSE_TAG.len()..].to_string();
                    self.in_think = false;
                    self.seen_end = true;
                } else {
                    // Discard think content; hold back enough bytes for a
                    // partial close-tag match.
                    let hold = CLOSE_TAG.len() - 1;
                    if self.buffer.len() > hold {
                        let discard_to = self.buffer.len() - hold;
                        self.buffer = self.buffer[discard_to..].to_string();
                    }
                    break;
                }
            }
        }

        if output.is_empty() {
            None
        } else {
            let output = self.trim_leading_if_needed(output);
            if output.is_empty() { None } else { Some(output) }
        }
    }

    /// After the first `</think>` block, trim leading whitespace/newlines from
    /// the very first visible text fragment so the answer doesn't start with a
    /// blank line.
    fn trim_leading_if_needed(&mut self, s: String) -> String {
        if self.seen_end {
            // One-shot: trim once, then stop trimming.
            self.seen_end = false;
            s.trim_start_matches(['\n', '\r', ' ']).to_string()
        } else {
            s
        }
    }
}

impl Default for ThinkFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(f: &mut ThinkFilter, s: &str) -> String {
        let mut out = String::new();
        if let Some(t) = f.push(s) { out.push_str(&t); }
        if let Some(t) = f.flush() { out.push_str(&t); }
        out
    }

    #[test]
    fn passthrough_when_no_think() {
        let mut f = ThinkFilter::new();
        assert_eq!(collect(&mut f, "Hello world"), "Hello world");
    }

    #[test]
    fn strips_single_think_block() {
        let mut f = ThinkFilter::new();
        let res = collect(&mut f, "<think>internal reasoning</think>Hello!");
        assert_eq!(res, "Hello!");
    }

    #[test]
    fn strips_multiple_think_blocks() {
        let mut f = ThinkFilter::new();
        let input = "<think>step 1</think>Answer: <think>step 2</think>Done.";
        let res = collect(&mut f, input);
        assert_eq!(res, "Answer: Done.");
    }

    #[test]
    fn handles_split_across_pushes() {
        let mut f = ThinkFilter::new();
        // Split the closing tag across two push calls.
        let mut out = String::new();
        if let Some(t) = f.push("<think>reason</") { out.push_str(&t); }
        if let Some(t) = f.push("think>Answer") { out.push_str(&t); }
        if let Some(t) = f.flush() { out.push_str(&t); }
        assert_eq!(out, "Answer");
    }

    #[test]
    fn unclosed_think_emitted_as_fallback() {
        // When the model is cut off before </think>, the partial thinking
        // content is shown rather than being silently discarded.
        let mut f = ThinkFilter::new();
        let mut out = String::new();
        if let Some(t) = f.push("<think>never closed") { out.push_str(&t); }
        if let Some(t) = f.flush() { out.push_str(&t); }
        assert_eq!(out, "never closed");
    }
}
