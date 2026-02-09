//! Chat Template Support for Multi-Turn Conversations
//!
//! This module provides:
//! - `ChatRole` and `ChatMessage` types for structured conversation history
//! - `ChatTemplate` with per-architecture prompt formatting
//! - Context window management via `trim_to_fit()`
//!
//! # Supported Formats
//! - **Mistral**: `[INST] ... [/INST]` format
//! - **LLaMA**: Llama 2/3 `[INST]` format with `<<SYS>>` blocks
//! - **Phi**: `<|user|>` / `<|assistant|>` / `<|end|>` format
//! - **Gemma**: `<start_of_turn>` / `<end_of_turn>` format
//! - **Qwen**: `<|im_start|>` / `<|im_end|>` format
//! - **GPT-OSS**: OpenAI Harmony format (ChatML-compatible)

use crate::model::architecture::ModelArchitecture;
use crate::tokenizer::Tokenizer;
use std::cell::Cell;
use std::io::Write;

/// Role of a message in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl std::fmt::Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatRole::System => write!(f, "system"),
            ChatRole::User => write!(f, "user"),
            ChatRole::Assistant => write!(f, "assistant"),
        }
    }
}

/// A single message in a conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

/// Chat template for formatting multi-turn conversations into model prompts.
///
/// Each variant knows how to format a conversation for its respective model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Mistral: `<s>[INST] {user} [/INST]{assistant}</s>[INST] {user} [/INST]`
    Mistral,
    /// LLaMA 2/3: `<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant} </s><s>[INST] {user} [/INST]`
    LLaMA,
    /// Phi-3: `<|user|>\n{user}<|end|>\n<|assistant|>\n{assistant}<|end|>\n`
    Phi,
    /// Gemma: `<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n{assistant}<end_of_turn>\n`
    Gemma,
    /// Qwen: `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n`
    Qwen,
    /// GPT-OSS: OpenAI Harmony format (ChatML-compatible for basic chat)
    /// Uses `<|im_start|>` / `<|im_end|>` with Harmony-style system prompt
    GptOss,
}

impl ChatTemplate {
    /// Select the appropriate chat template for a model architecture.
    pub fn for_architecture(arch: ModelArchitecture) -> Option<Self> {
        match arch {
            ModelArchitecture::Mistral => Some(ChatTemplate::Mistral),
            ModelArchitecture::LLaMA => Some(ChatTemplate::LLaMA),
            ModelArchitecture::Phi => Some(ChatTemplate::Phi),
            ModelArchitecture::Gemma => Some(ChatTemplate::Gemma),
            ModelArchitecture::Qwen => Some(ChatTemplate::Qwen),
            // DeepSeek uses a similar chat format to Qwen (ChatML-style)
            ModelArchitecture::DeepSeek => Some(ChatTemplate::Qwen),
            // GPT-OSS uses Harmony format (ChatML-compatible for basic chat)
            ModelArchitecture::GptOss => Some(ChatTemplate::GptOss),
            ModelArchitecture::Unknown => None,
        }
    }

    /// Returns the EOS token string(s) for this template.
    ///
    /// Generation should stop when any of these tokens are produced.
    pub fn eos_tokens(&self) -> &[&str] {
        match self {
            ChatTemplate::Mistral => &["</s>"],
            ChatTemplate::LLaMA => &["</s>"],
            ChatTemplate::Phi => &["<|end|>", "<|endoftext|>"],
            ChatTemplate::Gemma => &["<end_of_turn>"],
            ChatTemplate::Qwen => &["<|im_end|>", "<|endoftext|>"],
            ChatTemplate::GptOss => &["<|im_end|>", "<|endoftext|>"],
        }
    }

    /// Resolve EOS token strings to their token IDs using the model's tokenizer.
    ///
    /// Returns all EOS token IDs that were found in the vocabulary.
    pub fn eos_token_ids(&self, tokenizer: &Tokenizer) -> Vec<u32> {
        self.eos_tokens()
            .iter()
            .filter_map(|tok| tokenizer.token_id(tok))
            .collect()
    }

    /// Returns (open, close) thinking delimiter tokens, if this architecture
    /// has distilled thinking model variants.
    ///
    /// Qwen and LLaMA families have DeepSeek-R1 distilled variants that use
    /// `<think>`/`</think>` delimiters. Other architectures return `None`.
    pub fn thinking_tokens(&self) -> Option<(&str, &str)> {
        match self {
            ChatTemplate::Qwen => Some(("<think>", "</think>")),
            ChatTemplate::LLaMA => Some(("<think>", "</think>")),
            ChatTemplate::GptOss => Some(("<think>", "</think>")),
            _ => None, // Mistral/Phi/Gemma don't have distilled thinking variants
        }
    }

    /// Resolve thinking delimiter token IDs from the tokenizer vocabulary.
    ///
    /// Returns `None` if:
    /// - The architecture doesn't support thinking models, or
    /// - The tokens aren't in the vocabulary (i.e., it's a standard non-thinking model)
    ///
    /// This makes thinking support auto-detecting: if the vocab has `<think>` and
    /// `</think>`, it's a thinking model; otherwise it's treated as standard.
    pub fn thinking_token_ids(&self, tokenizer: &Tokenizer) -> Option<(u32, u32)> {
        let (open, close) = self.thinking_tokens()?;
        let open_id = tokenizer.token_id(open)?;
        let close_id = tokenizer.token_id(close)?;
        Some((open_id, close_id))
    }

    /// Format a conversation into a prompt string for the model.
    ///
    /// The returned string should be tokenized and fed to the model.
    /// It includes the trailing prompt for the assistant to continue generating.
    pub fn format_prompt(&self, messages: &[ChatMessage]) -> String {
        match self {
            ChatTemplate::Mistral => Self::format_mistral(messages),
            ChatTemplate::LLaMA => Self::format_llama(messages),
            ChatTemplate::Phi => Self::format_phi(messages),
            ChatTemplate::Gemma => Self::format_gemma(messages),
            ChatTemplate::Qwen => Self::format_qwen(messages),
            ChatTemplate::GptOss => Self::format_gpt_oss(messages),
        }
    }

    /// Trim conversation history to fit within a token budget.
    ///
    /// Drops the oldest user/assistant turn pairs (preserving any system prompt)
    /// until the formatted prompt fits within `max_tokens`.
    ///
    /// Returns `true` if messages were trimmed (meaning KV cache must be reset).
    pub fn trim_to_fit(
        &self,
        messages: &mut Vec<ChatMessage>,
        tokenizer: &Tokenizer,
        max_tokens: usize,
    ) -> bool {
        let original_len = messages.len();

        loop {
            let prompt = self.format_prompt(messages);
            let token_count = tokenizer.encode(&prompt).len();

            if token_count <= max_tokens {
                break;
            }

            // Find the first non-system message to remove
            if let Some(idx) = messages.iter().position(|m| m.role != ChatRole::System) {
                messages.remove(idx);
            } else {
                // Only system messages left -- nothing we can trim
                break;
            }
        }

        messages.len() != original_len
    }

    // =========================================================================
    // Per-architecture formatters
    // =========================================================================

    /// Mistral format:
    /// ```text
    /// <s>[INST] {system}\n\n{user} [/INST]{assistant}</s>[INST] {user} [/INST]
    /// ```
    fn format_mistral(messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        let mut system_prefix = String::new();
        let mut i = 0;

        // Extract system prompt if present
        if !messages.is_empty() && messages[0].role == ChatRole::System {
            system_prefix = format!("{}\n\n", messages[0].content);
            i = 1;
        }

        let mut first_user = true;
        while i < messages.len() {
            let msg = &messages[i];
            match msg.role {
                ChatRole::User => {
                    if first_user {
                        prompt
                            .push_str(&format!("[INST] {}{} [/INST]", system_prefix, msg.content));
                        first_user = false;
                    } else {
                        prompt.push_str(&format!("[INST] {} [/INST]", msg.content));
                    }
                }
                ChatRole::Assistant => {
                    prompt.push_str(&format!("{}</s>", msg.content));
                }
                ChatRole::System => {
                    // Additional system messages treated as user context
                    prompt.push_str(&format!("[INST] {} [/INST]", msg.content));
                }
            }
            i += 1;
        }

        prompt
    }

    /// LLaMA 2 format:
    /// ```text
    /// <s>[INST] <<SYS>>
    /// {system}
    /// <</SYS>>
    ///
    /// {user} [/INST] {assistant} </s><s>[INST] {user} [/INST]
    /// ```
    fn format_llama(messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        let mut system_text: Option<&str> = None;
        let mut i = 0;

        // Extract system prompt if present
        if !messages.is_empty() && messages[0].role == ChatRole::System {
            system_text = Some(&messages[0].content);
            i = 1;
        }

        let mut first_user = true;
        while i < messages.len() {
            let msg = &messages[i];
            match msg.role {
                ChatRole::User => {
                    if first_user {
                        if let Some(sys) = system_text {
                            prompt.push_str(&format!(
                                "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST] ",
                                sys, msg.content
                            ));
                        } else {
                            prompt.push_str(&format!("[INST] {} [/INST] ", msg.content));
                        }
                        first_user = false;
                    } else {
                        prompt.push_str(&format!("[INST] {} [/INST] ", msg.content));
                    }
                }
                ChatRole::Assistant => {
                    prompt.push_str(&format!("{} </s>", msg.content));
                }
                ChatRole::System => {
                    prompt.push_str(&format!("[INST] {} [/INST] ", msg.content));
                }
            }
            i += 1;
        }

        prompt
    }

    /// Phi-3 format:
    /// ```text
    /// <|system|>
    /// {system}<|end|>
    /// <|user|>
    /// {user}<|end|>
    /// <|assistant|>
    /// {assistant}<|end|>
    /// <|user|>
    /// {user}<|end|>
    /// <|assistant|>
    /// ```
    fn format_phi(messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            let role_tag = match msg.role {
                ChatRole::System => "<|system|>",
                ChatRole::User => "<|user|>",
                ChatRole::Assistant => "<|assistant|>",
            };
            prompt.push_str(&format!("{}\n{}<|end|>\n", role_tag, msg.content));
        }

        // Add the trailing assistant tag to prompt generation
        prompt.push_str("<|assistant|>\n");

        prompt
    }

    /// Gemma format:
    /// ```text
    /// <start_of_turn>user
    /// {user}<end_of_turn>
    /// <start_of_turn>model
    /// {assistant}<end_of_turn>
    /// <start_of_turn>user
    /// {user}<end_of_turn>
    /// <start_of_turn>model
    /// ```
    fn format_gemma(messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            let role_name = match msg.role {
                ChatRole::System => "user", // Gemma has no system role; prepend as user context
                ChatRole::User => "user",
                ChatRole::Assistant => "model",
            };
            prompt.push_str(&format!(
                "<start_of_turn>{}\n{}<end_of_turn>\n",
                role_name, msg.content
            ));
        }

        // Prompt the model to generate
        prompt.push_str("<start_of_turn>model\n");

        prompt
    }

    /// Qwen format:
    /// ```text
    /// <|im_start|>system
    /// {system}<|im_end|>
    /// <|im_start|>user
    /// {user}<|im_end|>
    /// <|im_start|>assistant
    /// {assistant}<|im_end|>
    /// <|im_start|>user
    /// {user}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    fn format_qwen(messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            let role_name = match msg.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            };
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                role_name, msg.content
            ));
        }

        // Prompt the assistant to generate
        prompt.push_str("<|im_start|>assistant\n");

        prompt
    }

    /// GPT-OSS format (Harmony / ChatML-compatible):
    /// ```text
    /// <|im_start|>system
    /// {system}<|im_end|>
    /// <|im_start|>user
    /// {user}<|im_end|>
    /// <|im_start|>assistant
    /// {assistant}<|im_end|>
    /// <|im_start|>user
    /// {user}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    fn format_gpt_oss(messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            let role_name = match msg.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            };
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                role_name, msg.content
            ));
        }

        // Prompt the assistant to generate
        prompt.push_str("<|im_start|>assistant\n");

        prompt
    }
}

// =============================================================================
// Thinking Model Support
// =============================================================================

/// Action to take when displaying a generated token in thinking-aware mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenAction {
    /// Regular token -- display normally.
    Display,
    /// Thinking content -- display dimmed (only when show_thinking is on).
    DisplayDim,
    /// Opening `<think>` delimiter -- display start marker (only when show_thinking is on).
    ThinkingStart,
    /// Closing `</think>` delimiter -- display end marker (only when show_thinking is on).
    ThinkingEnd,
    /// Token should be hidden (thinking content when show_thinking is off, or delimiters).
    Hide,
}

/// Tracks whether the model is currently generating inside a `<think>...</think>`
/// block and controls visibility of thinking content.
///
/// Uses `Cell` for interior mutability so it can be shared via `&` references
/// across closures without borrow-checker conflicts.
///
/// # Auto-detection
///
/// If constructed with `thinking_ids: None` (i.e., the model's vocabulary doesn't
/// contain `<think>`/`</think>` tokens), all tokens pass through as `TokenAction::Display`.
/// This means non-thinking models work without any special handling.
pub struct ThinkingState {
    inside_thinking: Cell<bool>,
    show_thinking: Cell<bool>,
    think_open_id: Option<u32>,
    think_close_id: Option<u32>,
}

impl ThinkingState {
    /// Create a new ThinkingState.
    ///
    /// `thinking_ids` should come from `ChatTemplate::thinking_token_ids()`.
    /// If `None`, this model has no thinking tokens and all output passes through.
    pub fn new(thinking_ids: Option<(u32, u32)>, show_thinking: bool) -> Self {
        let (open, close) = match thinking_ids {
            Some((o, c)) => (Some(o), Some(c)),
            None => (None, None),
        };
        Self {
            inside_thinking: Cell::new(false),
            show_thinking: Cell::new(show_thinking),
            think_open_id: open,
            think_close_id: close,
        }
    }

    /// Returns `true` if this model has thinking tokens in its vocabulary.
    pub fn is_thinking_model(&self) -> bool {
        self.think_open_id.is_some()
    }

    /// Returns `true` if currently inside a `<think>` block.
    pub fn inside(&self) -> bool {
        self.inside_thinking.get()
    }

    /// Returns `true` if thinking output should be shown.
    pub fn show(&self) -> bool {
        self.show_thinking.get()
    }

    /// Set whether thinking output should be shown.
    pub fn set_show(&self, show: bool) {
        self.show_thinking.set(show);
    }

    /// Toggle the show_thinking flag. Returns the new value.
    pub fn toggle_show(&self) -> bool {
        let new = !self.show_thinking.get();
        self.show_thinking.set(new);
        new
    }

    /// Reset thinking state for a new generation (e.g., new turn).
    pub fn reset(&self) {
        self.inside_thinking.set(false);
    }

    /// Process a generated token and return the display action.
    ///
    /// Call this for every token produced by the model during generation.
    /// The returned `TokenAction` tells the caller how to display (or hide) the token.
    pub fn process_token(&self, token_id: u32) -> TokenAction {
        // If no thinking tokens configured, everything passes through
        if self.think_open_id.is_none() {
            return TokenAction::Display;
        }

        if Some(token_id) == self.think_open_id {
            self.inside_thinking.set(true);
            if self.show_thinking.get() {
                TokenAction::ThinkingStart
            } else {
                TokenAction::Hide
            }
        } else if Some(token_id) == self.think_close_id {
            self.inside_thinking.set(false);
            if self.show_thinking.get() {
                TokenAction::ThinkingEnd
            } else {
                TokenAction::Hide
            }
        } else if self.inside_thinking.get() {
            if self.show_thinking.get() {
                TokenAction::DisplayDim
            } else {
                TokenAction::Hide
            }
        } else {
            TokenAction::Display
        }
    }
}

/// Strip `<think>...</think>` blocks from text.
///
/// Used to remove the thinking trace from assistant responses before storing
/// them in conversation history, so subsequent turns don't re-process the
/// thinking content (which wastes tokens and can confuse the model).
pub fn strip_thinking(text: &str) -> String {
    let mut result = String::new();
    let mut remaining = text;
    while let Some(start) = remaining.find("<think>") {
        result.push_str(&remaining[..start]);
        if let Some(end) = remaining[start..].find("</think>") {
            remaining = &remaining[start + end + "</think>".len()..];
        } else {
            // Unclosed <think> tag -- strip everything after it
            remaining = "";
        }
    }
    result.push_str(remaining);
    result.trim().to_string()
}

// =============================================================================
// Display helpers for thinking-aware token output
// =============================================================================

/// ANSI escape code to start dim text.
pub const ANSI_DIM: &str = "\x1b[2m";
/// ANSI escape code to reset all formatting.
pub const ANSI_RESET: &str = "\x1b[0m";

/// Display a token according to its `TokenAction`, using the provided decode function.
///
/// This helper encapsulates the ANSI formatting logic so callers don't need to
/// duplicate it. Pass a closure that decodes a token ID to its string representation.
pub fn display_thinking_token<F>(action: TokenAction, decode: F)
where
    F: FnOnce() -> String,
{
    match action {
        TokenAction::Display => {
            print!("{}", decode());
        }
        TokenAction::DisplayDim => {
            print!("{}{}{}", ANSI_DIM, decode(), ANSI_RESET);
        }
        TokenAction::ThinkingStart => {
            print!("\n{}[thinking] ", ANSI_DIM);
        }
        TokenAction::ThinkingEnd => {
            println!(" [/thinking]{}", ANSI_RESET);
        }
        TokenAction::Hide => {}
    }
}

/// Display a token to a specific writer according to its [`TokenAction`].
///
/// Like [`display_thinking_token`], but writes to the given [`Write`] sink instead of stdout.
pub fn display_thinking_token_to<F>(output: &mut dyn Write, action: TokenAction, decode: F)
where
    F: FnOnce() -> String,
{
    match action {
        TokenAction::Display => {
            let _ = write!(output, "{}", decode());
        }
        TokenAction::DisplayDim => {
            let _ = write!(output, "{}{}{}", ANSI_DIM, decode(), ANSI_RESET);
        }
        TokenAction::ThinkingStart => {
            let _ = write!(output, "\n{}[thinking] ", ANSI_DIM);
        }
        TokenAction::ThinkingEnd => {
            let _ = writeln!(output, " [/thinking]{}", ANSI_RESET);
        }
        TokenAction::Hide => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_simple() {
        let messages = vec![ChatMessage::user("Hello")];
        let prompt = ChatTemplate::Mistral.format_prompt(&messages);
        assert_eq!(prompt, "[INST] Hello [/INST]");
    }

    #[test]
    fn test_mistral_with_system() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let prompt = ChatTemplate::Mistral.format_prompt(&messages);
        assert_eq!(prompt, "[INST] You are helpful.\n\nHello [/INST]");
    }

    #[test]
    fn test_mistral_multi_turn() {
        let messages = vec![
            ChatMessage::user("Hi"),
            ChatMessage::assistant("Hello!"),
            ChatMessage::user("How are you?"),
        ];
        let prompt = ChatTemplate::Mistral.format_prompt(&messages);
        assert_eq!(
            prompt,
            "[INST] Hi [/INST]Hello!</s>[INST] How are you? [/INST]"
        );
    }

    #[test]
    fn test_llama_with_system() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let prompt = ChatTemplate::LLaMA.format_prompt(&messages);
        assert_eq!(
            prompt,
            "[INST] <<SYS>>\nYou are helpful.\n<</SYS>>\n\nHello [/INST] "
        );
    }

    #[test]
    fn test_phi_simple() {
        let messages = vec![ChatMessage::user("Hello")];
        let prompt = ChatTemplate::Phi.format_prompt(&messages);
        assert_eq!(prompt, "<|user|>\nHello<|end|>\n<|assistant|>\n");
    }

    #[test]
    fn test_gemma_simple() {
        let messages = vec![ChatMessage::user("Hello")];
        let prompt = ChatTemplate::Gemma.format_prompt(&messages);
        assert_eq!(
            prompt,
            "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn test_qwen_with_system() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let prompt = ChatTemplate::Qwen.format_prompt(&messages);
        assert_eq!(
            prompt,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_eos_tokens() {
        assert_eq!(ChatTemplate::Mistral.eos_tokens(), &["</s>"]);
        assert_eq!(
            ChatTemplate::Phi.eos_tokens(),
            &["<|end|>", "<|endoftext|>"]
        );
        assert_eq!(
            ChatTemplate::Qwen.eos_tokens(),
            &["<|im_end|>", "<|endoftext|>"]
        );
    }

    #[test]
    fn test_for_architecture() {
        assert_eq!(
            ChatTemplate::for_architecture(ModelArchitecture::Mistral),
            Some(ChatTemplate::Mistral)
        );
        assert_eq!(
            ChatTemplate::for_architecture(ModelArchitecture::Unknown),
            None
        );
    }

    // =========================================================================
    // Thinking model tests
    // =========================================================================

    #[test]
    fn test_thinking_tokens_qwen() {
        assert_eq!(
            ChatTemplate::Qwen.thinking_tokens(),
            Some(("<think>", "</think>"))
        );
    }

    #[test]
    fn test_thinking_tokens_llama() {
        assert_eq!(
            ChatTemplate::LLaMA.thinking_tokens(),
            Some(("<think>", "</think>"))
        );
    }

    #[test]
    fn test_thinking_tokens_none_for_mistral() {
        assert_eq!(ChatTemplate::Mistral.thinking_tokens(), None);
    }

    #[test]
    fn test_thinking_tokens_none_for_phi() {
        assert_eq!(ChatTemplate::Phi.thinking_tokens(), None);
    }

    #[test]
    fn test_thinking_tokens_none_for_gemma() {
        assert_eq!(ChatTemplate::Gemma.thinking_tokens(), None);
    }

    #[test]
    fn test_thinking_state_no_thinking_model() {
        let ts = ThinkingState::new(None, false);
        assert!(!ts.is_thinking_model());
        // All tokens pass through as Display
        assert_eq!(ts.process_token(42), TokenAction::Display);
        assert_eq!(ts.process_token(100), TokenAction::Display);
    }

    #[test]
    fn test_thinking_state_hide_thinking() {
        // Token IDs: 10 = <think>, 11 = </think>
        let ts = ThinkingState::new(Some((10, 11)), false);
        assert!(ts.is_thinking_model());

        // Regular token before thinking
        assert_eq!(ts.process_token(42), TokenAction::Display);

        // Enter thinking
        assert_eq!(ts.process_token(10), TokenAction::Hide);
        assert!(ts.inside());

        // Tokens inside thinking are hidden
        assert_eq!(ts.process_token(42), TokenAction::Hide);
        assert_eq!(ts.process_token(99), TokenAction::Hide);

        // Exit thinking
        assert_eq!(ts.process_token(11), TokenAction::Hide);
        assert!(!ts.inside());

        // Back to normal
        assert_eq!(ts.process_token(42), TokenAction::Display);
    }

    #[test]
    fn test_thinking_state_show_thinking() {
        let ts = ThinkingState::new(Some((10, 11)), true);

        // Regular token
        assert_eq!(ts.process_token(42), TokenAction::Display);

        // Enter thinking (shown as start marker)
        assert_eq!(ts.process_token(10), TokenAction::ThinkingStart);
        assert!(ts.inside());

        // Tokens inside thinking are displayed dim
        assert_eq!(ts.process_token(42), TokenAction::DisplayDim);

        // Exit thinking (shown as end marker)
        assert_eq!(ts.process_token(11), TokenAction::ThinkingEnd);
        assert!(!ts.inside());

        // Back to normal
        assert_eq!(ts.process_token(42), TokenAction::Display);
    }

    #[test]
    fn test_thinking_state_toggle() {
        let ts = ThinkingState::new(Some((10, 11)), false);
        assert!(!ts.show());

        let new = ts.toggle_show();
        assert!(new);
        assert!(ts.show());

        let new = ts.toggle_show();
        assert!(!new);
        assert!(!ts.show());
    }

    #[test]
    fn test_strip_thinking_basic() {
        let text = "<think>Let me reason about this...</think>The answer is 4.";
        assert_eq!(strip_thinking(text), "The answer is 4.");
    }

    #[test]
    fn test_strip_thinking_no_thinking() {
        let text = "The answer is 4.";
        assert_eq!(strip_thinking(text), "The answer is 4.");
    }

    #[test]
    fn test_strip_thinking_unclosed() {
        let text = "<think>Reasoning forever...";
        assert_eq!(strip_thinking(text), "");
    }

    #[test]
    fn test_strip_thinking_with_newlines() {
        let text = "<think>\nStep 1: think\nStep 2: verify\n</think>\n\nThe answer is 42.";
        assert_eq!(strip_thinking(text), "The answer is 42.");
    }
}
