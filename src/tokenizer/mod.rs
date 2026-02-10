use std::collections::HashMap;

#[cfg(test)]
mod tests;

const METASPACE: &str = "▁";
const BOS_TOKEN: &str = "<s>";
const EOS_TOKEN: &str = "</s>";

/// How byte-level tokens are encoded in the vocabulary.
#[derive(Clone, Debug, PartialEq)]
enum ByteEncoding {
    /// No byte-level encoding. Token strings are direct UTF-8.
    None,
    /// SentencePiece `<0xXX>` byte-fallback tokens.
    HexFallback,
    /// GPT-2 byte-to-unicode mapping (each byte maps to a specific Unicode char).
    Gpt2 { unicode_to_byte: HashMap<char, u8> },
}

/// Build the GPT-2 byte-to-unicode mapping table.
///
/// Mirrors Python's `bytes_to_unicode()` from the original GPT-2 tokenizer.
/// Each byte value (0-255) maps to a unique printable Unicode character.
/// Bytes in printable ASCII and Latin-1 ranges map to themselves (identity);
/// the remaining 68 bytes are shifted to the U+0100–U+0143 range.
fn build_gpt2_byte_to_unicode() -> [char; 256] {
    let mut identity = [false; 256];
    for b in 0x21u16..=0x7Eu16 {
        identity[b as usize] = true;
    } // ! through ~
    for b in 0xA1u16..=0xACu16 {
        identity[b as usize] = true;
    } // ¡ through ¬
    for b in 0xAEu16..=0xFFu16 {
        identity[b as usize] = true;
    } // ® through ÿ

    let mut table = ['\0'; 256];
    let mut n: u32 = 0;
    for b in 0u16..=255u16 {
        if identity[b as usize] {
            table[b as usize] = char::from_u32(b as u32).unwrap();
        } else {
            table[b as usize] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    table
}

/// Build the reverse GPT-2 mapping: Unicode character → byte value.
fn build_gpt2_unicode_to_byte() -> HashMap<char, u8> {
    let table = build_gpt2_byte_to_unicode();
    let mut map = HashMap::with_capacity(256);
    for (byte_val, &ch) in table.iter().enumerate() {
        map.insert(ch, byte_val as u8);
    }
    map
}

#[derive(Clone)]
pub struct Tokenizer {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
    // Maps packed (left, right) pair to its rank in merge order
    merge_to_rank: HashMap<u64, usize>,
    // Maps packed (left, right) pair to the merged token ID
    merge_to_id: HashMap<u64, u32>,
    // Explicit EOS token IDs (e.g. from GGUF tokenizer.ggml.eos_token_id)
    explicit_eos_ids: Vec<u32>,
    // How byte-level tokens are stored in the vocabulary
    byte_encoding: ByteEncoding,
}

impl Tokenizer {
    pub fn new(vocab: HashMap<String, u32>, merges: Vec<String>) -> Self {
        let vocab_size = vocab.len();
        let mut id_to_token = vec![String::new(); vocab_size];

        for (token, id) in &vocab {
            id_to_token[*id as usize] = token.clone();
        }

        let mut merge_to_rank = HashMap::new();
        let mut merge_to_id = HashMap::new();

        for (rank, merge) in merges.iter().enumerate() {
            let parts: Vec<&str> = merge.split_whitespace().collect();
            if parts.len() != 2 {
                continue;
            }

            let token1 = parts[0];
            let token2 = parts[1];

            if let (Some(&id1), Some(&id2)) = (vocab.get(token1), vocab.get(token2)) {
                let packed = Self::pack(id1, id2);
                merge_to_rank.insert(packed, rank);

                // The merged token is the concatenation of the two tokens
                let merged_token = format!("{}{}", token1, token2);
                if let Some(&merged_id) = vocab.get(&merged_token) {
                    merge_to_id.insert(packed, merged_id);
                }
            }
        }

        // Auto-detect byte encoding style from the vocabulary
        let byte_encoding = Self::detect_byte_encoding(&vocab);

        Self {
            token_to_id: vocab,
            id_to_token,
            merge_to_rank,
            merge_to_id,
            explicit_eos_ids: Vec::new(),
            byte_encoding,
        }
    }

    /// Detect the byte encoding style of the vocabulary.
    ///
    /// - If `<0x41>` is present → SentencePiece hex-fallback
    /// - If any token contains GPT-2 shifted characters (U+0100–U+0143) → GPT-2
    /// - Otherwise → no special byte encoding
    fn detect_byte_encoding(vocab: &HashMap<String, u32>) -> ByteEncoding {
        // Check for SentencePiece hex fallback: <0x41> is byte 0x41 = 'A'
        if vocab.contains_key("<0x41>") {
            return ByteEncoding::HexFallback;
        }

        // Check for GPT-2 encoding by looking for characters in the shifted range
        // (U+0100–U+0143). These only appear in GPT-2-style byte-encoded vocabularies.
        for token_str in vocab.keys() {
            for ch in token_str.chars() {
                if ('\u{0100}'..='\u{0143}').contains(&ch) {
                    return ByteEncoding::Gpt2 {
                        unicode_to_byte: build_gpt2_unicode_to_byte(),
                    };
                }
            }
        }

        ByteEncoding::None
    }

    /// Set explicit EOS token IDs (e.g. from GGUF metadata).
    ///
    /// These are used as a fallback when string-based EOS lookup fails.
    pub fn set_explicit_eos_ids(&mut self, ids: Vec<u32>) {
        self.explicit_eos_ids = ids;
    }

    /// Get explicitly configured EOS token IDs.
    pub fn explicit_eos_ids(&self) -> &[u32] {
        &self.explicit_eos_ids
    }

    /// Pack two u32 token IDs into a u64
    fn pack(left: u32, right: u32) -> u64 {
        ((left as u64) << 32) | (right as u64)
    }

    /// Unpack u64 into two u32 token IDs
    fn unpack(packed: u64) -> (u32, u32) {
        let left = (packed >> 32) as u32;
        let right = (packed & 0xFFFFFFFF) as u32;
        (left, right)
    }

    /// Get token ID(s) for a string, using byte fallback if not in vocab
    fn get_id(&self, token: &str) -> Vec<u32> {
        if let Some(&id) = self.token_to_id.get(token) {
            return vec![id];
        }

        // Byte fallback - encode each byte as <0xXX>
        token
            .bytes()
            .filter_map(|b| {
                let byte_token = format!("<0x{:02X}>", b);
                self.token_to_id.get(&byte_token).copied()
            })
            .collect()
    }

    /// Apply Metaspace pre-tokenization: replace spaces with ▁ and prepend ▁
    fn pre_tokenize_mistral(&self, text: &str) -> String {
        let mut result = String::from(METASPACE);
        for c in text.chars() {
            if c == ' ' {
                result.push_str(METASPACE);
            } else {
                result.push(c);
            }
        }
        result
    }

    /// Find the lowest-rank merge pair in the token sequence
    fn get_lowest_pair(&self, tokens: &[u32]) -> Option<u64> {
        let mut lowest_rank = usize::MAX;
        let mut result = None;

        for i in 0..tokens.len().saturating_sub(1) {
            let packed = Self::pack(tokens[i], tokens[i + 1]);
            if let Some(&rank) = self.merge_to_rank.get(&packed) {
                if rank < lowest_rank {
                    lowest_rank = rank;
                    result = Some(packed);
                }
            }
        }

        result
    }

    /// Merge all occurrences of (left, right) pair into merged token
    fn merge(&self, tokens: &[u32], left: u32, right: u32, merged: u32) -> Vec<u32> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == left && tokens[i + 1] == right {
                result.push(merged);
                i += 2;
            } else {
                result.push(tokens[i]);
                i += 1;
            }
        }

        result
    }

    /// Encode text into token IDs using BPE
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Pre-tokenization
        let text = self.pre_tokenize_mistral(text);

        // Convert UTF-8 text to initial tokens
        let mut tokens = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            let b0 = bytes[i];

            // Determine UTF-8 character length
            let len = if (b0 & 0x80) == 0x00 {
                1 // 0xxxxxxx
            } else if (b0 & 0xE0) == 0xC0 {
                2 // 110xxxxx
            } else if (b0 & 0xF0) == 0xE0 {
                3 // 1110xxxx
            } else if (b0 & 0xF8) == 0xF0 {
                4 // 11110xxx
            } else {
                1 // Invalid, treat as single byte
            };

            let char_str = std::str::from_utf8(&bytes[i..i + len]).unwrap_or_else(|_| {
                // Fallback to single byte if invalid UTF-8
                std::str::from_utf8(&bytes[i..i + 1]).unwrap_or("")
            });

            let ids = self.get_id(char_str);
            tokens.extend(ids);
            i += len;
        }

        // Apply BPE merges
        while let Some(packed) = self.get_lowest_pair(&tokens) {
            let (left, right) = Self::unpack(packed);
            if let Some(&merged) = self.merge_to_id.get(&packed) {
                tokens = self.merge(&tokens, left, right, merged);
            } else {
                break;
            }
        }

        // Add BOS token at the beginning
        if let Some(&bos_id) = self.token_to_id.get(BOS_TOKEN) {
            tokens.insert(0, bos_id);
        }

        tokens
    }

    /// Post-process Mistral decoding: replace ▁ with spaces
    fn decode_mistral(&self, s: &str) -> String {
        s.replace(METASPACE, " ")
    }

    /// Look up the token ID for a special token string (e.g. `</s>`, `<|end|>`).
    ///
    /// Returns `None` if the token is not in the vocabulary.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get the EOS (`</s>`) token ID, if present in the vocabulary.
    pub fn eos_id(&self) -> Option<u32> {
        self.token_to_id.get(EOS_TOKEN).copied()
    }

    /// Get the BOS (`<s>`) token ID, if present in the vocabulary.
    pub fn bos_id(&self) -> Option<u32> {
        self.token_to_id.get(BOS_TOKEN).copied()
    }

    /// Encode text into token IDs using BPE, without prepending BOS.
    ///
    /// Use this when the chat template already handles BOS positioning.
    pub fn encode_no_bos(&self, text: &str) -> Vec<u32> {
        // Pre-tokenization
        let text = self.pre_tokenize_mistral(text);

        // Convert UTF-8 text to initial tokens
        let mut tokens = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            let b0 = bytes[i];

            let len = if (b0 & 0x80) == 0x00 {
                1
            } else if (b0 & 0xE0) == 0xC0 {
                2
            } else if (b0 & 0xF0) == 0xE0 {
                3
            } else if (b0 & 0xF8) == 0xF0 {
                4
            } else {
                1
            };

            let char_str = std::str::from_utf8(&bytes[i..i + len])
                .unwrap_or_else(|_| std::str::from_utf8(&bytes[i..i + 1]).unwrap_or(""));

            let ids = self.get_id(char_str);
            tokens.extend(ids);
            i += len;
        }

        // Apply BPE merges
        while let Some(packed) = self.get_lowest_pair(&tokens) {
            let (left, right) = Self::unpack(packed);
            if let Some(&merged) = self.merge_to_id.get(&packed) {
                tokens = self.merge(&tokens, left, right, merged);
            } else {
                break;
            }
        }

        tokens
    }

    /// Convert a token's vocabulary string to raw bytes.
    ///
    /// - **GPT-2**: each character is reverse-mapped through the GPT-2 byte table.
    /// - **HexFallback**: `<0xXX>` tokens are parsed to their single-byte value.
    /// - **None**: the token's UTF-8 bytes are returned directly.
    fn token_str_to_bytes(&self, token_str: &str) -> Vec<u8> {
        match &self.byte_encoding {
            ByteEncoding::Gpt2 { unicode_to_byte } => {
                let mut bytes = Vec::with_capacity(token_str.len());
                for c in token_str.chars() {
                    if let Some(&b) = unicode_to_byte.get(&c) {
                        bytes.push(b);
                    } else {
                        // Character not in GPT-2 table (e.g. ▁); emit its UTF-8 bytes.
                        let mut buf = [0u8; 4];
                        let encoded = c.encode_utf8(&mut buf);
                        bytes.extend_from_slice(encoded.as_bytes());
                    }
                }
                bytes
            }
            ByteEncoding::HexFallback => {
                // Check for <0xXX> byte-fallback token
                if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6
                {
                    if let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16) {
                        return vec![byte];
                    }
                }
                // Regular text token — use UTF-8 bytes directly
                token_str.as_bytes().to_vec()
            }
            ByteEncoding::None => token_str.as_bytes().to_vec(),
        }
    }

    /// Convert a token ID to raw bytes (without metaspace replacement).
    fn token_id_to_raw_bytes(&self, id: u32) -> Vec<u8> {
        match self.id_to_token.get(id as usize) {
            Some(s) => self.token_str_to_bytes(s),
            None => Vec::new(),
        }
    }

    /// Decode token IDs back into text.
    ///
    /// Handles byte-level tokens (GPT-2 and SentencePiece `<0xXX>` styles) by
    /// collecting raw bytes and converting to a proper UTF-8 string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        // Collect raw bytes from all tokens
        let mut bytes = Vec::new();
        for &id in tokens {
            bytes.extend_from_slice(&self.token_id_to_raw_bytes(id));
        }
        // Convert bytes to UTF-8 string (lossy fallback for malformed sequences)
        let text = String::from_utf8(bytes)
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned());
        // Apply metaspace replacement
        self.decode_mistral(&text)
    }

    /// Create an incremental decoder for streaming token-by-token output.
    ///
    /// The returned decoder buffers incomplete UTF-8 byte sequences across
    /// token boundaries, emitting text only when complete characters are available.
    pub fn incremental_decoder(&self) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: self,
            buffer: Vec::new(),
        }
    }
}

/// Streaming decoder that correctly handles multi-byte UTF-8 sequences
/// when tokens are decoded one at a time.
///
/// Some byte-level tokens (GPT-2 or SentencePiece byte-fallback) represent
/// individual bytes of a multi-byte UTF-8 character. When generated in sequence,
/// these must be combined before converting to text.
pub struct IncrementalDecoder<'a> {
    tokenizer: &'a Tokenizer,
    buffer: Vec<u8>,
}

impl<'a> IncrementalDecoder<'a> {
    /// Push a token ID and return any complete UTF-8 text that can be emitted.
    ///
    /// The returned string may be empty if the token's bytes form an incomplete
    /// UTF-8 sequence (the bytes are buffered until more arrive).
    pub fn push(&mut self, token_id: u32) -> String {
        let bytes = self.tokenizer.token_id_to_raw_bytes(token_id);
        self.buffer.extend_from_slice(&bytes);
        self.emit_valid()
    }

    /// Extract the longest valid UTF-8 prefix from the buffer and return it.
    fn emit_valid(&mut self) -> String {
        match std::str::from_utf8(&self.buffer) {
            Ok(s) => {
                let result = self.tokenizer.decode_mistral(s);
                self.buffer.clear();
                result
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if valid_up_to > 0 {
                    let valid = std::str::from_utf8(&self.buffer[..valid_up_to]).unwrap();
                    let result = self.tokenizer.decode_mistral(valid);
                    self.buffer = self.buffer[valid_up_to..].to_vec();
                    result
                } else {
                    String::new()
                }
            }
        }
    }

    /// Flush any remaining buffered bytes as (possibly lossy) UTF-8.
    ///
    /// Call this at the end of generation to emit any remaining bytes.
    pub fn flush(&mut self) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }
        let text = String::from_utf8_lossy(&self.buffer).into_owned();
        self.buffer.clear();
        self.tokenizer.decode_mistral(&text)
    }
}
