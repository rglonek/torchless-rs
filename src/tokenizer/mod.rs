use std::collections::HashMap;

#[cfg(test)]
mod tests;

const METASPACE: &str = "▁";
const BOS_TOKEN: &str = "<s>";

#[derive(Clone)]
pub struct Tokenizer {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
    // Maps packed (left, right) pair to its rank in merge order
    merge_to_rank: HashMap<u64, usize>,
    // Maps packed (left, right) pair to the merged token ID
    merge_to_id: HashMap<u64, u32>,
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

        Self {
            token_to_id: vocab,
            id_to_token,
            merge_to_rank,
            merge_to_id,
        }
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

    /// Decode token IDs back into text
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&id| self.id_to_token.get(id as usize))
            .map(|s| self.decode_mistral(s))
            .collect::<String>()
    }
}
