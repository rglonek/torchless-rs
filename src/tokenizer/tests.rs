use super::Tokenizer;
use std::collections::HashMap;

fn create_test_tokenizer() -> Tokenizer {
    let mut vocab = HashMap::new();
    vocab.insert("<s>".to_string(), 0);
    vocab.insert("▁".to_string(), 1);
    vocab.insert("H".to_string(), 2);
    vocab.insert("e".to_string(), 3);
    vocab.insert("l".to_string(), 4);
    vocab.insert("o".to_string(), 5);
    vocab.insert("▁H".to_string(), 6);
    vocab.insert("He".to_string(), 7);
    vocab.insert("ll".to_string(), 8);
    vocab.insert("Hello".to_string(), 9);

    // Add byte fallback tokens
    for i in 0..256u32 {
        let token = format!("<0x{:02X}>", i);
        vocab.insert(token, 10 + i);
    }

    let merges = vec![
        "▁ H".to_string(),
        "H e".to_string(),
        "l l".to_string(),
        "He ll".to_string(),
        "Hell o".to_string(),
    ];

    Tokenizer::new(vocab, merges)
}

#[test]
fn test_pack_unpack() {
    let left = 42u32;
    let right = 123u32;
    let packed = Tokenizer::pack(left, right);
    let (l, r) = Tokenizer::unpack(packed);
    assert_eq!(l, left);
    assert_eq!(r, right);
}

#[test]
fn test_pre_tokenize() {
    let tokenizer = create_test_tokenizer();
    let result = tokenizer.pre_tokenize_mistral("Hello world");
    assert_eq!(result, "▁Hello▁world");
}

#[test]
fn test_encode_simple() {
    let tokenizer = create_test_tokenizer();
    let tokens = tokenizer.encode("Hello");

    // Should start with BOS token
    assert_eq!(tokens[0], 0); // <s>

    // After BPE merges, "Hello" should be tokenized
    // The exact token IDs depend on the merge order
    assert!(tokens.len() > 1);
}

#[test]
fn test_decode() {
    let tokenizer = create_test_tokenizer();

    // Decode a simple sequence
    let tokens = vec![1, 2, 3, 4, 4, 5]; // ▁ H e l l o
    let decoded = tokenizer.decode(&tokens);

    // Should replace ▁ with space
    assert!(decoded.contains("Hello") || decoded.contains("ello"));
}

#[test]
fn test_byte_fallback() {
    let tokenizer = create_test_tokenizer();

    // Get ID for a character not in vocab (should use byte fallback)
    let ids = tokenizer.get_id("€"); // Euro sign, multi-byte UTF-8

    // Euro sign is 3 bytes in UTF-8: 0xE2 0x82 0xAC
    assert_eq!(ids.len(), 3);
}

// Integration tests with actual vocab - to be added when fixtures are available
#[test]
#[ignore]
fn test_encode_with_real_vocab() {
    // TODO: Load actual Mistral tokenizer and test
}
