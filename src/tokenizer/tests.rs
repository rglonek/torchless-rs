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

/// Test that <0xXX> byte-fallback tokens are decoded to actual bytes.
/// Cyrillic "е" = UTF-8 bytes [0xD0, 0xB5].
#[test]
fn test_decode_hex_fallback_multibyte() {
    let tokenizer = create_test_tokenizer();

    // Token IDs for <0xD0> and <0xB5> (base 10 + byte value)
    let d0_id = 10 + 0xD0; // <0xD0>
    let b5_id = 10 + 0xB5; // <0xB5>

    let decoded = tokenizer.decode(&[d0_id, b5_id]);
    assert_eq!(decoded, "е"); // Cyrillic small letter ie
}

/// Test that the incremental decoder correctly buffers partial UTF-8 sequences.
#[test]
fn test_incremental_decoder_multibyte() {
    let tokenizer = create_test_tokenizer();
    let mut dec = tokenizer.incremental_decoder();

    let d0_id = 10 + 0xD0;
    let b5_id = 10 + 0xB5;

    // First byte of a 2-byte UTF-8 sequence — should buffer, emit nothing
    let out1 = dec.push(d0_id);
    assert_eq!(out1, "");

    // Second byte completes the character
    let out2 = dec.push(b5_id);
    assert_eq!(out2, "е");
}

/// Test incremental decoder with a mix of ASCII and multi-byte tokens.
#[test]
fn test_incremental_decoder_mixed() {
    let tokenizer = create_test_tokenizer();
    let mut dec = tokenizer.incremental_decoder();

    // 'H' is token ID 2, which is a normal ASCII token
    let out = dec.push(2);
    assert_eq!(out, "H");

    // Then push multi-byte Cyrillic
    let d0_id = 10 + 0xD0;
    let b5_id = 10 + 0xB5;
    assert_eq!(dec.push(d0_id), "");
    assert_eq!(dec.push(b5_id), "е");

    // Flush should return empty since everything was emitted
    assert_eq!(dec.flush(), "");
}

/// Test that GPT-2 byte encoding is detected and handled correctly.
#[test]
fn test_gpt2_byte_encoding_detection() {
    use super::build_gpt2_byte_to_unicode;

    let table = build_gpt2_byte_to_unicode();
    let mut vocab = HashMap::new();

    // Build a vocab using GPT-2 byte encoding
    for (byte_val, &ch) in table.iter().enumerate() {
        vocab.insert(ch.to_string(), byte_val as u32);
    }
    // Add a multi-char token: "Hello" using GPT-2 chars (all ASCII identity range)
    vocab.insert("Hello".to_string(), 256);

    let tokenizer = Tokenizer::new(vocab, vec![]);

    // Decode Cyrillic "ем" = bytes [0xD0, 0xB5, 0xD0, 0xBC]
    // In GPT-2: byte 0xD0 → U+00D0 (Ð), byte 0xB5 → U+00B5 (µ),
    //           byte 0xBC → U+00BC (¼)
    let d0 = 0xD0u32; // token for byte 0xD0
    let b5 = 0xB5u32; // token for byte 0xB5
    let bc = 0xBCu32; // token for byte 0xBC

    let decoded = tokenizer.decode(&[d0, b5, d0, bc]);
    assert_eq!(decoded, "ем"); // Cyrillic "em"
}

// Integration tests with actual vocab - to be added when fixtures are available
#[test]
#[ignore]
fn test_encode_with_real_vocab() {
    // TODO: Load actual Mistral tokenizer and test
}
