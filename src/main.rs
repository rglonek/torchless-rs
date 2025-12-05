use std::env;
use torchless::{generate, InferenceState, Mistral, Parameters};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    // Check for --debug flag
    let debug = args.iter().any(|arg| arg == "--debug");

    // Filter out --debug from args
    let args: Vec<String> = args.into_iter().filter(|arg| arg != "--debug").collect();

    if args.len() < 3 {
        eprintln!("Usage: {} [--debug] <model_path> <prompt>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let prompt = &args[2];

    println!("Loading model from: {}", model_path);
    let params = Parameters::load(model_path)?;

    println!("Loading weights...");
    let model = Mistral::load(params)?;

    println!("Initializing inference state...");
    let mut state = InferenceState::new(model.config.clone());

    println!("Tokenizing prompt: {}", prompt);
    let tokens = model.tokenizer.encode(prompt);
    println!("Tokens: {:?}", tokens);

    // Process prompt tokens (all but last)
    if debug {
        eprintln!("Processing {} prompt tokens...", tokens.len() - 1);
    }
    for (i, &token) in tokens[..tokens.len() - 1].iter().enumerate() {
        if debug {
            eprintln!("\nPrompt token {}/{}", i + 1, tokens.len() - 1);
        }
        model.forward(&mut state, token, debug);
        state.pos += 1;
    }

    // Generate tokens
    if debug {
        eprintln!("\nGenerating tokens:");
    }
    print!("{}", prompt);

    let mut token = *tokens.last().unwrap();
    let temperature = 0.0; // greedy sampling

    for i in 0..50 {
        if debug && i % 5 == 0 {
            eprintln!("\nGeneration step {}/50", i);
        }
        token = generate(&model, &mut state, token, temperature, debug);
        let decoded = model.tokenizer.decode(&[token]);
        print!("{}", decoded);
        std::io::Write::flush(&mut std::io::stdout())?;
        state.pos += 1;
    }

    println!("\n");

    Ok(())
}
