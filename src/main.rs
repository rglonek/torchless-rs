use std::env;
use torchless::{
    generate, init_backend, print_backend_summary, BackendPreference, InferenceState, Mistral,
    Parameters,
};

fn print_usage(program: &str) {
    eprintln!("Usage: {} [OPTIONS] <model_path> <prompt>", program);
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --backend <BACKEND>   Compute backend: auto, cpu, cuda, rocm, metal, opencl");
    eprintln!("                        (default: auto - selects best available)");
    eprintln!("  --max-tokens <N>      Maximum tokens to generate (default: 50)");
    eprintln!("  --temperature <T>     Sampling temperature, 0.0 for greedy (default: 0.0)");
    eprintln!("  --list-backends       List available backends and exit");
    eprintln!("  --debug               Enable debug output");
    eprintln!("  --help                Show this help message");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {} model.bin \"Hello, world\"", program);
    eprintln!(
        "  {} --backend cuda model.gguf \"The capital of France is\"",
        program
    );
    eprintln!(
        "  {} --backend cpu --max-tokens 100 model.bin \"Once upon a time\"",
        program
    );
}

fn parse_backend(s: &str) -> Option<BackendPreference> {
    match s.to_lowercase().as_str() {
        "auto" => Some(BackendPreference::Auto),
        "cpu" => Some(BackendPreference::Cpu),
        #[cfg(feature = "cuda")]
        "cuda" => Some(BackendPreference::Cuda),
        #[cfg(feature = "rocm")]
        "rocm" => Some(BackendPreference::Rocm),
        #[cfg(feature = "metal-gpu")]
        "metal" => Some(BackendPreference::Metal),
        #[cfg(feature = "opencl")]
        "opencl" => Some(BackendPreference::OpenCL),
        _ => None,
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let program = &args[0];

    // Parse options
    let mut debug = false;
    let mut backend_pref = BackendPreference::Auto;
    let mut max_tokens: usize = 50;
    let mut temperature: f32 = 0.0;
    let mut list_backends = false;
    let mut positional_args: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--debug" => {
                debug = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_usage(program);
                return Ok(());
            }
            "--list-backends" => {
                list_backends = true;
                i += 1;
            }
            "--backend" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: --backend requires a value");
                    std::process::exit(1);
                }
                match parse_backend(&args[i + 1]) {
                    Some(pref) => backend_pref = pref,
                    None => {
                        eprintln!(
                            "Error: unknown backend '{}'. Available: auto, cpu",
                            args[i + 1]
                        );
                        #[cfg(feature = "cuda")]
                        eprint!(", cuda");
                        #[cfg(feature = "rocm")]
                        eprint!(", rocm");
                        #[cfg(feature = "metal-gpu")]
                        eprint!(", metal");
                        #[cfg(feature = "opencl")]
                        eprint!(", opencl");
                        eprintln!();
                        std::process::exit(1);
                    }
                }
                i += 2;
            }
            "--max-tokens" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: --max-tokens requires a value");
                    std::process::exit(1);
                }
                max_tokens = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Error: invalid --max-tokens value '{}'", args[i + 1]);
                    std::process::exit(1);
                });
                i += 2;
            }
            "--temperature" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: --temperature requires a value");
                    std::process::exit(1);
                }
                temperature = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Error: invalid --temperature value '{}'", args[i + 1]);
                    std::process::exit(1);
                });
                i += 2;
            }
            arg if arg.starts_with("-") => {
                eprintln!("Error: unknown option '{}'", arg);
                eprintln!("Use --help for usage information.");
                std::process::exit(1);
            }
            _ => {
                positional_args.push(args[i].clone());
                i += 1;
            }
        }
    }

    // Handle --list-backends
    if list_backends {
        print_backend_summary();
        return Ok(());
    }

    // Check required arguments
    if positional_args.len() < 2 {
        print_usage(program);
        std::process::exit(1);
    }

    let model_path = &positional_args[0];
    let prompt = &positional_args[1];

    // Initialize backend
    let backend = init_backend(backend_pref)?;
    println!("Using backend: {}", backend.name());

    println!("Loading model from: {}", model_path);
    let params = Parameters::load(model_path)?;

    println!("Loading weights...");
    let model = Mistral::load(params)?;

    println!("Initializing inference state...");
    let mut state = InferenceState::new(model.config.clone());

    println!("Tokenizing prompt: {}", prompt);
    let tokens = model.tokenizer.encode(prompt);
    if debug {
        println!("Tokens: {:?}", tokens);
    }

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
        eprintln!("\nGenerating {} tokens:", max_tokens);
    }
    print!("{}", prompt);

    let mut token = *tokens.last().unwrap();

    for i in 0..max_tokens {
        if debug && i.is_multiple_of(10) {
            eprintln!("\nGeneration step {}/{}", i, max_tokens);
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
