use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use torchless::tokenizer::Tokenizer;
use torchless::{
    display_thinking_token, generate, generate_lazy, generate_lazy_until_eos, generate_until_eos,
    init_backend, print_backend_summary, strip_thinking, BackendPreference, ChatMessage, ChatRole,
    ChatTemplate, GenerationResult, InferenceState, LazyMistral, Mistral, Parameters,
    SamplingConfig, SelfSpeculativeDecoder, SpeculativeConfig, ThinkingState,
};

/// Mutable session configuration for the chat REPL.
/// All fields can be changed at runtime via in-session commands.
struct ChatSessionConfig {
    sampling: SamplingConfig,
    speculative: bool,
    max_tokens: usize,
    debug: bool,
    system_prompt: Option<String>,
}

fn print_usage(program: &str) {
    eprintln!("Usage: {} [OPTIONS] <model_path> [prompt]", program);
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --backend <BACKEND>   Compute backend: auto, cpu, cuda, rocm, metal, opencl");
    eprintln!("                        (default: auto - selects best available)");
    eprintln!("  --max-tokens <N>      Maximum tokens to generate per response");
    eprintln!("                        (default: max-seq-len/4, clamped to 1024..32768)");
    eprintln!("  --max-seq-len <N>     Maximum sequence length / context window");
    eprintln!("                        (default: model's max_position_embeddings)");
    eprintln!("  --temperature <T>     Sampling temperature (default: 0.7)");
    eprintln!(
        "                        0.0 = greedy/deterministic, 0.7 = balanced, 1.0+ = creative"
    );
    eprintln!("  --top-k <K>           Top-k sampling: only consider the top K tokens");
    eprintln!("                        (disabled by default, typical: 40-100)");
    eprintln!("  --top-p <P>           Nucleus sampling: keep smallest set of tokens with");
    eprintln!("                        cumulative probability >= P (disabled by default, typical: 0.9-0.95)");
    eprintln!("  --lazy                Use lazy model loading (memory-efficient, loads tensors on demand)");
    eprintln!("  --speculative         Enable self-speculative decoding for faster generation");
    eprintln!("  --chat                Enter interactive chat mode");
    eprintln!("  --system <MSG>        System prompt for chat mode");
    eprintln!(
        "  --show-thinking       Show <think>...</think> reasoning traces from thinking models"
    );
    eprintln!(
        "                        (auto-detected; use /thinking in chat to toggle at runtime)"
    );
    eprintln!("  --list-backends       List available backends and exit");
    eprintln!("  --debug               Enable debug output");
    eprintln!("  --help                Show this help message");
    eprintln!();
    eprintln!("See docs/params.md for detailed parameter documentation.");
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
    eprintln!(
        "  {} --chat --system \"You are a helpful assistant.\" model.bin",
        program
    );
    eprintln!(
        "  {} --lazy --top-k 50 --top-p 0.9 model.bin \"Tell me a story\"",
        program
    );
    eprintln!(
        "  {} --speculative model.bin \"Explain quantum computing\"",
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

/// Resolve max_seq_len: default to model's max_position_embeddings, warn if user exceeds it.
fn resolve_max_seq_len(user_value: Option<usize>, model_max: usize) -> usize {
    match user_value {
        Some(requested) => {
            if requested > model_max {
                eprintln!(
                    "WARNING: --max-seq-len {} exceeds the model's trained context length ({}).",
                    requested, model_max
                );
                eprintln!(
                    "WARNING: Output quality will likely degrade beyond position {} \
                     because the model's positional embeddings (RoPE) were not trained \
                     for longer sequences.",
                    model_max
                );
                eprintln!(
                    "WARNING: Consider using --max-seq-len {} or lower for best results.",
                    model_max
                );
            }
            requested
        }
        None => {
            // Default to model's trained context length
            model_max
        }
    }
}

/// Resolve max_tokens: default to max_seq_len/4, clamped to a sane range.
///
/// - If max_seq_len >= 4096: clamp to 2048..32768
/// - If max_seq_len < 4096:  clamp to 1024..32768
fn resolve_max_tokens(user_value: Option<usize>, max_seq_len: usize) -> usize {
    match user_value {
        Some(requested) => requested,
        None => {
            let default = max_seq_len / 4;
            let min_tokens = if max_seq_len >= 4096 { 2048 } else { 1024 };
            default.clamp(min_tokens, 32768)
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let program = &args[0];

    // Parse options -- max_seq_len and max_tokens are Option<usize> because
    // their defaults depend on the model config (resolved after model load).
    let mut debug = false;
    let mut backend_pref = BackendPreference::Auto;
    let mut max_tokens_arg: Option<usize> = None;
    let mut max_seq_len_arg: Option<usize> = None;
    let mut temperature: f32 = 0.7;
    let mut top_k_arg: Option<usize> = None;
    let mut top_p_arg: Option<f32> = None;
    let mut lazy = false;
    let mut speculative = false;
    let mut show_thinking = false;
    let mut list_backends = false;
    let mut chat_mode = false;
    let mut system_prompt: Option<String> = None;
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
            "--chat" => {
                chat_mode = true;
                i += 1;
            }
            "--lazy" => {
                lazy = true;
                i += 1;
            }
            "--speculative" => {
                speculative = true;
                i += 1;
            }
            "--show-thinking" => {
                show_thinking = true;
                i += 1;
            }
            "--system" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: --system requires a value");
                    std::process::exit(1);
                }
                system_prompt = Some(args[i + 1].clone());
                i += 2;
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
                max_tokens_arg = Some(args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Error: invalid --max-tokens value '{}'", args[i + 1]);
                    std::process::exit(1);
                }));
                i += 2;
            }
            "--max-seq-len" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: --max-seq-len requires a value");
                    std::process::exit(1);
                }
                max_seq_len_arg = Some(args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Error: invalid --max-seq-len value '{}'", args[i + 1]);
                    std::process::exit(1);
                }));
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
            "--top-k" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: --top-k requires a value");
                    std::process::exit(1);
                }
                top_k_arg = Some(args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Error: invalid --top-k value '{}'", args[i + 1]);
                    std::process::exit(1);
                }));
                i += 2;
            }
            "--top-p" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: --top-p requires a value");
                    std::process::exit(1);
                }
                top_p_arg = Some(args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Error: invalid --top-p value '{}'", args[i + 1]);
                    std::process::exit(1);
                }));
                i += 2;
            }
            arg if arg.starts_with('-') => {
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

    // Build sampling config from parsed arguments
    let sampling_config = SamplingConfig {
        temperature,
        top_k: top_k_arg,
        top_p: top_p_arg,
    };

    // Warn about speculative + top-k/top-p incompatibility
    if speculative && (top_k_arg.is_some() || top_p_arg.is_some()) {
        eprintln!(
            "Warning: --top-k and --top-p are ignored in --speculative mode. \
             Speculative decoding uses its own temperature-based sampling."
        );
    }

    // In chat mode, only model_path is required
    if chat_mode {
        if positional_args.is_empty() {
            eprintln!("Error: <model_path> is required");
            print_usage(program);
            std::process::exit(1);
        }
        let model_path = &positional_args[0];
        return run_chat(
            model_path,
            backend_pref,
            max_tokens_arg,
            max_seq_len_arg,
            &sampling_config,
            system_prompt.as_deref(),
            lazy,
            speculative,
            show_thinking,
            debug,
        );
    }

    // Single-shot mode: both model_path and prompt are required
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

    if lazy {
        // ------------------------------------------------------------------
        // Lazy loading: memory-efficient, loads tensors on demand via mmap
        // ------------------------------------------------------------------
        println!("Loading weights (lazy mode)...");
        let model = LazyMistral::load(&params)?;

        // Resolve defaults now that we know the model config
        let max_seq_len =
            resolve_max_seq_len(max_seq_len_arg, model.config.max_position_embeddings);
        let max_tokens = resolve_max_tokens(max_tokens_arg, max_seq_len);

        if debug {
            eprintln!(
                "Model max_position_embeddings: {}, using max_seq_len: {}, max_tokens: {}",
                model.config.max_position_embeddings, max_seq_len, max_tokens
            );
        }

        println!("Initializing inference state...");
        let mut state = InferenceState::with_seq_len(model.config.clone(), max_seq_len);

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

        if speculative {
            generate_speculative_until_eos(
                |s, t| model.forward(s, t, debug),
                &model.tokenizer,
                &mut state,
                token,
                temperature,
                max_tokens,
                &[],  // no EOS detection in single-shot mode
                None, // no thinking filtering in single-shot mode
                debug,
            );
        } else {
            for i in 0..max_tokens {
                if debug && i % 10 == 0 {
                    eprintln!("\nGeneration step {}/{}", i, max_tokens);
                }
                token = generate_lazy(&model, &mut state, token, &sampling_config, debug);
                let decoded = model.tokenizer.decode(&[token]);
                print!("{}", decoded);
                io::stdout().flush()?;
                state.pos += 1;
            }
        }
    } else {
        // ------------------------------------------------------------------
        // Eager loading: all weights loaded into memory up front
        // ------------------------------------------------------------------
        println!("Loading weights...");
        let model = Mistral::load(params)?;

        // Resolve defaults now that we know the model config
        let max_seq_len =
            resolve_max_seq_len(max_seq_len_arg, model.config.max_position_embeddings);
        let max_tokens = resolve_max_tokens(max_tokens_arg, max_seq_len);

        if debug {
            eprintln!(
                "Model max_position_embeddings: {}, using max_seq_len: {}, max_tokens: {}",
                model.config.max_position_embeddings, max_seq_len, max_tokens
            );
        }

        println!("Initializing inference state...");
        let mut state = InferenceState::with_seq_len(model.config.clone(), max_seq_len);

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

        if speculative {
            generate_speculative_until_eos(
                |s, t| model.forward(s, t, debug),
                &model.tokenizer,
                &mut state,
                token,
                temperature,
                max_tokens,
                &[],  // no EOS detection in single-shot mode
                None, // no thinking filtering in single-shot mode
                debug,
            );
        } else {
            for i in 0..max_tokens {
                if debug && i % 10 == 0 {
                    eprintln!("\nGeneration step {}/{}", i, max_tokens);
                }
                token = generate(&model, &mut state, token, &sampling_config, debug);
                let decoded = model.tokenizer.decode(&[token]);
                print!("{}", decoded);
                io::stdout().flush()?;
                state.pos += 1;
            }
        }
    }

    println!("\n");

    Ok(())
}

// =============================================================================
// Speculative Generation
// =============================================================================

/// Generate tokens using self-speculative decoding until EOS or `max_tokens` is reached.
///
/// Uses the same model at different temperatures for draft and verification.
/// Pass an empty `eos_token_ids` slice to disable EOS detection (single-shot mode).
#[allow(clippy::too_many_arguments)]
fn generate_speculative_until_eos<'a, F>(
    forward: F,
    tokenizer: &Tokenizer,
    state: &mut InferenceState,
    first_token: u32,
    temperature: f32,
    max_tokens: usize,
    eos_token_ids: &[u32],
    thinking: Option<&ThinkingState>,
    debug: bool,
) -> GenerationResult
where
    F: Fn(&mut InferenceState, u32) + 'a,
{
    let draft_temp = (temperature * 1.5).max(0.01);
    let main_temp = temperature.max(0.01);
    let config = SpeculativeConfig {
        temperature,
        ..Default::default()
    };

    let mut decoder = SelfSpeculativeDecoder::new(forward, config, draft_temp, main_temp);
    let mut tokens = Vec::new();
    let mut token = first_token;
    let mut stopped_at_eos = false;

    while tokens.len() < max_tokens {
        let pos_before = state.pos;
        let accepted = decoder.generate_step(state, token);

        for (i, &t) in accepted.iter().enumerate() {
            if eos_token_ids.contains(&t) {
                // Roll back state.pos to just after the useful tokens
                state.pos = pos_before + i;
                stopped_at_eos = true;
                break;
            }
            if let Some(ts) = thinking {
                let action = ts.process_token(t);
                display_thinking_token(action, || tokenizer.decode(&[t]));
            } else {
                let text = tokenizer.decode(&[t]);
                print!("{}", text);
            }
            let _ = io::stdout().flush();
            tokens.push(t);
            if tokens.len() >= max_tokens {
                break;
            }
        }

        if stopped_at_eos || tokens.len() >= max_tokens {
            break;
        }

        if let Some(&last) = accepted.last() {
            token = last;
        }
    }

    if debug {
        let stats = decoder.stats();
        eprintln!(
            "[speculative: acceptance_rate={:.1}%, tokens/iteration={:.1}]",
            stats.acceptance_rate() * 100.0,
            stats.tokens_per_iteration()
        );
    }

    GenerationResult {
        tokens,
        stopped_at_eos,
    }
}

// =============================================================================
// Chat Mode
// =============================================================================

/// Run the interactive chat REPL.
#[allow(clippy::too_many_arguments)]
fn run_chat(
    model_path: &str,
    backend_pref: BackendPreference,
    max_tokens_arg: Option<usize>,
    max_seq_len_arg: Option<usize>,
    sampling_config: &SamplingConfig,
    system_prompt: Option<&str>,
    lazy: bool,
    speculative: bool,
    show_thinking: bool,
    debug: bool,
) -> anyhow::Result<()> {
    // Initialize backend
    let backend = init_backend(backend_pref)?;
    println!("Using backend: {}", backend.name());

    println!("Loading model from: {}", model_path);
    let params = Parameters::load(model_path)?;

    if lazy {
        // Lazy loading: memory-efficient
        println!("Loading weights (lazy mode)...");
        let model = LazyMistral::load(&params)?;

        let max_seq_len =
            resolve_max_seq_len(max_seq_len_arg, model.config.max_position_embeddings);
        let max_tokens = resolve_max_tokens(max_tokens_arg, max_seq_len);

        println!(
            "Context window: {} tokens (model trained: {}), max response: {} tokens",
            max_seq_len, model.config.max_position_embeddings, max_tokens
        );

        let state = InferenceState::with_seq_len(model.config.clone(), max_seq_len);

        // Initialize thinking state (auto-detects from tokenizer vocabulary)
        let chat_template = ChatTemplate::Mistral; // resolved again inside run_chat_repl
        let thinking_ids = chat_template.thinking_token_ids(&model.tokenizer);
        let thinking_state = ThinkingState::new(thinking_ids, show_thinking);

        if thinking_state.is_thinking_model() {
            eprintln!(
                "Thinking model detected. Use /thinking to toggle reasoning trace display{}.",
                if show_thinking {
                    " (currently: shown)"
                } else {
                    " (currently: hidden)"
                }
            );
        }

        let config = ChatSessionConfig {
            sampling: sampling_config.clone(),
            speculative,
            max_tokens,
            debug,
            system_prompt: system_prompt.map(|s| s.to_string()),
        };

        run_chat_repl(
            state,
            &model.tokenizer,
            config,
            max_seq_len,
            &thinking_state,
            &mut |state, tokens, debug| {
                for (i, &token) in tokens.iter().enumerate() {
                    if debug && (i % 50 == 0) {
                        eprintln!("Processing prompt token {}/{}", i + 1, tokens.len());
                    }
                    model.forward(state, token, debug);
                    state.pos += 1;
                }
            },
            &mut |state, first_token, eos_ids, config| {
                let debug = config.debug;
                if config.speculative {
                    generate_speculative_until_eos(
                        |s, t| model.forward(s, t, debug),
                        &model.tokenizer,
                        state,
                        first_token,
                        config.sampling.temperature,
                        config.max_tokens,
                        eos_ids,
                        Some(&thinking_state),
                        debug,
                    )
                } else {
                    generate_lazy_until_eos(
                        &model,
                        state,
                        first_token,
                        &config.sampling,
                        config.max_tokens,
                        eos_ids,
                        debug,
                        |token_id| {
                            let action = thinking_state.process_token(token_id);
                            display_thinking_token(action, || model.tokenizer.decode(&[token_id]));
                            let _ = io::stdout().flush();
                        },
                    )
                }
            },
        )
    } else {
        // Eager loading: all weights in memory
        println!("Loading weights...");
        let model = Mistral::load(params)?;

        let max_seq_len =
            resolve_max_seq_len(max_seq_len_arg, model.config.max_position_embeddings);
        let max_tokens = resolve_max_tokens(max_tokens_arg, max_seq_len);

        println!(
            "Context window: {} tokens (model trained: {}), max response: {} tokens",
            max_seq_len, model.config.max_position_embeddings, max_tokens
        );

        let state = InferenceState::with_seq_len(model.config.clone(), max_seq_len);

        // Initialize thinking state (auto-detects from tokenizer vocabulary)
        let chat_template = ChatTemplate::Mistral;
        let thinking_ids = chat_template.thinking_token_ids(&model.tokenizer);
        let thinking_state = ThinkingState::new(thinking_ids, show_thinking);

        if thinking_state.is_thinking_model() {
            eprintln!(
                "Thinking model detected. Use /thinking to toggle reasoning trace display{}.",
                if show_thinking {
                    " (currently: shown)"
                } else {
                    " (currently: hidden)"
                }
            );
        }

        let config = ChatSessionConfig {
            sampling: sampling_config.clone(),
            speculative,
            max_tokens,
            debug,
            system_prompt: system_prompt.map(|s| s.to_string()),
        };

        run_chat_repl(
            state,
            &model.tokenizer,
            config,
            max_seq_len,
            &thinking_state,
            &mut |state, tokens, debug| {
                for (i, &token) in tokens.iter().enumerate() {
                    if debug && (i % 50 == 0) {
                        eprintln!("Processing prompt token {}/{}", i + 1, tokens.len());
                    }
                    model.forward(state, token, debug);
                    state.pos += 1;
                }
            },
            &mut |state, first_token, eos_ids, config| {
                let debug = config.debug;
                if config.speculative {
                    generate_speculative_until_eos(
                        |s, t| model.forward(s, t, debug),
                        &model.tokenizer,
                        state,
                        first_token,
                        config.sampling.temperature,
                        config.max_tokens,
                        eos_ids,
                        Some(&thinking_state),
                        debug,
                    )
                } else {
                    generate_until_eos(
                        &model,
                        state,
                        first_token,
                        &config.sampling,
                        config.max_tokens,
                        eos_ids,
                        debug,
                        |token_id| {
                            let action = thinking_state.process_token(token_id);
                            display_thinking_token(action, || model.tokenizer.decode(&[token_id]));
                            let _ = io::stdout().flush();
                        },
                    )
                }
            },
        )
    }
}

/// The interactive chat REPL loop, parameterized by model-specific closures.
///
/// `config` owns the mutable session configuration (sampling, speculative, max_tokens, debug,
/// system_prompt) so that in-session commands can change these values at runtime.
/// `process_tokens` processes prompt tokens through the model (advancing state.pos).
/// `generate_response` generates a response from a starting token, returning the result.
/// `thinking_state` tracks thinking model output filtering (auto-detected; no-op if not a thinking model).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_chat_repl(
    mut state: InferenceState,
    tokenizer: &Tokenizer,
    mut config: ChatSessionConfig,
    max_seq_len: usize,
    thinking_state: &ThinkingState,
    process_tokens: &mut dyn FnMut(&mut InferenceState, &[u32], bool),
    generate_response: &mut dyn FnMut(
        &mut InferenceState,
        u32,
        &[u32],
        &ChatSessionConfig,
    ) -> GenerationResult,
) -> anyhow::Result<()> {
    // Select chat template (default to Mistral for now)
    let chat_template = ChatTemplate::Mistral;
    let eos_ids = chat_template.eos_token_ids(tokenizer);

    if eos_ids.is_empty() {
        eprintln!(
            "Warning: no EOS tokens found in vocabulary for {:?} template. \
             Generation will only stop at --max-tokens ({}).",
            chat_template, config.max_tokens
        );
    }

    println!("Initializing inference state...");

    // Build initial conversation history
    let mut history: Vec<ChatMessage> = Vec::new();
    if let Some(ref sys) = config.system_prompt {
        history.push(ChatMessage::system(sys));
    }

    // Track how many tokens have been processed so far (for KV cache reuse)
    let mut processed_tokens: Vec<u32> = Vec::new();

    // Track length of the last assistant generation (for /context and /retry)
    let mut last_generation_len: usize = 0;

    println!();
    println!("Chat mode active. Type /help for commands.");
    println!();

    let stdin = io::stdin();
    let mut reader = stdin.lock().lines();

    loop {
        // Print prompt
        print!("> ");
        io::stdout().flush()?;

        // Read user input
        let line = match reader.next() {
            Some(Ok(line)) => line,
            Some(Err(e)) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
            None => break, // EOF
        };

        let input = line.trim().to_string();

        if input.is_empty() {
            continue;
        }

        // Flag for /retry: skip message processing, go straight to generation
        let mut skip_processing = false;

        // Handle commands: split on first space for arg parsing
        let (cmd, arg) = match input.find(' ') {
            Some(pos) => (&input[..pos], Some(input[pos + 1..].trim())),
            None => (input.as_str(), None),
        };

        match cmd {
            "/quit" | "/exit" | "/q" => {
                println!("Goodbye!");
                break;
            }
            "/clear" => {
                history.clear();
                if let Some(ref sys) = config.system_prompt {
                    history.push(ChatMessage::system(sys));
                }
                state.reset();
                processed_tokens.clear();
                println!("Conversation cleared.");
                continue;
            }
            "/thinking" => {
                match arg {
                    Some("on") => {
                        thinking_state.set_show(true);
                        println!("Thinking display: on (traces shown dimmed)");
                    }
                    Some("off") => {
                        thinking_state.set_show(false);
                        println!("Thinking display: off (traces hidden)");
                    }
                    None => {
                        let new = thinking_state.toggle_show();
                        println!(
                            "Thinking display: {}",
                            if new {
                                "on (traces shown dimmed)"
                            } else {
                                "off (traces hidden)"
                            }
                        );
                    }
                    Some(other) => {
                        eprintln!("Invalid argument '{}'. Usage: /thinking [on|off]", other);
                    }
                }
                continue;
            }
            "/temperature" => {
                match arg {
                    Some(val) => match val.parse::<f32>() {
                        Ok(t) if t >= 0.0 => {
                            config.sampling.temperature = t;
                            println!("Temperature set to {}", t);
                        }
                        Ok(_) => eprintln!("Temperature must be >= 0.0"),
                        Err(_) => {
                            eprintln!("Invalid temperature value '{}'. Expected a number.", val)
                        }
                    },
                    None => eprintln!(
                        "Usage: /temperature <value>  (current: {})",
                        config.sampling.temperature
                    ),
                }
                continue;
            }
            "/top-k" => {
                match arg {
                    Some("off") | Some("none") => {
                        config.sampling.top_k = None;
                        println!("Top-k sampling disabled.");
                    }
                    Some(val) => match val.parse::<usize>() {
                        Ok(k) if k > 0 => {
                            config.sampling.top_k = Some(k);
                            println!("Top-k set to {}", k);
                            if config.speculative {
                                eprintln!("Warning: top-k is ignored in speculative mode.");
                            }
                        }
                        Ok(_) => eprintln!("Top-k must be > 0. Use '/top-k off' to disable."),
                        Err(_) => {
                            eprintln!("Invalid top-k value '{}'. Expected a number or 'off'.", val)
                        }
                    },
                    None => {
                        let current = match config.sampling.top_k {
                            Some(k) => format!("{}", k),
                            None => "off".to_string(),
                        };
                        eprintln!("Usage: /top-k <value|off>  (current: {})", current);
                    }
                }
                continue;
            }
            "/top-p" => {
                match arg {
                    Some("off") | Some("none") => {
                        config.sampling.top_p = None;
                        println!("Top-p (nucleus) sampling disabled.");
                    }
                    Some(val) => match val.parse::<f32>() {
                        Ok(p) if (0.0..=1.0).contains(&p) => {
                            config.sampling.top_p = Some(p);
                            println!("Top-p set to {}", p);
                            if config.speculative {
                                eprintln!("Warning: top-p is ignored in speculative mode.");
                            }
                        }
                        Ok(_) => eprintln!(
                            "Top-p must be between 0.0 and 1.0. Use '/top-p off' to disable."
                        ),
                        Err(_) => {
                            eprintln!("Invalid top-p value '{}'. Expected a number or 'off'.", val)
                        }
                    },
                    None => {
                        let current = match config.sampling.top_p {
                            Some(p) => format!("{}", p),
                            None => "off".to_string(),
                        };
                        eprintln!("Usage: /top-p <value|off>  (current: {})", current);
                    }
                }
                continue;
            }
            "/speculative" => {
                config.speculative = !config.speculative;
                println!(
                    "Speculative decoding: {}",
                    if config.speculative {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                if config.speculative
                    && (config.sampling.top_k.is_some() || config.sampling.top_p.is_some())
                {
                    eprintln!(
                        "Warning: top-k and top-p are ignored in speculative mode. \
                         Speculative decoding uses its own temperature-based sampling."
                    );
                }
                continue;
            }
            "/max-tokens" => {
                match arg {
                    Some(val) => match val.parse::<usize>() {
                        Ok(n) if n > 0 => {
                            config.max_tokens = n;
                            println!("Max tokens per response set to {}", n);
                        }
                        Ok(_) => eprintln!("Max tokens must be > 0."),
                        Err(_) => {
                            eprintln!("Invalid max-tokens value '{}'. Expected a number.", val)
                        }
                    },
                    None => eprintln!(
                        "Usage: /max-tokens <value>  (current: {})",
                        config.max_tokens
                    ),
                }
                continue;
            }
            "/debug" => {
                config.debug = !config.debug;
                println!(
                    "Debug output: {}",
                    if config.debug { "enabled" } else { "disabled" }
                );
                continue;
            }
            "/system" => {
                match arg {
                    Some("off") | Some("none") => {
                        // Remove system prompt
                        if !history.is_empty() && history[0].role == ChatRole::System {
                            history.remove(0);
                        }
                        config.system_prompt = None;
                        state.reset();
                        processed_tokens.clear();
                        println!("System prompt removed. KV cache reset.");
                    }
                    Some(msg) if !msg.is_empty() => {
                        // Set or replace system prompt
                        if !history.is_empty() && history[0].role == ChatRole::System {
                            history[0] = ChatMessage::system(msg);
                        } else {
                            history.insert(0, ChatMessage::system(msg));
                        }
                        config.system_prompt = Some(msg.to_string());
                        state.reset();
                        processed_tokens.clear();
                        println!("System prompt updated. KV cache reset.");
                    }
                    _ => {
                        let current = match config.system_prompt {
                            Some(ref s) => {
                                if s.len() > 60 {
                                    format!("\"{}...\"", &s[..60])
                                } else {
                                    format!("\"{}\"", s)
                                }
                            }
                            None => "none".to_string(),
                        };
                        eprintln!(
                            "Usage: /system <message>  or  /system off  (current: {})",
                            current
                        );
                    }
                }
                continue;
            }
            "/settings" => {
                println!("Current settings:");
                println!("  temperature:  {}", config.sampling.temperature);
                println!(
                    "  top-k:        {}",
                    match config.sampling.top_k {
                        Some(k) => format!("{}", k),
                        None => "off".to_string(),
                    }
                );
                println!(
                    "  top-p:        {}",
                    match config.sampling.top_p {
                        Some(p) => format!("{}", p),
                        None => "off".to_string(),
                    }
                );
                println!(
                    "  speculative:  {}",
                    if config.speculative {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                println!("  max-tokens:   {}", config.max_tokens);
                println!(
                    "  debug:        {}",
                    if config.debug { "enabled" } else { "disabled" }
                );
                println!(
                    "  system:       {}",
                    match config.system_prompt {
                        Some(ref s) => {
                            if s.len() > 60 {
                                format!("\"{}...\"", &s[..60])
                            } else {
                                format!("\"{}\"", s)
                            }
                        }
                        None => "none".to_string(),
                    }
                );
                continue;
            }
            "/context" => {
                let total_processed = processed_tokens.len();
                let pct = if max_seq_len > 0 {
                    (total_processed as f64 / max_seq_len as f64) * 100.0
                } else {
                    0.0
                };
                let remaining = max_seq_len.saturating_sub(total_processed);

                // Count message roles
                let system_count = history
                    .iter()
                    .filter(|m| m.role == ChatRole::System)
                    .count();
                let user_count = history.iter().filter(|m| m.role == ChatRole::User).count();
                let assistant_count = history
                    .iter()
                    .filter(|m| m.role == ChatRole::Assistant)
                    .count();
                let total_messages = history.len();

                println!("Context usage:");
                println!(
                    "  Processed tokens: {} / {} ({:.1}%)",
                    total_processed, max_seq_len, pct
                );
                println!(
                    "  Remaining:        {} tokens ({:.1}%)",
                    remaining,
                    if max_seq_len > 0 {
                        (remaining as f64 / max_seq_len as f64) * 100.0
                    } else {
                        0.0
                    }
                );
                println!(
                    "  Conversation:     {} message{}{}",
                    total_messages,
                    if total_messages == 1 { "" } else { "s" },
                    if total_messages > 0 {
                        format!(
                            " ({})",
                            [
                                if system_count > 0 {
                                    Some(format!("{} system", system_count))
                                } else {
                                    None
                                },
                                if user_count > 0 {
                                    Some(format!("{} user", user_count))
                                } else {
                                    None
                                },
                                if assistant_count > 0 {
                                    Some(format!("{} assistant", assistant_count))
                                } else {
                                    None
                                },
                            ]
                            .iter()
                            .filter_map(|s| s.as_ref().map(|s| s.as_str().to_string()))
                            .collect::<Vec<_>>()
                            .join(", ")
                        )
                    } else {
                        String::new()
                    }
                );
                println!("  Last response:    {} tokens", last_generation_len);
                println!("  Max response:     {} tokens", config.max_tokens);
                continue;
            }
            "/save" => {
                match arg {
                    Some(path) if !path.is_empty() => {
                        let messages: Vec<serde_json::Value> = history
                            .iter()
                            .map(|m| {
                                serde_json::json!({
                                    "role": match m.role {
                                        ChatRole::System => "system",
                                        ChatRole::User => "user",
                                        ChatRole::Assistant => "assistant",
                                    },
                                    "content": m.content,
                                })
                            })
                            .collect();
                        let doc = serde_json::json!({
                            "version": 1,
                            "messages": messages,
                            "config": serde_json::Value::Null,
                        });
                        match serde_json::to_string_pretty(&doc) {
                            Ok(json) => match fs::write(path, &json) {
                                Ok(()) => println!("Saved {} messages to {}", history.len(), path),
                                Err(e) => eprintln!("Error writing file '{}': {}", path, e),
                            },
                            Err(e) => eprintln!("Error serializing conversation: {}", e),
                        }
                    }
                    _ => eprintln!("Usage: /save <file>"),
                }
                continue;
            }
            "/fullsave" => {
                match arg {
                    Some(path) if !path.is_empty() => {
                        let messages: Vec<serde_json::Value> = history
                            .iter()
                            .map(|m| {
                                serde_json::json!({
                                    "role": match m.role {
                                        ChatRole::System => "system",
                                        ChatRole::User => "user",
                                        ChatRole::Assistant => "assistant",
                                    },
                                    "content": m.content,
                                })
                            })
                            .collect();
                        let config_json = serde_json::json!({
                            "temperature": config.sampling.temperature,
                            "top_k": config.sampling.top_k,
                            "top_p": config.sampling.top_p,
                            "speculative": config.speculative,
                            "max_tokens": config.max_tokens,
                            "debug": config.debug,
                        });
                        let doc = serde_json::json!({
                            "version": 1,
                            "messages": messages,
                            "config": config_json,
                        });
                        match serde_json::to_string_pretty(&doc) {
                            Ok(json) => match fs::write(path, &json) {
                                Ok(()) => println!(
                                    "Saved {} messages + config to {}",
                                    history.len(),
                                    path
                                ),
                                Err(e) => eprintln!("Error writing file '{}': {}", path, e),
                            },
                            Err(e) => eprintln!("Error serializing conversation: {}", e),
                        }
                    }
                    _ => eprintln!("Usage: /fullsave <file>"),
                }
                continue;
            }
            "/load" => {
                match arg {
                    Some(path) if !path.is_empty() => {
                        let data = match fs::read_to_string(path) {
                            Ok(d) => d,
                            Err(e) => {
                                eprintln!("Error reading file '{}': {}", path, e);
                                continue;
                            }
                        };
                        let doc: serde_json::Value = match serde_json::from_str(&data) {
                            Ok(v) => v,
                            Err(e) => {
                                eprintln!("Error parsing JSON from '{}': {}", path, e);
                                continue;
                            }
                        };

                        // Parse messages
                        let msgs = match doc.get("messages").and_then(|v| v.as_array()) {
                            Some(arr) => arr,
                            None => {
                                eprintln!(
                                    "Invalid save file: missing or invalid 'messages' array."
                                );
                                continue;
                            }
                        };

                        let mut new_history: Vec<ChatMessage> = Vec::new();
                        let mut parse_ok = true;
                        for (i, msg) in msgs.iter().enumerate() {
                            let role_str = match msg.get("role").and_then(|v| v.as_str()) {
                                Some(r) => r,
                                None => {
                                    eprintln!("Invalid message at index {}: missing 'role'.", i);
                                    parse_ok = false;
                                    break;
                                }
                            };
                            let content = match msg.get("content").and_then(|v| v.as_str()) {
                                Some(c) => c,
                                None => {
                                    eprintln!("Invalid message at index {}: missing 'content'.", i);
                                    parse_ok = false;
                                    break;
                                }
                            };
                            let role = match role_str {
                                "system" => ChatRole::System,
                                "user" => ChatRole::User,
                                "assistant" => ChatRole::Assistant,
                                other => {
                                    eprintln!(
                                        "Invalid message at index {}: unknown role '{}'.",
                                        i, other
                                    );
                                    parse_ok = false;
                                    break;
                                }
                            };
                            new_history.push(ChatMessage {
                                role,
                                content: content.to_string(),
                            });
                        }
                        if !parse_ok {
                            continue;
                        }

                        // Apply loaded messages
                        history = new_history;
                        state.reset();
                        processed_tokens.clear();
                        last_generation_len = 0;

                        // Check for and apply config if present
                        let has_config = doc.get("config").is_some_and(|v| !v.is_null());
                        if has_config {
                            if let Some(cfg) = doc.get("config") {
                                if let Some(t) = cfg.get("temperature").and_then(|v| v.as_f64()) {
                                    config.sampling.temperature = t as f32;
                                }
                                if let Some(k) = cfg.get("top_k") {
                                    if k.is_null() {
                                        config.sampling.top_k = None;
                                    } else if let Some(k_val) = k.as_u64() {
                                        config.sampling.top_k = Some(k_val as usize);
                                    }
                                }
                                if let Some(p) = cfg.get("top_p") {
                                    if p.is_null() {
                                        config.sampling.top_p = None;
                                    } else if let Some(p_val) = p.as_f64() {
                                        config.sampling.top_p = Some(p_val as f32);
                                    }
                                }
                                if let Some(s) = cfg.get("speculative").and_then(|v| v.as_bool()) {
                                    config.speculative = s;
                                }
                                if let Some(mt) = cfg.get("max_tokens").and_then(|v| v.as_u64()) {
                                    config.max_tokens = mt as usize;
                                }
                                if let Some(d) = cfg.get("debug").and_then(|v| v.as_bool()) {
                                    config.debug = d;
                                }
                            }
                        }

                        // Update system_prompt from loaded history
                        config.system_prompt = history.first().and_then(|m| {
                            if m.role == ChatRole::System {
                                Some(m.content.clone())
                            } else {
                                None
                            }
                        });

                        println!(
                            "Loaded {} messages from {}{}",
                            history.len(),
                            path,
                            if has_config {
                                " (config restored)"
                            } else {
                                " (config not included)"
                            }
                        );
                    }
                    _ => eprintln!("Usage: /load <file>"),
                }
                continue;
            }
            "/retry" => {
                // Check that the last message is an assistant message
                match history.last() {
                    Some(msg) if msg.role == ChatRole::Assistant => {}
                    _ => {
                        eprintln!("Nothing to retry: last message is not an assistant response.");
                        continue;
                    }
                }
                if last_generation_len == 0 {
                    eprintln!("Nothing to retry: no previous generation to roll back.");
                    continue;
                }
                // Roll back: remove last assistant message, rewind state and token tracking
                history.pop();
                let rollback = last_generation_len;
                state.pos = state.pos.saturating_sub(rollback);
                processed_tokens.truncate(processed_tokens.len().saturating_sub(rollback));
                println!("Rolling back {} tokens and regenerating...", rollback);
                skip_processing = true;
                // Don't continue -- fall through to generation
            }
            "/help" => {
                println!("Commands:");
                println!("  /quit, /exit, /q   Exit chat");
                println!("  /clear             Clear conversation history");
                println!("  /retry             Regenerate last assistant response");
                println!("  /thinking [on|off] Toggle or set thinking trace visibility");
                println!(
                    "  /temperature <T>   Set sampling temperature (0.0 = greedy, 1.0+ = creative)"
                );
                println!("  /top-k <K|off>     Set or disable top-k sampling");
                println!("  /top-p <P|off>     Set or disable nucleus (top-p) sampling");
                println!("  /speculative       Toggle self-speculative decoding");
                println!("  /max-tokens <N>    Set max tokens per response");
                println!("  /system <MSG|off>  Set, change, or remove the system prompt");
                println!("  /debug             Toggle debug output");
                println!("  /settings          Show all current settings");
                println!("  /context           Show token usage and context headroom");
                println!("  /save <file>       Save conversation history to JSON file");
                println!("  /fullsave <file>   Save conversation + settings to JSON file");
                println!(
                    "  /load <file>       Load conversation (and optional settings) from JSON file"
                );
                println!("  /help              Show this help");
                println!();
                println!("Anything else is sent as a message to the model.");
                continue;
            }
            _ if cmd.starts_with('/') => {
                eprintln!(
                    "Unknown command '{}'. Type /help for available commands.",
                    cmd
                );
                continue;
            }
            _ => {}
        }

        if !skip_processing {
            // Add user message to history
            history.push(ChatMessage::user(&input));

            // Format the full conversation as a prompt
            let prompt_str = chat_template.format_prompt(&history);

            // Tokenize the full prompt
            let full_tokens = tokenizer.encode(&prompt_str);

            // Check context window limits and trim if needed
            if full_tokens.len() >= max_seq_len {
                let trimmed = chat_template.trim_to_fit(
                    &mut history,
                    tokenizer,
                    max_seq_len - config.max_tokens, // reserve room for generation
                );
                if trimmed {
                    if config.debug {
                        eprintln!("Context trimmed; resetting KV cache.");
                    }
                    state.reset();
                    processed_tokens.clear();
                    // Re-tokenize after trimming
                    let prompt_str = chat_template.format_prompt(&history);
                    let full_tokens = tokenizer.encode(&prompt_str);
                    // Process all tokens from scratch
                    process_tokens(&mut state, &full_tokens, config.debug);
                    processed_tokens = full_tokens;
                }
            } else {
                // Determine how many tokens are new (delta from last time)
                let new_start = processed_tokens.len();

                if new_start == 0 {
                    // First turn: process all prompt tokens
                    process_tokens(&mut state, &full_tokens, config.debug);
                } else {
                    // Subsequent turn: only process the new tokens since last generation
                    // We need to process from where we left off
                    let new_tokens = &full_tokens[new_start..];
                    if config.debug {
                        eprintln!(
                            "KV cache reuse: {} cached, {} new tokens to process",
                            new_start,
                            new_tokens.len()
                        );
                    }
                    process_tokens(&mut state, new_tokens, config.debug);
                }

                processed_tokens = full_tokens;
            }
        }

        // The last token of the prompt is used to kick off generation
        let last_prompt_token = *processed_tokens.last().unwrap();

        // Reset thinking state for the new generation turn
        thinking_state.reset();

        // Generate the response
        let result = generate_response(&mut state, last_prompt_token, &eos_ids, &config);

        println!();

        // Track length of this generation for /context and /retry
        last_generation_len = result.tokens.len();

        // Collect the assistant's response text, stripping thinking traces
        // from history so subsequent turns don't waste context on them.
        let raw_response = tokenizer.decode(&result.tokens);
        let response_text = if thinking_state.is_thinking_model() {
            strip_thinking(&raw_response)
        } else {
            raw_response
        };

        if config.debug {
            if result.stopped_at_eos {
                eprintln!("[stopped at EOS]");
            } else {
                eprintln!("[stopped at max_tokens ({})]", config.max_tokens);
            }
        }

        // Add assistant response to history (thinking stripped)
        history.push(ChatMessage::assistant(response_text));

        // Update processed_tokens to include the generated tokens
        // (so next turn can compute the correct delta)
        processed_tokens.extend_from_slice(&result.tokens);

        println!();
    }

    Ok(())
}
