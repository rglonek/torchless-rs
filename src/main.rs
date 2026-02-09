use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use torchless::tokenizer::Tokenizer;
use torchless::{
    apply_edit, coding_system_prompt, display_thinking_token_to, expand_file_references,
    format_edit_diff, generate, generate_lazy, generate_lazy_until_eos, generate_until_eos,
    init_backend, parse_edit_blocks, print_backend_summary, strip_thinking, BackendPreference,
    ChatMessage, ChatRole, ChatTemplate, GenerationResult, InferenceState, KVDtype, LazyMistral,
    Mistral, Parameters, PendingEdit, SamplingConfig, SelfSpeculativeDecoder, SpeculativeConfig,
    ThinkingState,
};

#[cfg(unix)]
mod server;

/// Mutable session configuration for the chat REPL.
/// All fields can be changed at runtime via in-session commands.
pub(crate) struct ChatSessionConfig {
    pub(crate) sampling: SamplingConfig,
    pub(crate) speculative: bool,
    pub(crate) max_tokens: usize,
    pub(crate) debug: bool,
    pub(crate) system_prompt: Option<String>,
}

fn print_usage(program: &str) {
    eprintln!("Usage: {} [OPTIONS] <model_path> [prompt]", program);
    eprintln!();
    eprintln!("Options:");
    eprintln!(
        "  --backend <BACKEND>   Compute backend: auto, cpu, cuda, rocm, metal, opencl, webgpu"
    );
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
    eprintln!("  --socket <PATH>       Start a Unix socket server at PATH (multi-user chat)");
    eprintln!(
        "  --chat-save-root <DIR> Root directory for per-user chat saves (required with --socket)"
    );
    eprintln!("  --list-backends       List available backends and exit");
    eprintln!("  --kv-dtype <TYPE>     KV cache precision: f16 (default, half memory) or f32");
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
        #[cfg(feature = "webgpu")]
        "webgpu" => Some(BackendPreference::WebGPU),
        _ => None,
    }
}

/// Resolve max_seq_len: default to model's max_position_embeddings, warn if user exceeds it.
pub(crate) fn resolve_max_seq_len(user_value: Option<usize>, model_max: usize) -> usize {
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
pub(crate) fn resolve_max_tokens(user_value: Option<usize>, max_seq_len: usize) -> usize {
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
    let mut kv_dtype = KVDtype::F16; // default to FP16
    let mut show_thinking = false;
    let mut list_backends = false;
    let mut chat_mode = false;
    let mut system_prompt: Option<String> = None;
    let mut socket_path: Option<String> = None;
    let mut chat_save_root: Option<String> = None;
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
            "--socket" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: --socket requires a path");
                    std::process::exit(1);
                }
                socket_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--chat-save-root" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: --chat-save-root requires a directory path");
                    std::process::exit(1);
                }
                chat_save_root = Some(args[i + 1].clone());
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
                        #[cfg(feature = "webgpu")]
                        eprint!(", webgpu");
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
            "--kv-dtype" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: --kv-dtype requires a value (f16 or f32)");
                    std::process::exit(1);
                }
                kv_dtype = match args[i + 1].to_lowercase().as_str() {
                    "f16" | "fp16" => KVDtype::F16,
                    "f32" | "fp32" => KVDtype::F32,
                    _ => {
                        eprintln!(
                            "Error: invalid --kv-dtype value '{}'. Use f16 or f32",
                            args[i + 1]
                        );
                        std::process::exit(1);
                    }
                };
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

    // Socket server mode
    #[cfg(unix)]
    if let Some(ref sock_path) = socket_path {
        let save_root = match chat_save_root {
            Some(ref dir) => dir.clone(),
            None => {
                eprintln!("Error: --chat-save-root is required when using --socket");
                std::process::exit(1);
            }
        };
        if positional_args.is_empty() {
            eprintln!("Error: <model_path> is required");
            print_usage(program);
            std::process::exit(1);
        }
        let model_path = &positional_args[0];
        return server::run_socket_server(
            sock_path,
            &save_root,
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
            kv_dtype,
        );
    }

    #[cfg(not(unix))]
    {
        // chat_save_root is also a socket-server-only option (Unix-only)
        let _ = &chat_save_root;
        if socket_path.is_some() {
            eprintln!("Error: --socket and --chat-save-root are only supported on Unix systems");
            std::process::exit(1);
        }
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
            kv_dtype,
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
        let mut state = InferenceState::with_seq_len(model.config.clone(), max_seq_len, kv_dtype);

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
                &mut io::stdout(),
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
        let mut state = InferenceState::with_seq_len(model.config.clone(), max_seq_len, kv_dtype);

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
                &mut io::stdout(),
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
pub(crate) fn generate_speculative_until_eos<'a, F>(
    forward: F,
    tokenizer: &Tokenizer,
    state: &mut InferenceState,
    first_token: u32,
    temperature: f32,
    max_tokens: usize,
    eos_token_ids: &[u32],
    thinking: Option<&ThinkingState>,
    debug: bool,
    output: &mut dyn Write,
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
                display_thinking_token_to(output, action, || tokenizer.decode(&[t]));
            } else {
                let text = tokenizer.decode(&[t]);
                let _ = write!(output, "{}", text);
            }
            let _ = output.flush();
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

/// Sanitize a filename: allow only alphanumeric, underscore, hyphen, space, dot.
/// Returns None if the name is invalid or contains path traversal attempts.
fn sanitize_filename(name: &str) -> Option<&str> {
    let name = name.trim();
    if name.is_empty() || name.len() > 255 {
        return None;
    }
    if name.contains("..") || name.contains('/') || name.contains('\\') {
        return None;
    }
    if name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | ' ' | '.'))
    {
        Some(name)
    } else {
        None
    }
}

/// Resolve a save/load file path. In socket mode (`save_root` is `Some`), the filename
/// is resolved relative to `save_root` with sanitization. In CLI mode, the path is used as-is.
fn resolve_save_path(save_root: Option<&Path>, arg: &str) -> Result<PathBuf, String> {
    match save_root {
        Some(root) => {
            let name = sanitize_filename(arg).ok_or_else(|| {
                format!(
                    "Invalid filename '{}'. Only letters, numbers, underscore, hyphen, space, dot allowed.",
                    arg
                )
            })?;
            let mut filename = name.to_string();
            if !filename.ends_with(".json") {
                filename.push_str(".json");
            }
            Ok(root.join(filename))
        }
        None => Ok(PathBuf::from(arg)),
    }
}

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
    kv_dtype: KVDtype,
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

        let state = InferenceState::with_seq_len(model.config.clone(), max_seq_len, kv_dtype);

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

        let stdin = io::stdin();
        let mut stdin_lock = stdin.lock();
        let mut stdout = io::stdout();
        run_chat_repl(
            state,
            &model.tokenizer,
            config,
            max_seq_len,
            &thinking_state,
            None,
            &mut stdin_lock,
            &mut stdout,
            &mut |state, tokens, debug| {
                for (i, &token) in tokens.iter().enumerate() {
                    if debug && (i % 50 == 0) {
                        eprintln!("Processing prompt token {}/{}", i + 1, tokens.len());
                    }
                    model.forward(state, token, debug);
                    state.pos += 1;
                }
            },
            &mut |state, first_token, eos_ids, config, output| {
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
                        output,
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
                            display_thinking_token_to(output, action, || {
                                model.tokenizer.decode(&[token_id])
                            });
                            let _ = output.flush();
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

        let state = InferenceState::with_seq_len(model.config.clone(), max_seq_len, kv_dtype);

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

        let stdin = io::stdin();
        let mut stdin_lock = stdin.lock();
        let mut stdout = io::stdout();
        run_chat_repl(
            state,
            &model.tokenizer,
            config,
            max_seq_len,
            &thinking_state,
            None,
            &mut stdin_lock,
            &mut stdout,
            &mut |state, tokens, debug| {
                for (i, &token) in tokens.iter().enumerate() {
                    if debug && (i % 50 == 0) {
                        eprintln!("Processing prompt token {}/{}", i + 1, tokens.len());
                    }
                    model.forward(state, token, debug);
                    state.pos += 1;
                }
            },
            &mut |state, first_token, eos_ids, config, output| {
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
                        output,
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
                            display_thinking_token_to(output, action, || {
                                model.tokenizer.decode(&[token_id])
                            });
                            let _ = output.flush();
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
/// `save_root` is an optional root directory for save/load operations (socket mode uses per-user dirs).
/// `input` is the input stream (stdin for CLI, socket for server).
/// `output` is the output stream (stdout for CLI, socket for server).
/// `process_tokens` processes prompt tokens through the model (advancing state.pos).
/// `generate_response` generates a response from a starting token, returning the result.
/// `thinking_state` tracks thinking model output filtering (auto-detected; no-op if not a thinking model).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn run_chat_repl(
    mut state: InferenceState,
    tokenizer: &Tokenizer,
    mut config: ChatSessionConfig,
    max_seq_len: usize,
    thinking_state: &ThinkingState,
    save_root: Option<&Path>,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    process_tokens: &mut dyn FnMut(&mut InferenceState, &[u32], bool),
    generate_response: &mut dyn FnMut(
        &mut InferenceState,
        u32,
        &[u32],
        &ChatSessionConfig,
        &mut dyn Write,
    ) -> GenerationResult,
) -> anyhow::Result<()> {
    // Select chat template (default to Mistral for now)
    let chat_template = ChatTemplate::Mistral;
    let eos_ids = chat_template.eos_token_ids(tokenizer);

    if eos_ids.is_empty() {
        writeln!(
            output,
            "Warning: no EOS tokens found in vocabulary for {:?} template. \
             Generation will only stop at --max-tokens ({}).",
            chat_template, config.max_tokens
        )?;
    }

    writeln!(output, "Initializing inference state...")?;

    // Build initial conversation history
    let mut history: Vec<ChatMessage> = Vec::new();
    if let Some(ref sys) = config.system_prompt {
        history.push(ChatMessage::system(sys));
    }

    // Track how many tokens have been processed so far (for KV cache reuse)
    let mut processed_tokens: Vec<u32> = Vec::new();

    // Track length of the last assistant generation (for /context and /retry)
    let mut last_generation_len: usize = 0;

    // Coding mode state
    let mut coding_mode = false;
    let mut pending_edits: Vec<PendingEdit> = Vec::new();
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    writeln!(output)?;
    writeln!(output, "Chat mode active. Type /help for commands.")?;
    writeln!(output)?;

    loop {
        // Print prompt
        write!(output, "> ")?;
        output.flush()?;

        // Read user input
        let mut line = String::new();
        match input.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                let _ = writeln!(output, "Error reading input: {}", e);
                break;
            }
        }

        let user_input = line.trim().to_string();

        if user_input.is_empty() {
            continue;
        }

        // Flag for /retry: skip message processing, go straight to generation
        let mut skip_processing = false;

        // Handle commands: split on first space for arg parsing
        let (cmd, arg) = match user_input.find(' ') {
            Some(pos) => (&user_input[..pos], Some(user_input[pos + 1..].trim())),
            None => (user_input.as_str(), None),
        };

        match cmd {
            "/quit" | "/exit" | "/q" => {
                writeln!(output, "Goodbye!")?;
                break;
            }
            "/clear" => {
                history.clear();
                if let Some(ref sys) = config.system_prompt {
                    history.push(ChatMessage::system(sys));
                }
                state.reset();
                processed_tokens.clear();
                writeln!(output, "Conversation cleared.")?;
                continue;
            }
            "/thinking" => {
                match arg {
                    Some("on") => {
                        thinking_state.set_show(true);
                        writeln!(output, "Thinking display: on (traces shown dimmed)")?;
                    }
                    Some("off") => {
                        thinking_state.set_show(false);
                        writeln!(output, "Thinking display: off (traces hidden)")?;
                    }
                    None => {
                        let new = thinking_state.toggle_show();
                        writeln!(
                            output,
                            "Thinking display: {}",
                            if new {
                                "on (traces shown dimmed)"
                            } else {
                                "off (traces hidden)"
                            }
                        )?;
                    }
                    Some(other) => {
                        writeln!(
                            output,
                            "Invalid argument '{}'. Usage: /thinking [on|off]",
                            other
                        )?;
                    }
                }
                continue;
            }
            "/temperature" => {
                match arg {
                    Some(val) => match val.parse::<f32>() {
                        Ok(t) if t >= 0.0 => {
                            config.sampling.temperature = t;
                            writeln!(output, "Temperature set to {}", t)?;
                        }
                        Ok(_) => writeln!(output, "Temperature must be >= 0.0")?,
                        Err(_) => writeln!(
                            output,
                            "Invalid temperature value '{}'. Expected a number.",
                            val
                        )?,
                    },
                    None => writeln!(
                        output,
                        "Usage: /temperature <value>  (current: {})",
                        config.sampling.temperature
                    )?,
                }
                continue;
            }
            "/top-k" => {
                match arg {
                    Some("off") | Some("none") => {
                        config.sampling.top_k = None;
                        writeln!(output, "Top-k sampling disabled.")?;
                    }
                    Some(val) => match val.parse::<usize>() {
                        Ok(k) if k > 0 => {
                            config.sampling.top_k = Some(k);
                            writeln!(output, "Top-k set to {}", k)?;
                            if config.speculative {
                                writeln!(output, "Warning: top-k is ignored in speculative mode.")?;
                            }
                        }
                        Ok(_) => {
                            writeln!(output, "Top-k must be > 0. Use '/top-k off' to disable.")?
                        }
                        Err(_) => writeln!(
                            output,
                            "Invalid top-k value '{}'. Expected a number or 'off'.",
                            val
                        )?,
                    },
                    None => {
                        let current = match config.sampling.top_k {
                            Some(k) => format!("{}", k),
                            None => "off".to_string(),
                        };
                        writeln!(output, "Usage: /top-k <value|off>  (current: {})", current)?;
                    }
                }
                continue;
            }
            "/top-p" => {
                match arg {
                    Some("off") | Some("none") => {
                        config.sampling.top_p = None;
                        writeln!(output, "Top-p (nucleus) sampling disabled.")?;
                    }
                    Some(val) => match val.parse::<f32>() {
                        Ok(p) if (0.0..=1.0).contains(&p) => {
                            config.sampling.top_p = Some(p);
                            writeln!(output, "Top-p set to {}", p)?;
                            if config.speculative {
                                writeln!(output, "Warning: top-p is ignored in speculative mode.")?;
                            }
                        }
                        Ok(_) => writeln!(
                            output,
                            "Top-p must be between 0.0 and 1.0. Use '/top-p off' to disable."
                        )?,
                        Err(_) => writeln!(
                            output,
                            "Invalid top-p value '{}'. Expected a number or 'off'.",
                            val
                        )?,
                    },
                    None => {
                        let current = match config.sampling.top_p {
                            Some(p) => format!("{}", p),
                            None => "off".to_string(),
                        };
                        writeln!(output, "Usage: /top-p <value|off>  (current: {})", current)?;
                    }
                }
                continue;
            }
            "/speculative" => {
                config.speculative = !config.speculative;
                writeln!(
                    output,
                    "Speculative decoding: {}",
                    if config.speculative {
                        "enabled"
                    } else {
                        "disabled"
                    }
                )?;
                if config.speculative
                    && (config.sampling.top_k.is_some() || config.sampling.top_p.is_some())
                {
                    writeln!(
                        output,
                        "Warning: top-k and top-p are ignored in speculative mode. \
                         Speculative decoding uses its own temperature-based sampling."
                    )?;
                }
                continue;
            }
            "/max-tokens" => {
                match arg {
                    Some(val) => match val.parse::<usize>() {
                        Ok(n) if n > 0 => {
                            config.max_tokens = n;
                            writeln!(output, "Max tokens per response set to {}", n)?;
                        }
                        Ok(_) => writeln!(output, "Max tokens must be > 0.")?,
                        Err(_) => writeln!(
                            output,
                            "Invalid max-tokens value '{}'. Expected a number.",
                            val
                        )?,
                    },
                    None => writeln!(
                        output,
                        "Usage: /max-tokens <value>  (current: {})",
                        config.max_tokens
                    )?,
                }
                continue;
            }
            "/debug" => {
                config.debug = !config.debug;
                writeln!(
                    output,
                    "Debug output: {}",
                    if config.debug { "enabled" } else { "disabled" }
                )?;
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
                        writeln!(output, "System prompt removed. KV cache reset.")?;
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
                        writeln!(output, "System prompt updated. KV cache reset.")?;
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
                        writeln!(
                            output,
                            "Usage: /system <message>  or  /system off  (current: {})",
                            current
                        )?;
                    }
                }
                continue;
            }
            "/settings" => {
                writeln!(output, "Current settings:")?;
                writeln!(output, "  temperature:  {}", config.sampling.temperature)?;
                writeln!(
                    output,
                    "  top-k:        {}",
                    match config.sampling.top_k {
                        Some(k) => format!("{}", k),
                        None => "off".to_string(),
                    }
                )?;
                writeln!(
                    output,
                    "  top-p:        {}",
                    match config.sampling.top_p {
                        Some(p) => format!("{}", p),
                        None => "off".to_string(),
                    }
                )?;
                writeln!(
                    output,
                    "  speculative:  {}",
                    if config.speculative {
                        "enabled"
                    } else {
                        "disabled"
                    }
                )?;
                writeln!(output, "  max-tokens:   {}", config.max_tokens)?;
                writeln!(
                    output,
                    "  debug:        {}",
                    if config.debug { "enabled" } else { "disabled" }
                )?;
                writeln!(
                    output,
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
                )?;
                writeln!(
                    output,
                    "  coding:       {}{}",
                    if coding_mode { "enabled" } else { "disabled" },
                    if !pending_edits.is_empty() {
                        format!(" ({} pending edit(s))", pending_edits.len())
                    } else {
                        String::new()
                    }
                )?;
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

                writeln!(output, "Context usage:")?;
                writeln!(
                    output,
                    "  Processed tokens: {} / {} ({:.1}%)",
                    total_processed, max_seq_len, pct
                )?;
                writeln!(
                    output,
                    "  Remaining:        {} tokens ({:.1}%)",
                    remaining,
                    if max_seq_len > 0 {
                        (remaining as f64 / max_seq_len as f64) * 100.0
                    } else {
                        0.0
                    }
                )?;
                writeln!(
                    output,
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
                )?;
                writeln!(output, "  Last response:    {} tokens", last_generation_len)?;
                writeln!(output, "  Max response:     {} tokens", config.max_tokens)?;
                continue;
            }
            "/save" => {
                match arg {
                    Some(name) if !name.is_empty() => {
                        let resolved = match resolve_save_path(save_root, name) {
                            Ok(p) => p,
                            Err(e) => {
                                writeln!(output, "{}", e)?;
                                continue;
                            }
                        };
                        if let Some(parent) = resolved.parent() {
                            let _ = fs::create_dir_all(parent);
                        }
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
                            Ok(json) => match fs::write(&resolved, &json) {
                                Ok(()) => writeln!(
                                    output,
                                    "Saved {} messages to {}",
                                    history.len(),
                                    resolved.display()
                                )?,
                                Err(e) => writeln!(
                                    output,
                                    "Error writing file '{}': {}",
                                    resolved.display(),
                                    e
                                )?,
                            },
                            Err(e) => writeln!(output, "Error serializing conversation: {}", e)?,
                        }
                    }
                    _ => writeln!(output, "Usage: /save <file>")?,
                }
                continue;
            }
            "/fullsave" => {
                match arg {
                    Some(name) if !name.is_empty() => {
                        let resolved = match resolve_save_path(save_root, name) {
                            Ok(p) => p,
                            Err(e) => {
                                writeln!(output, "{}", e)?;
                                continue;
                            }
                        };
                        if let Some(parent) = resolved.parent() {
                            let _ = fs::create_dir_all(parent);
                        }
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
                            Ok(json) => match fs::write(&resolved, &json) {
                                Ok(()) => writeln!(
                                    output,
                                    "Saved {} messages + config to {}",
                                    history.len(),
                                    resolved.display()
                                )?,
                                Err(e) => writeln!(
                                    output,
                                    "Error writing file '{}': {}",
                                    resolved.display(),
                                    e
                                )?,
                            },
                            Err(e) => writeln!(output, "Error serializing conversation: {}", e)?,
                        }
                    }
                    _ => writeln!(output, "Usage: /fullsave <file>")?,
                }
                continue;
            }
            "/load" => {
                match arg {
                    Some(name) if !name.is_empty() => {
                        let resolved = match resolve_save_path(save_root, name) {
                            Ok(p) => p,
                            Err(e) => {
                                writeln!(output, "{}", e)?;
                                continue;
                            }
                        };
                        let data = match fs::read_to_string(&resolved) {
                            Ok(d) => d,
                            Err(e) => {
                                writeln!(
                                    output,
                                    "Error reading file '{}': {}",
                                    resolved.display(),
                                    e
                                )?;
                                continue;
                            }
                        };
                        let doc: serde_json::Value = match serde_json::from_str(&data) {
                            Ok(v) => v,
                            Err(e) => {
                                writeln!(
                                    output,
                                    "Error parsing JSON from '{}': {}",
                                    resolved.display(),
                                    e
                                )?;
                                continue;
                            }
                        };

                        // Parse messages
                        let msgs = match doc.get("messages").and_then(|v| v.as_array()) {
                            Some(arr) => arr,
                            None => {
                                writeln!(
                                    output,
                                    "Invalid save file: missing or invalid 'messages' array."
                                )?;
                                continue;
                            }
                        };

                        let mut new_history: Vec<ChatMessage> = Vec::new();
                        let mut parse_ok = true;
                        for (i, msg) in msgs.iter().enumerate() {
                            let role_str = match msg.get("role").and_then(|v| v.as_str()) {
                                Some(r) => r,
                                None => {
                                    writeln!(
                                        output,
                                        "Invalid message at index {}: missing 'role'.",
                                        i
                                    )?;
                                    parse_ok = false;
                                    break;
                                }
                            };
                            let content = match msg.get("content").and_then(|v| v.as_str()) {
                                Some(c) => c,
                                None => {
                                    writeln!(
                                        output,
                                        "Invalid message at index {}: missing 'content'.",
                                        i
                                    )?;
                                    parse_ok = false;
                                    break;
                                }
                            };
                            let role = match role_str {
                                "system" => ChatRole::System,
                                "user" => ChatRole::User,
                                "assistant" => ChatRole::Assistant,
                                other => {
                                    writeln!(
                                        output,
                                        "Invalid message at index {}: unknown role '{}'.",
                                        i, other
                                    )?;
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

                        writeln!(
                            output,
                            "Loaded {} messages from {}{}",
                            history.len(),
                            resolved.display(),
                            if has_config {
                                " (config restored)"
                            } else {
                                " (config not included)"
                            }
                        )?;
                    }
                    _ => writeln!(output, "Usage: /load <file>")?,
                }
                continue;
            }
            "/retry" => {
                // Check that the last message is an assistant message
                match history.last() {
                    Some(msg) if msg.role == ChatRole::Assistant => {}
                    _ => {
                        writeln!(
                            output,
                            "Nothing to retry: last message is not an assistant response."
                        )?;
                        continue;
                    }
                }
                if last_generation_len == 0 {
                    writeln!(
                        output,
                        "Nothing to retry: no previous generation to roll back."
                    )?;
                    continue;
                }
                // Roll back: remove last assistant message, rewind state and token tracking
                history.pop();
                let rollback = last_generation_len;
                state.pos = state.pos.saturating_sub(rollback);
                processed_tokens.truncate(processed_tokens.len().saturating_sub(rollback));
                writeln!(
                    output,
                    "Rolling back {} tokens and regenerating...",
                    rollback
                )?;
                skip_processing = true;
                // Don't continue -- fall through to generation
            }
            "/code" => {
                match arg {
                    Some("on") => {
                        if !coding_mode {
                            coding_mode = true;
                            // Append coding instructions to the system prompt
                            let addendum = coding_system_prompt();
                            match config.system_prompt {
                                Some(ref mut sys) => {
                                    sys.push_str(addendum);
                                }
                                None => {
                                    config.system_prompt = Some(addendum.trim_start().to_string());
                                }
                            }
                            // Update history
                            if !history.is_empty() && history[0].role == ChatRole::System {
                                history[0] =
                                    ChatMessage::system(config.system_prompt.as_deref().unwrap());
                            } else {
                                history.insert(
                                    0,
                                    ChatMessage::system(config.system_prompt.as_deref().unwrap()),
                                );
                            }
                            state.reset();
                            processed_tokens.clear();
                            writeln!(
                                output,
                                "Coding mode: on. Use @file references and the model will propose edits. KV cache reset."
                            )?;
                        } else {
                            writeln!(output, "Coding mode is already on.")?;
                        }
                    }
                    Some("off") => {
                        if coding_mode {
                            coding_mode = false;
                            // Remove coding addendum from system prompt
                            let addendum = coding_system_prompt();
                            if let Some(ref mut sys) = config.system_prompt {
                                if let Some(pos) = sys.find(addendum) {
                                    sys.replace_range(pos..pos + addendum.len(), "");
                                }
                                if sys.trim().is_empty() {
                                    config.system_prompt = None;
                                    if !history.is_empty() && history[0].role == ChatRole::System {
                                        history.remove(0);
                                    }
                                } else if !history.is_empty() && history[0].role == ChatRole::System
                                {
                                    history[0] = ChatMessage::system(sys.as_str());
                                }
                            }
                            state.reset();
                            processed_tokens.clear();
                            writeln!(output, "Coding mode: off. KV cache reset.")?;
                        } else {
                            writeln!(output, "Coding mode is already off.")?;
                        }
                    }
                    None => {
                        // Toggle
                        if coding_mode {
                            // Simulate /code off
                            coding_mode = false;
                            let addendum = coding_system_prompt();
                            if let Some(ref mut sys) = config.system_prompt {
                                if let Some(pos) = sys.find(addendum) {
                                    sys.replace_range(pos..pos + addendum.len(), "");
                                }
                                if sys.trim().is_empty() {
                                    config.system_prompt = None;
                                    if !history.is_empty() && history[0].role == ChatRole::System {
                                        history.remove(0);
                                    }
                                } else if !history.is_empty() && history[0].role == ChatRole::System
                                {
                                    history[0] = ChatMessage::system(sys.as_str());
                                }
                            }
                            state.reset();
                            processed_tokens.clear();
                            writeln!(output, "Coding mode: off. KV cache reset.")?;
                        } else {
                            coding_mode = true;
                            let addendum = coding_system_prompt();
                            match config.system_prompt {
                                Some(ref mut sys) => {
                                    sys.push_str(addendum);
                                }
                                None => {
                                    config.system_prompt = Some(addendum.trim_start().to_string());
                                }
                            }
                            if !history.is_empty() && history[0].role == ChatRole::System {
                                history[0] =
                                    ChatMessage::system(config.system_prompt.as_deref().unwrap());
                            } else {
                                history.insert(
                                    0,
                                    ChatMessage::system(config.system_prompt.as_deref().unwrap()),
                                );
                            }
                            state.reset();
                            processed_tokens.clear();
                            writeln!(
                                output,
                                "Coding mode: on. Use @file references and the model will propose edits. KV cache reset."
                            )?;
                        }
                    }
                    Some(other) => {
                        writeln!(
                            output,
                            "Invalid argument '{}'. Usage: /code [on|off]",
                            other
                        )?;
                    }
                }
                continue;
            }
            "/diff" => {
                if pending_edits.is_empty() {
                    writeln!(output, "No pending edits.")?;
                } else {
                    writeln!(output, "{} pending edit(s):\n", pending_edits.len())?;
                    for (i, edit) in pending_edits.iter().enumerate() {
                        write!(output, "{}", format_edit_diff(edit, i))?;
                    }
                }
                continue;
            }
            "/apply" => {
                if pending_edits.is_empty() {
                    writeln!(output, "No pending edits to apply.")?;
                    continue;
                }

                // Parse argument: optional index (1-based) or "all"
                let indices: Vec<usize> = match arg {
                    None | Some("all") => (0..pending_edits.len()).collect(),
                    Some(n) => match n.parse::<usize>() {
                        Ok(idx) if idx >= 1 && idx <= pending_edits.len() => {
                            vec![idx - 1]
                        }
                        _ => {
                            writeln!(
                                output,
                                "Usage: /apply [all|N]  (N is 1-{})",
                                pending_edits.len()
                            )?;
                            continue;
                        }
                    },
                };

                let mut applied = 0usize;
                let mut skipped = 0usize;
                let mut failed = 0usize;
                let total = indices.len();
                let mut applied_indices: Vec<usize> = Vec::new();

                for (seq, &idx) in indices.iter().enumerate() {
                    let edit = &pending_edits[idx];
                    write!(output, "{}", format_edit_diff(edit, idx))?;
                    write!(output, "Apply edit {}/{}? [y/N]: ", seq + 1, total)?;
                    output.flush()?;

                    let mut confirm = String::new();
                    match input.read_line(&mut confirm) {
                        Ok(0) => break,
                        Ok(_) => {}
                        Err(_) => break,
                    }

                    if confirm.trim().eq_ignore_ascii_case("y") {
                        match apply_edit(edit, &cwd) {
                            Ok(()) => {
                                writeln!(output, "Applied.")?;
                                applied += 1;
                                applied_indices.push(idx);
                            }
                            Err(e) => {
                                writeln!(output, "Error: {}", e)?;
                                failed += 1;
                            }
                        }
                    } else {
                        writeln!(output, "Skipped.")?;
                        skipped += 1;
                    }
                }

                // Remove applied edits (in reverse order to preserve indices)
                applied_indices.sort_unstable();
                for &idx in applied_indices.iter().rev() {
                    pending_edits.remove(idx);
                }

                if total > 1 || failed > 0 {
                    writeln!(
                        output,
                        "Summary: {} applied, {} skipped, {} failed. {} pending.",
                        applied,
                        skipped,
                        failed,
                        pending_edits.len()
                    )?;
                }
                continue;
            }
            "/discard" => {
                if pending_edits.is_empty() {
                    writeln!(output, "No pending edits to discard.")?;
                    continue;
                }

                match arg {
                    None | Some("all") => {
                        let count = pending_edits.len();
                        pending_edits.clear();
                        writeln!(output, "Discarded {} edit(s).", count)?;
                    }
                    Some(n) => match n.parse::<usize>() {
                        Ok(idx) if idx >= 1 && idx <= pending_edits.len() => {
                            pending_edits.remove(idx - 1);
                            writeln!(
                                output,
                                "Discarded edit {}. {} remaining.",
                                idx,
                                pending_edits.len()
                            )?;
                        }
                        _ => {
                            writeln!(
                                output,
                                "Usage: /discard [all|N]  (N is 1-{})",
                                pending_edits.len()
                            )?;
                        }
                    },
                }
                continue;
            }
            "/help" => {
                writeln!(output, "Commands:")?;
                writeln!(output, "  /quit, /exit, /q   Exit chat")?;
                writeln!(output, "  /clear             Clear conversation history")?;
                writeln!(
                    output,
                    "  /retry             Regenerate last assistant response"
                )?;
                writeln!(
                    output,
                    "  /thinking [on|off] Toggle or set thinking trace visibility"
                )?;
                writeln!(
                    output,
                    "  /temperature <T>   Set sampling temperature (0.0 = greedy, 1.0+ = creative)"
                )?;
                writeln!(output, "  /top-k <K|off>     Set or disable top-k sampling")?;
                writeln!(
                    output,
                    "  /top-p <P|off>     Set or disable nucleus (top-p) sampling"
                )?;
                writeln!(
                    output,
                    "  /speculative       Toggle self-speculative decoding"
                )?;
                writeln!(output, "  /max-tokens <N>    Set max tokens per response")?;
                writeln!(
                    output,
                    "  /system <MSG|off>  Set, change, or remove the system prompt"
                )?;
                writeln!(output, "  /debug             Toggle debug output")?;
                writeln!(output, "  /settings          Show all current settings")?;
                writeln!(
                    output,
                    "  /context           Show token usage and context headroom"
                )?;
                writeln!(
                    output,
                    "  /save <file>       Save conversation history to JSON file"
                )?;
                writeln!(
                    output,
                    "  /fullsave <file>   Save conversation + settings to JSON file"
                )?;
                writeln!(
                    output,
                    "  /load <file>       Load conversation (and optional settings) from JSON file"
                )?;
                writeln!(output)?;
                writeln!(output, "Coding mode:")?;
                writeln!(
                    output,
                    "  /code [on|off]     Toggle coding mode (structured edit proposals)"
                )?;
                writeln!(
                    output,
                    "  /diff              Show pending file edits proposed by the model"
                )?;
                writeln!(
                    output,
                    "  /apply [all|N]     Apply pending edits to files (with confirmation)"
                )?;
                writeln!(
                    output,
                    "  /discard [all|N]   Discard pending edits without applying"
                )?;
                writeln!(output)?;
                writeln!(output, "  /help              Show this help")?;
                writeln!(output)?;
                writeln!(
                    output,
                    "Use @filepath in messages to include file contents (e.g. @src/main.rs:10-50)."
                )?;
                writeln!(output, "Anything else is sent as a message to the model.")?;
                continue;
            }
            _ if cmd.starts_with('/') => {
                writeln!(
                    output,
                    "Unknown command '{}'. Type /help for available commands.",
                    cmd
                )?;
                continue;
            }
            _ => {}
        }

        if !skip_processing {
            // Expand @file references in user input (works with or without /code mode)
            let effective_input = if user_input.contains('@') {
                let (expanded, _paths) = expand_file_references(&user_input, &cwd);
                if config.debug && expanded != user_input {
                    writeln!(output, "[expanded {} @-reference(s)]", {
                        let refs = torchless::parse_file_references(&user_input, &cwd);
                        refs.len()
                    })?;
                }
                expanded
            } else {
                user_input.clone()
            };

            // Add user message to history
            history.push(ChatMessage::user(&effective_input));

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
                        writeln!(output, "Context trimmed; resetting KV cache.")?;
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
                        writeln!(
                            output,
                            "KV cache reuse: {} cached, {} new tokens to process",
                            new_start,
                            new_tokens.len()
                        )?;
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
        let result = generate_response(&mut state, last_prompt_token, &eos_ids, &config, output);

        writeln!(output)?;

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
                writeln!(output, "[stopped at EOS]")?;
            } else {
                writeln!(output, "[stopped at max_tokens ({})]", config.max_tokens)?;
            }
        }

        // Parse edit blocks from response when coding mode is active
        if coding_mode {
            let new_edits = parse_edit_blocks(&response_text);
            if !new_edits.is_empty() {
                writeln!(
                    output,
                    "[{} edit(s) proposed. Use /diff to review, /apply to apply.]",
                    new_edits.len()
                )?;
                pending_edits.extend(new_edits);
            }
        }

        // Add assistant response to history (thinking stripped)
        history.push(ChatMessage::assistant(response_text));

        // Update processed_tokens to include the generated tokens
        // (so next turn can compute the correct delta)
        processed_tokens.extend_from_slice(&result.tokens);

        writeln!(output)?;
    }

    Ok(())
}
