//! Unix socket server for multi-user chat access.
//!
//! Allows multiple users to connect via a Unix domain socket (e.g. with telnet, socat, nc)
//! and chat with the loaded model. Each connection gets its own conversation state,
//! with save/load operations scoped to a per-user directory under the configured save root.

use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use torchless::{
    display_thinking_token_to, generate_lazy_until_eos, generate_until_eos, init_backend,
    BackendPreference, ChatTemplate, InferenceState, KVDtype, LazyMistral, Mistral, Parameters,
    SamplingConfig, ThinkingState,
};

use crate::{
    generate_speculative_until_eos, resolve_max_seq_len, resolve_max_tokens, run_chat_repl,
    ChatSessionConfig,
};

/// Validate a username: allow only a-z, A-Z, 0-9, underscore, hyphen, space.
/// Must be 1-64 characters.
fn validate_username(name: &str) -> Option<String> {
    let name = name.trim();
    if name.is_empty() || name.len() > 64 {
        return None;
    }
    if name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | ' '))
    {
        Some(name.to_string())
    } else {
        None
    }
}

/// Run the Unix socket server, accepting connections and spawning per-user chat sessions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_socket_server(
    socket_path: &str,
    chat_save_root: &str,
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
    eprintln!("[server] Using backend: {}", backend.name());

    // Create save root directory
    let save_root = PathBuf::from(chat_save_root);
    fs::create_dir_all(&save_root)?;

    // Remove stale socket file if it exists
    if Path::new(socket_path).exists() {
        eprintln!("[server] Removing stale socket file: {}", socket_path);
        fs::remove_file(socket_path)?;
    }

    eprintln!("[server] Loading model from: {}", model_path);
    let params = Parameters::load(model_path)?;

    // Store shared config values for spawning threads
    let sampling_config = sampling_config.clone();
    let system_prompt = system_prompt.map(|s| s.to_string());
    let save_root = Arc::new(save_root);

    // The inference lock serializes model access across threads.
    // Only one user can run inference at a time (GPU backends may not support concurrent access).
    let inference_lock = Arc::new(Mutex::new(()));

    if lazy {
        // Lazy loading: share Parameters across threads via Arc
        let params = Arc::new(params);

        eprintln!("[server] Loading weights (lazy mode)...");
        // Load once to get config/tokenizer info
        let probe_model = LazyMistral::load(&params)?;
        let max_seq_len =
            resolve_max_seq_len(max_seq_len_arg, probe_model.config.max_position_embeddings);
        let max_tokens = resolve_max_tokens(max_tokens_arg, max_seq_len);
        let model_config = probe_model.config.clone();

        // Initialize thinking state info from the probe model
        let chat_template = ChatTemplate::Mistral;
        let thinking_ids = chat_template.thinking_token_ids(&probe_model.tokenizer);
        let thinking_ids = Arc::new(thinking_ids);
        drop(probe_model); // Free the probe model; each thread loads its own

        eprintln!(
            "[server] Context window: {} tokens, max response: {} tokens",
            max_seq_len, max_tokens
        );

        // Bind socket
        let listener = UnixListener::bind(socket_path)?;
        eprintln!("[server] Listening on: {}", socket_path);
        eprintln!("[server] Ready for connections (Ctrl+C to stop)");

        setup_cleanup_notice(socket_path);

        for stream in listener.incoming() {
            let stream = match stream {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[server] Accept error: {}", e);
                    continue;
                }
            };

            let params = params.clone();
            let inference_lock = inference_lock.clone();
            let save_root = save_root.clone();
            let sampling_config = sampling_config.clone();
            let system_prompt = system_prompt.clone();
            let model_config = model_config.clone();
            let thinking_ids = thinking_ids.clone();

            thread::spawn(move || {
                let peer = format!("{:?}", stream.peer_addr());
                eprintln!("[server] New connection from {}", peer);

                let result = handle_connection_lazy(
                    stream,
                    &params,
                    &inference_lock,
                    &save_root,
                    &sampling_config,
                    system_prompt.as_deref(),
                    &model_config,
                    &thinking_ids,
                    max_seq_len,
                    max_tokens,
                    speculative,
                    show_thinking,
                    debug,
                    kv_dtype,
                );

                match result {
                    Ok(username) => eprintln!("[server] {} ({}) disconnected", username, peer),
                    Err(e) => eprintln!("[server] Connection {} error: {}", peer, e),
                }
            });
        }
    } else {
        // Eager loading: share model via Arc
        eprintln!("[server] Loading weights...");
        let model = Arc::new(Mistral::load(params)?);

        let max_seq_len =
            resolve_max_seq_len(max_seq_len_arg, model.config.max_position_embeddings);
        let max_tokens = resolve_max_tokens(max_tokens_arg, max_seq_len);

        // Initialize thinking state info
        let chat_template = ChatTemplate::Mistral;
        let thinking_ids = chat_template.thinking_token_ids(&model.tokenizer);
        let thinking_ids = Arc::new(thinking_ids);

        eprintln!(
            "[server] Context window: {} tokens, max response: {} tokens",
            max_seq_len, max_tokens
        );

        // Bind socket
        let listener = UnixListener::bind(socket_path)?;
        eprintln!("[server] Listening on: {}", socket_path);
        eprintln!("[server] Ready for connections (Ctrl+C to stop)");

        setup_cleanup_notice(socket_path);

        for stream in listener.incoming() {
            let stream = match stream {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[server] Accept error: {}", e);
                    continue;
                }
            };

            let model = model.clone();
            let inference_lock = inference_lock.clone();
            let save_root = save_root.clone();
            let sampling_config = sampling_config.clone();
            let system_prompt = system_prompt.clone();
            let thinking_ids = thinking_ids.clone();

            thread::spawn(move || {
                let peer = format!("{:?}", stream.peer_addr());
                eprintln!("[server] New connection from {}", peer);

                let result = handle_connection_eager(
                    stream,
                    &model,
                    &inference_lock,
                    &save_root,
                    &sampling_config,
                    system_prompt.as_deref(),
                    &thinking_ids,
                    max_seq_len,
                    max_tokens,
                    speculative,
                    show_thinking,
                    debug,
                    kv_dtype,
                );

                match result {
                    Ok(username) => eprintln!("[server] {} ({}) disconnected", username, peer),
                    Err(e) => eprintln!("[server] Connection {} error: {}", peer, e),
                }
            });
        }
    }

    Ok(())
}

/// Handle a single connection using lazy model loading.
#[allow(clippy::too_many_arguments)]
fn handle_connection_lazy(
    stream: std::os::unix::net::UnixStream,
    params: &Arc<Parameters>,
    inference_lock: &Arc<Mutex<()>>,
    save_root: &Path,
    sampling_config: &SamplingConfig,
    system_prompt: Option<&str>,
    model_config: &torchless::Config,
    thinking_ids: &Option<(u32, u32)>,
    max_seq_len: usize,
    max_tokens: usize,
    speculative: bool,
    show_thinking: bool,
    debug: bool,
    kv_dtype: KVDtype,
) -> anyhow::Result<String> {
    let read_stream = stream.try_clone()?;
    let mut reader = BufReader::new(read_stream);
    let mut writer = BufWriter::new(stream);

    // Username negotiation
    let username = negotiate_username(&mut reader, &mut writer)?;
    let user_save_root = save_root.join(&username);

    // Create per-user InferenceState
    let state = InferenceState::with_seq_len(model_config.clone(), max_seq_len, kv_dtype);

    // Create per-user thinking state
    let thinking_state = ThinkingState::new(*thinking_ids, show_thinking);

    let config = ChatSessionConfig {
        sampling: sampling_config.clone(),
        speculative,
        max_tokens,
        debug,
        system_prompt: system_prompt.map(|s| s.to_string()),
    };

    // Load model for this thread (lazy - shares the underlying Parameters via Arc)
    let model = LazyMistral::load(params)?;
    let lock = inference_lock.clone();

    run_chat_repl(
        state,
        &model.tokenizer,
        config,
        max_seq_len,
        &thinking_state,
        Some(&user_save_root),
        &mut reader,
        &mut writer,
        &mut |state, tokens, dbg| {
            let _guard = lock.lock().unwrap();
            for (i, &token) in tokens.iter().enumerate() {
                if dbg && (i % 50 == 0) {
                    eprintln!("Processing prompt token {}/{}", i + 1, tokens.len());
                }
                model.forward(state, token, dbg);
                state.pos += 1;
            }
        },
        &mut |state, first_token, eos_ids, config, output| {
            let _guard = lock.lock().unwrap();
            let dbg = config.debug;
            if config.speculative {
                generate_speculative_until_eos(
                    |s, t| model.forward(s, t, dbg),
                    &model.tokenizer,
                    state,
                    first_token,
                    config.sampling.temperature,
                    config.max_tokens,
                    eos_ids,
                    Some(&thinking_state),
                    dbg,
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
                    dbg,
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
    )?;

    Ok(username)
}

/// Handle a single connection using eager model loading.
#[allow(clippy::too_many_arguments)]
fn handle_connection_eager(
    stream: std::os::unix::net::UnixStream,
    model: &Arc<Mistral>,
    inference_lock: &Arc<Mutex<()>>,
    save_root: &Path,
    sampling_config: &SamplingConfig,
    system_prompt: Option<&str>,
    thinking_ids: &Option<(u32, u32)>,
    max_seq_len: usize,
    max_tokens: usize,
    speculative: bool,
    show_thinking: bool,
    debug: bool,
    kv_dtype: KVDtype,
) -> anyhow::Result<String> {
    let read_stream = stream.try_clone()?;
    let mut reader = BufReader::new(read_stream);
    let mut writer = BufWriter::new(stream);

    // Username negotiation
    let username = negotiate_username(&mut reader, &mut writer)?;
    let user_save_root = save_root.join(&username);

    // Create per-user InferenceState
    let state = InferenceState::with_seq_len(model.config.clone(), max_seq_len, kv_dtype);

    // Create per-user thinking state
    let thinking_state = ThinkingState::new(*thinking_ids, show_thinking);

    let config = ChatSessionConfig {
        sampling: sampling_config.clone(),
        speculative,
        max_tokens,
        debug,
        system_prompt: system_prompt.map(|s| s.to_string()),
    };

    let lock = inference_lock.clone();
    let model_ref = model.clone();

    run_chat_repl(
        state,
        &model.tokenizer,
        config,
        max_seq_len,
        &thinking_state,
        Some(&user_save_root),
        &mut reader,
        &mut writer,
        &mut |state, tokens, dbg| {
            let _guard = lock.lock().unwrap();
            for (i, &token) in tokens.iter().enumerate() {
                if dbg && (i % 50 == 0) {
                    eprintln!("Processing prompt token {}/{}", i + 1, tokens.len());
                }
                model_ref.forward(state, token, dbg);
                state.pos += 1;
            }
        },
        &mut |state, first_token, eos_ids, config, output| {
            let _guard = lock.lock().unwrap();
            let dbg = config.debug;
            if config.speculative {
                generate_speculative_until_eos(
                    |s, t| model_ref.forward(s, t, dbg),
                    &model_ref.tokenizer,
                    state,
                    first_token,
                    config.sampling.temperature,
                    config.max_tokens,
                    eos_ids,
                    Some(&thinking_state),
                    dbg,
                    output,
                )
            } else {
                generate_until_eos(
                    &model_ref,
                    state,
                    first_token,
                    &config.sampling,
                    config.max_tokens,
                    eos_ids,
                    dbg,
                    |token_id| {
                        let action = thinking_state.process_token(token_id);
                        display_thinking_token_to(output, action, || {
                            model_ref.tokenizer.decode(&[token_id])
                        });
                        let _ = output.flush();
                    },
                )
            }
        },
    )?;

    Ok(username)
}

/// Negotiate a username with a newly connected client.
/// Sends a welcome message, reads the username, validates it, and returns it.
fn negotiate_username(reader: &mut dyn BufRead, writer: &mut dyn Write) -> anyhow::Result<String> {
    writeln!(writer, "Welcome to torchless chat server.")?;
    write!(writer, "Enter your username: ")?;
    writer.flush()?;

    let mut line = String::new();
    match reader.read_line(&mut line) {
        Ok(0) => anyhow::bail!("Client disconnected before sending username"),
        Ok(_) => {}
        Err(e) => anyhow::bail!("Error reading username: {}", e),
    }

    let username = match validate_username(&line) {
        Some(u) => u,
        None => {
            writeln!(
                writer,
                "Invalid username. Use only letters, numbers, underscore, hyphen, space (1-64 chars)."
            )?;
            writer.flush()?;
            anyhow::bail!("Invalid username: {:?}", line.trim());
        }
    };

    writeln!(writer, "Hello, {}! Type /help for commands.", username)?;
    writer.flush()?;

    eprintln!("[server] User '{}' authenticated", username);
    Ok(username)
}

/// Best-effort socket cleanup on server shutdown.
///
/// Rust's standard library doesn't provide portable signal handling, so we rely on
/// the stale socket removal at startup to handle the common case where the process
/// is killed without cleanup. The socket file is automatically removed when the
/// server starts and finds a stale socket.
fn setup_cleanup_notice(socket_path: &str) {
    eprintln!(
        "[server] Socket: {}  (stale sockets are auto-removed on next startup)",
        socket_path
    );
}
