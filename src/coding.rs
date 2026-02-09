//! Coding Mode: File Reference Expansion and Edit Application
//!
//! This module provides:
//! - `@filepath` reference parsing and file content injection into prompts
//! - Search/replace edit block parsing from model responses
//! - Edit application with diff display
//! - Gitignore and sensitive file filtering
//!
//! All I/O goes through the `BufRead`/`Write` abstractions so it works in both
//! CLI and Unix-socket (multi-user) modes.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// ANSI colour codes for diff display.
pub const ANSI_RED: &str = "\x1b[31m";
pub const ANSI_GREEN: &str = "\x1b[32m";
pub const ANSI_CYAN: &str = "\x1b[36m";
pub const ANSI_BOLD: &str = "\x1b[1m";

// Re-use the reset code from the chat module.
pub use crate::chat::ANSI_RESET;

/// Maximum number of lines to include when no line range is specified.
const MAX_LINES_NO_RANGE: usize = 500;
/// When truncating, show this many lines from the start.
const TRUNCATE_HEAD: usize = 200;
/// When truncating, show this many lines from the end.
const TRUNCATE_TAIL: usize = 100;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A parsed `@filepath` or `@filepath:N-M` reference from user input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileReference {
    /// The raw text matched (e.g. `@src/main.rs:10-50`).
    pub raw: String,
    /// Resolved absolute path.
    pub path: PathBuf,
    /// Optional 1-indexed inclusive line range.
    pub line_range: Option<(usize, usize)>,
}

/// A pending edit proposed by the model via a search/replace block.
#[derive(Debug, Clone)]
pub struct PendingEdit {
    /// Target file path (as written by the model).
    pub filepath: PathBuf,
    /// Content to search for (must match exactly).
    pub search: String,
    /// Replacement content.
    pub replace: String,
}

// ---------------------------------------------------------------------------
// File reference parsing
// ---------------------------------------------------------------------------

/// Parse `@filepath` and `@filepath:N-M` references from user input.
///
/// Paths are resolved relative to `cwd`.  An `@` is only treated as a
/// reference when it appears at the start of the input or is preceded by
/// whitespace (so `user@example.com` is not matched).
pub fn parse_file_references(input: &str, cwd: &Path) -> Vec<FileReference> {
    let mut refs = Vec::new();
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        if bytes[i] == b'@' {
            // Only match @ at the start or after whitespace.
            if i > 0 && !bytes[i - 1].is_ascii_whitespace() {
                i += 1;
                continue;
            }

            let start = i;
            i += 1; // skip '@'

            // Collect the filepath portion (non-whitespace, stop at ':' which may start a range).
            let path_start = i;
            while i < len && !bytes[i].is_ascii_whitespace() && bytes[i] != b':' {
                i += 1;
            }

            if i == path_start {
                continue; // bare '@'
            }

            let path_str = &input[path_start..i];

            // Optional `:N` or `:N-M` line range suffix.
            let mut line_range = None;
            if i < len && bytes[i] == b':' {
                let colon_pos = i;
                i += 1; // skip ':'
                let num_start = i;
                while i < len && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i > num_start {
                    let start_line: usize = input[num_start..i].parse().unwrap_or(0);
                    if i < len && bytes[i] == b'-' {
                        i += 1;
                        let end_start = i;
                        while i < len && bytes[i].is_ascii_digit() {
                            i += 1;
                        }
                        if i > end_start {
                            let end_line: usize = input[end_start..i].parse().unwrap_or(0);
                            if start_line > 0 && end_line >= start_line {
                                line_range = Some((start_line, end_line));
                            }
                        }
                    } else if start_line > 0 {
                        line_range = Some((start_line, start_line));
                    }
                } else {
                    i = colon_pos; // colon but no digits -- not a range
                }
            }

            let raw = input[start..i].to_string();
            let path = Path::new(path_str);
            let resolved = if path.is_absolute() {
                path.to_path_buf()
            } else {
                cwd.join(path)
            };

            refs.push(FileReference {
                raw,
                path: resolved,
                line_range,
            });
        } else {
            i += 1;
        }
    }

    refs
}

// ---------------------------------------------------------------------------
// Path safety
// ---------------------------------------------------------------------------

/// Returns `Some(reason)` if the path should be blocked from reading.
pub fn check_path_blocked(path: &Path) -> Option<&'static str> {
    let path_str = path.to_string_lossy();
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

    // Environment / secret files
    if file_name == ".env" || file_name.starts_with(".env.") {
        return Some("environment file (may contain secrets)");
    }
    if file_name == "credentials"
        || file_name == "credentials.json"
        || file_name == "credentials.yaml"
    {
        return Some("credentials file");
    }
    if file_name.ends_with("_rsa")
        || file_name.ends_with("_ed25519")
        || file_name.ends_with("_ecdsa")
        || file_name.ends_with(".pem")
        || file_name.ends_with(".key")
    {
        return Some("private key file");
    }

    // Sensitive directories
    if path_str.contains("/.ssh/") || path_str.contains("\\.ssh\\") {
        return Some(".ssh directory");
    }
    if path_str.contains("/.gnupg/") || path_str.contains("\\.gnupg\\") {
        return Some(".gnupg directory");
    }

    // Gitignore (best-effort)
    if is_gitignored(path) {
        return Some("gitignored file");
    }

    None
}

/// Check whether `path` is ignored by git.  Returns `false` if not in a repo
/// or if the command fails.
fn is_gitignored(path: &Path) -> bool {
    let dir = path.parent().unwrap_or(Path::new("."));
    Command::new("git")
        .args(["check-ignore", "-q"])
        .arg(path)
        .current_dir(dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success()) // exit 0 → ignored
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// File reading
// ---------------------------------------------------------------------------

/// Read a file (or a line range) and format it with line numbers.
///
/// Returns `(formatted_content, total_lines, was_truncated)`.
fn read_file_formatted(
    path: &Path,
    line_range: Option<(usize, usize)>,
) -> anyhow::Result<(String, usize, bool)> {
    let content = fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Cannot read '{}': {}", path.display(), e))?;

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();

    match line_range {
        Some((start, end)) => {
            let start_idx = start.saturating_sub(1).min(total);
            let end_idx = end.min(total);
            let selected: Vec<String> = lines[start_idx..end_idx]
                .iter()
                .enumerate()
                .map(|(i, line)| format!("{:>6}|{}", start + i, line))
                .collect();
            Ok((selected.join("\n"), total, false))
        }
        None => {
            if total > MAX_LINES_NO_RANGE {
                let mut out = Vec::new();
                for (i, line) in lines[..TRUNCATE_HEAD].iter().enumerate() {
                    out.push(format!("{:>6}|{}", i + 1, line));
                }
                out.push(format!(
                    "       ... ({} lines omitted, use @file:{}-{} to see specific range) ...",
                    total - TRUNCATE_HEAD - TRUNCATE_TAIL,
                    TRUNCATE_HEAD + 1,
                    total
                ));
                for (i, line) in lines[total - TRUNCATE_TAIL..].iter().enumerate() {
                    out.push(format!("{:>6}|{}", total - TRUNCATE_TAIL + i + 1, line));
                }
                Ok((out.join("\n"), total, true))
            } else {
                let formatted: Vec<String> = lines
                    .iter()
                    .enumerate()
                    .map(|(i, line)| format!("{:>6}|{}", i + 1, line))
                    .collect();
                Ok((formatted.join("\n"), total, false))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// File reference expansion
// ---------------------------------------------------------------------------

/// Expand `@filepath` references in user input by reading the referenced files
/// and injecting their contents.
///
/// Returns `(expanded_message, list_of_resolved_paths)`.
pub fn expand_file_references(input: &str, cwd: &Path) -> (String, Vec<PathBuf>) {
    let refs = parse_file_references(input, cwd);
    if refs.is_empty() {
        return (input.to_string(), vec![]);
    }

    let mut result = input.to_string();
    let mut paths = Vec::new();
    let mut file_blocks = Vec::new();

    // Process in reverse order of their position in `input` so that earlier
    // replacements don't shift indices for later ones.
    let mut refs_with_pos: Vec<(usize, &FileReference)> = refs
        .iter()
        .filter_map(|r| input.find(&r.raw).map(|pos| (pos, r)))
        .collect();
    refs_with_pos.sort_by(|a, b| b.0.cmp(&a.0));

    for (_pos, fref) in &refs_with_pos {
        let rel_path = fref
            .path
            .strip_prefix(cwd)
            .unwrap_or(&fref.path)
            .to_path_buf();

        let range_suffix = match fref.line_range {
            Some((s, e)) if s == e => format!(" (line {})", s),
            Some((s, e)) => format!(" (lines {}-{})", s, e),
            None => String::new(),
        };

        // Check if blocked
        if let Some(reason) = check_path_blocked(&fref.path) {
            let replacement = format!(
                "`{}`{} [blocked: {}]",
                rel_path.display(),
                range_suffix,
                reason
            );
            result = result.replacen(&fref.raw, &replacement, 1);
            continue;
        }

        // Try reading
        match read_file_formatted(&fref.path, fref.line_range) {
            Ok((content, total_lines, truncated)) => {
                let header = match fref.line_range {
                    Some((s, e)) if s == e => format!(
                        "[File: {} (line {}, {} total lines)]",
                        rel_path.display(),
                        s,
                        total_lines
                    ),
                    Some((s, e)) => format!(
                        "[File: {} (lines {}-{}, {} total lines)]",
                        rel_path.display(),
                        s,
                        e,
                        total_lines
                    ),
                    None if truncated => format!(
                        "[File: {} ({} lines, truncated)]",
                        rel_path.display(),
                        total_lines
                    ),
                    None => format!("[File: {} ({} lines)]", rel_path.display(), total_lines),
                };
                file_blocks.push(format!("{}\n```\n{}\n```", header, content));
                paths.push(fref.path.clone());

                let replacement = format!("`{}`{}", rel_path.display(), range_suffix);
                result = result.replacen(&fref.raw, &replacement, 1);
            }
            Err(e) => {
                let replacement =
                    format!("`{}`{} [error: {}]", rel_path.display(), range_suffix, e);
                result = result.replacen(&fref.raw, &replacement, 1);
            }
        }
    }

    if file_blocks.is_empty() {
        (result, paths)
    } else {
        // File contents first, then the (rewritten) user message.
        file_blocks.reverse(); // restore original order
        let combined = format!("{}\n\n{}", file_blocks.join("\n\n"), result);
        (combined, paths)
    }
}

// ---------------------------------------------------------------------------
// Edit block parsing
// ---------------------------------------------------------------------------

/// Parse search/replace edit blocks from a model response.
///
/// Expected format:
///
/// ```text
/// path/to/file
/// <<<<<<< SEARCH
/// exact content to find
/// =======
/// replacement content
/// >>>>>>> REPLACE
/// ```
pub fn parse_edit_blocks(response: &str) -> Vec<PendingEdit> {
    let mut edits = Vec::new();
    let lines: Vec<&str> = response.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        if lines[i].trim() == "<<<<<<< SEARCH" {
            // Filepath must be on the preceding non-empty line and must look
            // like a path (not markdown / comment).
            let filepath = if i > 0 {
                let prev = lines[i - 1].trim();
                if !prev.is_empty()
                    && !prev.starts_with("```")
                    && !prev.starts_with('#')
                    && !prev.starts_with("//")
                    && !prev.starts_with("<!--")
                {
                    Some(prev.to_string())
                } else {
                    None
                }
            } else {
                None
            };

            let filepath = match filepath {
                Some(p) => p,
                None => {
                    i += 1;
                    continue;
                }
            };

            i += 1; // past <<<<<<< SEARCH

            // Collect SEARCH content until =======
            let mut search = Vec::new();
            while i < lines.len() && lines[i].trim() != "=======" {
                search.push(lines[i]);
                i += 1;
            }
            if i >= lines.len() {
                break;
            }
            i += 1; // past =======

            // Collect REPLACE content until >>>>>>> REPLACE
            let mut replace = Vec::new();
            while i < lines.len() && lines[i].trim() != ">>>>>>> REPLACE" {
                replace.push(lines[i]);
                i += 1;
            }
            if i < lines.len() {
                i += 1; // past >>>>>>> REPLACE
            }

            edits.push(PendingEdit {
                filepath: PathBuf::from(filepath),
                search: search.join("\n"),
                replace: replace.join("\n"),
            });
        } else {
            i += 1;
        }
    }

    edits
}

// ---------------------------------------------------------------------------
// Edit application
// ---------------------------------------------------------------------------

/// Apply a single edit to a file on disk.
///
/// `cwd` is used to resolve relative paths.
pub fn apply_edit(edit: &PendingEdit, cwd: &Path) -> anyhow::Result<()> {
    let path = if edit.filepath.is_absolute() {
        edit.filepath.clone()
    } else {
        cwd.join(&edit.filepath)
    };

    let content = fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("Cannot read '{}': {}", path.display(), e))?;

    if edit.search.is_empty() {
        // Empty SEARCH = create / overwrite file.
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(&path, &edit.replace)
            .map_err(|e| anyhow::anyhow!("Cannot write '{}': {}", path.display(), e))?;
        return Ok(());
    }

    match content.find(&edit.search) {
        Some(pos) => {
            let new_content = format!(
                "{}{}{}",
                &content[..pos],
                edit.replace,
                &content[pos + edit.search.len()..]
            );
            fs::write(&path, new_content)
                .map_err(|e| anyhow::anyhow!("Cannot write '{}': {}", path.display(), e))?;
            Ok(())
        }
        None => Err(anyhow::anyhow!(
            "Search content not found in '{}'. The file may have changed since the edit was proposed.",
            path.display()
        )),
    }
}

// ---------------------------------------------------------------------------
// Diff display
// ---------------------------------------------------------------------------

/// Format a single pending edit as a coloured unified-diff-style block.
pub fn format_edit_diff(edit: &PendingEdit, index: usize) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{}{}Edit {}: {}{}\n",
        ANSI_BOLD,
        ANSI_CYAN,
        index + 1,
        edit.filepath.display(),
        ANSI_RESET,
    ));
    if edit.search.is_empty() {
        out.push_str(&format!("  {}(new file){}\n", ANSI_GREEN, ANSI_RESET));
    } else {
        for line in edit.search.lines() {
            out.push_str(&format!("  {}-{}{}\n", ANSI_RED, line, ANSI_RESET));
        }
    }
    for line in edit.replace.lines() {
        out.push_str(&format!("  {}+{}{}\n", ANSI_GREEN, line, ANSI_RESET));
    }
    out
}

// ---------------------------------------------------------------------------
// Coding-mode system prompt
// ---------------------------------------------------------------------------

/// Returns the system prompt addendum that instructs the model to use
/// search/replace blocks when proposing edits.
pub fn coding_system_prompt() -> &'static str {
    "\n\nWhen proposing code changes, use SEARCH/REPLACE blocks with this exact format:\n\
     \n\
     path/to/file\n\
     <<<<<<< SEARCH\n\
     exact lines to find in the file\n\
     =======\n\
     replacement lines\n\
     >>>>>>> REPLACE\n\
     \n\
     Rules for edit blocks:\n\
     - The SEARCH section must match the existing file content exactly, including whitespace and indentation.\n\
     - Include enough surrounding context lines in SEARCH for a unique match (at least 3 lines of context).\n\
     - Use one SEARCH/REPLACE block per distinct change.\n\
     - Always put the file path on the line immediately before <<<<<<< SEARCH.\n\
     - You can have multiple SEARCH/REPLACE blocks for the same file.\n\
     - For creating a new file, use an empty SEARCH section with the full content in REPLACE."
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    // -- parse_file_references ------------------------------------------------

    #[test]
    fn test_parse_simple() {
        let cwd = Path::new("/home/user/project");
        let refs = parse_file_references("fix @src/main.rs please", cwd);
        assert_eq!(refs.len(), 1);
        assert_eq!(
            refs[0].path,
            PathBuf::from("/home/user/project/src/main.rs")
        );
        assert_eq!(refs[0].line_range, None);
        assert_eq!(refs[0].raw, "@src/main.rs");
    }

    #[test]
    fn test_parse_line_range() {
        let cwd = Path::new("/home/user/project");
        let refs = parse_file_references("look at @src/main.rs:10-50", cwd);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].line_range, Some((10, 50)));
        assert_eq!(refs[0].raw, "@src/main.rs:10-50");
    }

    #[test]
    fn test_parse_single_line() {
        let cwd = Path::new("/home/user/project");
        let refs = parse_file_references("check @src/lib.rs:42", cwd);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].line_range, Some((42, 42)));
    }

    #[test]
    fn test_parse_multiple() {
        let cwd = Path::new("/home/user/project");
        let refs = parse_file_references("compare @src/a.rs and @src/b.rs", cwd);
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].raw, "@src/a.rs");
        assert_eq!(refs[1].raw, "@src/b.rs");
    }

    #[test]
    fn test_parse_at_start() {
        let cwd = Path::new("/home/user");
        let refs = parse_file_references("@file.txt has a bug", cwd);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].path, PathBuf::from("/home/user/file.txt"));
    }

    #[test]
    fn test_parse_email_not_matched() {
        let cwd = Path::new("/home/user");
        let refs = parse_file_references("email user@example.com", cwd);
        assert_eq!(refs.len(), 0);
    }

    #[test]
    fn test_parse_absolute_path() {
        let cwd = Path::new("/home/user");
        let refs = parse_file_references("read @/etc/hosts", cwd);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].path, PathBuf::from("/etc/hosts"));
    }

    // -- check_path_blocked ---------------------------------------------------

    #[test]
    fn test_block_env() {
        assert!(check_path_blocked(Path::new("/project/.env")).is_some());
        assert!(check_path_blocked(Path::new("/project/.env.local")).is_some());
    }

    #[test]
    fn test_block_ssh() {
        assert!(check_path_blocked(Path::new("/home/user/.ssh/id_rsa")).is_some());
    }

    #[test]
    fn test_block_pem() {
        assert!(check_path_blocked(Path::new("/certs/server.pem")).is_some());
        assert!(check_path_blocked(Path::new("/certs/private.key")).is_some());
    }

    // -- parse_edit_blocks ----------------------------------------------------

    #[test]
    fn test_edit_blocks_basic() {
        let response = "\
Here's the fix:

src/main.rs
<<<<<<< SEARCH
fn old_code() {
    println!(\"old\");
}
=======
fn new_code() {
    println!(\"new\");
}
>>>>>>> REPLACE

That should fix it.";

        let edits = parse_edit_blocks(response);
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].filepath, PathBuf::from("src/main.rs"));
        assert!(edits[0].search.contains("fn old_code()"));
        assert!(edits[0].replace.contains("fn new_code()"));
    }

    #[test]
    fn test_edit_blocks_multiple() {
        let response = "\
Two changes needed:

src/a.rs
<<<<<<< SEARCH
old_a
=======
new_a
>>>>>>> REPLACE

src/b.rs
<<<<<<< SEARCH
old_b
=======
new_b
>>>>>>> REPLACE";

        let edits = parse_edit_blocks(response);
        assert_eq!(edits.len(), 2);
        assert_eq!(edits[0].filepath, PathBuf::from("src/a.rs"));
        assert_eq!(edits[1].filepath, PathBuf::from("src/b.rs"));
    }

    #[test]
    fn test_edit_blocks_empty_search() {
        let response = "\
new_file.rs
<<<<<<< SEARCH
=======
fn main() {}
>>>>>>> REPLACE";

        let edits = parse_edit_blocks(response);
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].search, "");
        assert_eq!(edits[0].replace, "fn main() {}");
    }

    #[test]
    fn test_edit_blocks_none() {
        let edits = parse_edit_blocks("No changes needed, everything looks good!");
        assert_eq!(edits.len(), 0);
    }

    // -- apply_edit -----------------------------------------------------------

    #[test]
    fn test_apply_edit_success() {
        let dir = std::env::temp_dir().join("torchless_test_apply");
        let _ = fs::create_dir_all(&dir);
        let file_path = dir.join("test_apply.rs");
        {
            let mut f = fs::File::create(&file_path).unwrap();
            write!(f, "fn main() {{\n    println!(\"hello\");\n}}").unwrap();
        }

        let edit = PendingEdit {
            filepath: PathBuf::from("test_apply.rs"),
            search: "println!(\"hello\")".to_string(),
            replace: "println!(\"world\")".to_string(),
        };

        apply_edit(&edit, &dir).unwrap();

        let content = fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("println!(\"world\")"));
        assert!(!content.contains("println!(\"hello\")"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_apply_edit_not_found() {
        let dir = std::env::temp_dir().join("torchless_test_notfound");
        let _ = fs::create_dir_all(&dir);
        let file_path = dir.join("test_nf.rs");
        fs::write(&file_path, "fn main() {}").unwrap();

        let edit = PendingEdit {
            filepath: PathBuf::from("test_nf.rs"),
            search: "this does not exist".to_string(),
            replace: "replacement".to_string(),
        };

        assert!(apply_edit(&edit, &dir).is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    // -- format_edit_diff -----------------------------------------------------

    #[test]
    fn test_format_diff() {
        let edit = PendingEdit {
            filepath: PathBuf::from("src/main.rs"),
            search: "old line".to_string(),
            replace: "new line".to_string(),
        };
        let diff = format_edit_diff(&edit, 0);
        assert!(diff.contains("Edit 1"));
        assert!(diff.contains("src/main.rs"));
        assert!(diff.contains("-old line"));
        assert!(diff.contains("+new line"));
    }

    #[test]
    fn test_format_diff_new_file() {
        let edit = PendingEdit {
            filepath: PathBuf::from("new.rs"),
            search: String::new(),
            replace: "fn main() {}".to_string(),
        };
        let diff = format_edit_diff(&edit, 0);
        assert!(diff.contains("(new file)"));
        assert!(diff.contains("+fn main() {}"));
    }

    // -- coding_system_prompt -------------------------------------------------

    #[test]
    fn test_system_prompt_content() {
        let p = coding_system_prompt();
        assert!(p.contains("SEARCH"));
        assert!(p.contains("REPLACE"));
        assert!(p.contains("exact"));
    }
}
