//! Runs real static Linux binaries — BusyBox applets — through the
//! environment, interpreted and with the JIT, and checks stdout and the exit
//! status against the same command run natively on the host.
//!
//! The binary is `$BUSYBOX`, or the first static `busybox` found in the usual
//! places (Fedora's `busybox`, Debian's `busybox-static`). Without one the
//! tests skip. Every case runs in a sandbox root populated with fixed files
//! and a `bin` directory of applet symlinks on `PATH`, with the host run's
//! working directory set to that root, so the two runs see the same relative
//! paths and the same commands; nothing depends on the clock, the pid or the
//! machine name.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio as HostStdio};
use std::sync::OnceLock;

use qcode_userland::fs::Stdio;
use qcode_userland::{Config, Process, ProcessExit};

/// A cap on retired p-code operations, so a hang fails instead of stalling
/// the suite. The heaviest case here, hashing the 1 KiB file, retires well
/// under a hundred million.
const BUDGET: u64 = 5_000_000_000;

fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(path) = std::env::var("BUSYBOX") {
        out.push(PathBuf::from(path));
    }
    out.extend(
        ["/usr/sbin/busybox", "/usr/bin/busybox", "/bin/busybox"]
            .iter()
            .map(PathBuf::from),
    );
    out
}

/// Whether an ELF image has no `PT_INTERP`, that is, needs no dynamic loader.
fn is_static(image: &[u8]) -> bool {
    let u16_at = |o: usize| u16::from_le_bytes([image[o], image[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes(image[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(image[o..o + 8].try_into().unwrap());
    if image.len() < 64 || &image[..4] != b"\x7fELF" {
        return false;
    }
    let phoff = u64_at(0x20) as usize;
    let phentsize = usize::from(u16_at(0x36));
    let phnum = usize::from(u16_at(0x38));
    (0..phnum).all(|i| {
        let at = phoff + i * phentsize;
        at + 4 <= image.len() && u32_at(at) != 3 // PT_INTERP
    })
}

/// The busybox to test, and its image; `None` when there is none to test.
fn busybox() -> Option<&'static (PathBuf, Vec<u8>)> {
    static FOUND: OnceLock<Option<(PathBuf, Vec<u8>)>> = OnceLock::new();
    FOUND
        .get_or_init(|| {
            for path in candidates() {
                let Ok(image) = std::fs::read(&path) else {
                    continue;
                };
                if is_static(&image) {
                    return Some((path, image));
                }
                eprintln!("{} is dynamically linked; looking further", path.display());
            }
            eprintln!("skipping: no static busybox (set BUSYBOX to point at one)");
            None
        })
        .as_ref()
}

/// Twenty numbered lines in a fixed scrambled order, for `sort` and friends.
fn scrambled_lines() -> String {
    let mut order: Vec<u32> = (1..=20).collect();
    let mut state = 0x2545_f491u32;
    for i in (1..order.len()).rev() {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
        order.swap(i, (state >> 16) as usize % (i + 1));
    }
    order.iter().map(|n| format!("{n}\n")).collect()
}

/// One KiB of fixed printable text, for the hashes.
fn text_block() -> String {
    let mut state = 0x9e37_79b9u32;
    let mut out = String::new();
    while out.len() < 1024 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        out.push_str(&format!("{state:08x} the quick brown fox\n"));
    }
    out
}

/// The sandbox root, built once per process.
fn root() -> &'static Path {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("busybox-root");
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("hello.txt"), "hello file\n").unwrap();
        std::fs::write(root.join("lines.txt"), scrambled_lines()).unwrap();
        std::fs::write(root.join("block.txt"), text_block()).unwrap();
        std::fs::write(root.join("sub/a.txt"), "a\n").unwrap();
        std::fs::write(root.join("sub/b.txt"), "b\n").unwrap();
        std::fs::write(root.join("large.txt"), text_block().repeat(100)).unwrap();
        // Cases that create files do so under here, so the listings of the
        // root and of `sub` stay stable while tests run in parallel.
        std::fs::create_dir_all(root.join("work")).unwrap();
        // The shell finds external commands here, on the host and in the
        // guest alike: each is the busybox binary under an applet's name.
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        if let Some((busybox, _)) = busybox() {
            for applet in [
                "cat",
                "sort",
                "head",
                "tail",
                "wc",
                "seq",
                "yes",
                "tr",
                "gzip",
                "gunzip",
                "sha256sum",
                "ls",
                "rm",
                "mkdir",
                "mv",
                "sh",
                "true",
                "false",
                "echo",
                "grep",
            ] {
                let link = bin.join(applet);
                let _ = std::fs::remove_file(&link);
                std::os::unix::fs::symlink(busybox, &link).unwrap();
            }
        }
        root
    })
}

struct Outcome {
    stdout: Vec<u8>,
    code: i32,
}

fn native(busybox: &Path, args: &[&str], stdin: &[u8]) -> Outcome {
    use std::io::Write;
    let mut child = Command::new(busybox)
        .args(args)
        .current_dir(root())
        .env_clear()
        .env("PATH", root().join("bin"))
        .stdin(HostStdio::piped())
        .stdout(HostStdio::piped())
        .stderr(HostStdio::inherit())
        .spawn()
        .expect("spawn the host busybox");
    child.stdin.take().unwrap().write_all(stdin).unwrap();
    let out = child.wait_with_output().unwrap();
    Outcome {
        stdout: out.stdout,
        code: out.status.code().expect("exited normally"),
    }
}

fn emulated(busybox: &Path, image: &[u8], jit: bool, args: &[&str], stdin: &[u8]) -> Outcome {
    let mut argv = vec!["busybox".to_owned()];
    argv.extend(args.iter().map(|s| s.to_string()));
    let config = Config {
        argv,
        envp: vec!["PATH=/bin".to_owned()],
        jit,
        trace: false,
        root: Some(root().to_path_buf()),
        stdio: Stdio::Captured,
        exe_path: busybox.display().to_string(),
    };
    let mut process = Process::new(image, config).expect("busybox loads");
    process.files_mut().set_stdin(stdin.to_vec());
    let strategy = if jit { "jit" } else { "interpreter" };
    let code = match process.run(BUDGET) {
        ProcessExit::Exited(code) => code,
        other => panic!(
            "busybox {args:?} ({strategy}) did not exit: {other:?}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&process.files().stdout()),
            String::from_utf8_lossy(&process.files().stderr()),
        ),
    };
    Outcome {
        stdout: process.files().stdout().to_vec(),
        code,
    }
}

/// Runs the applet natively and under both strategies; all three must agree.
fn check(args: &[&str], stdin: &[u8]) {
    let Some((path, image)) = busybox() else {
        return;
    };
    let want = native(path, args, stdin);
    for jit in [false, true] {
        let strategy = if jit { "jit" } else { "interpreter" };
        let got = emulated(path, image, jit, args, stdin);
        assert_eq!(
            String::from_utf8_lossy(&got.stdout),
            String::from_utf8_lossy(&want.stdout),
            "busybox {args:?} ({strategy}) stdout"
        );
        assert_eq!(
            got.code, want.code,
            "busybox {args:?} ({strategy}) exit status"
        );
    }
}

/// Runs the applet under both strategies against a literal expectation, for
/// the cases whose native output depends on the host.
fn check_literal(args: &[&str], stdin: &[u8], stdout: &str, code: i32) {
    let Some((path, image)) = busybox() else {
        return;
    };
    for jit in [false, true] {
        let strategy = if jit { "jit" } else { "interpreter" };
        let got = emulated(path, image, jit, args, stdin);
        assert_eq!(
            String::from_utf8_lossy(&got.stdout),
            stdout,
            "busybox {args:?} ({strategy}) stdout"
        );
        assert_eq!(got.code, code, "busybox {args:?} ({strategy}) exit status");
    }
}

#[test]
fn true_and_false() {
    check(&["true"], b"");
    check(&["false"], b"");
}

#[test]
fn echo() {
    check(&["echo", "hello", "world"], b"");
}

#[test]
fn printf_integers_and_floats() {
    check(&["printf", "%d %5.2f %s\\n", "42", "1.5", "x"], b"");
}

#[test]
fn seq() {
    check(&["seq", "5"], b"");
}

#[test]
fn expr() {
    check(&["expr", "2", "+", "3"], b"");
}

#[test]
fn cat_a_file_and_stdin() {
    check(&["cat", "hello.txt"], b"");
    check(&["cat"], b"from stdin\n");
}

#[test]
fn sort_a_file_and_stdin() {
    check(&["sort", "lines.txt"], b"");
    check(&["sort", "-rn"], scrambled_lines().as_bytes());
}

#[test]
fn wc() {
    check(&["wc", "-l", "lines.txt"], b"");
    check(&["wc"], b"one two\nthree\n");
}

#[test]
fn sha256sum() {
    check(&["sha256sum", "block.txt"], b"");
}

#[test]
fn md5sum() {
    check(&["md5sum", "block.txt", "hello.txt"], b"");
}

#[test]
fn shell_loop_and_variables() {
    check(
        &[
            "sh",
            "-c",
            "for i in 1 2 3; do echo $i; done; x=ab; echo ${x}c $((6*7))",
        ],
        b"",
    );
}

#[test]
fn shell_exit_status() {
    check(&["sh", "-c", "exit 3"], b"");
    check(&["sh", "-c", "false || echo recovered"], b"");
}

#[test]
fn head_tail_and_uniq() {
    check(&["head", "-n", "3", "lines.txt"], b"");
    check(&["tail", "-n", "2", "lines.txt"], b"");
    check(&["uniq", "-c"], b"a\na\nb\na\n");
}

#[test]
fn tr_cut_and_rev() {
    check(&["tr", "a-z", "A-Z"], b"hello file\n");
    check(&["cut", "-d", " ", "-f", "2", "hello.txt"], b"");
    check(&["rev", "hello.txt"], b"");
}

#[test]
fn sed_and_grep() {
    check(&["sed", "s/file/world/", "hello.txt"], b"");
    check(&["grep", "-n", "^1[0-9]$", "lines.txt"], b"");
}

#[test]
fn awk() {
    check(
        &[
            "awk",
            "BEGIN{print 1+2, 10/4} {s+=$1} END{print s, s/NR}",
            "lines.txt",
        ],
        b"",
    );
}

#[test]
fn base64_round_trip() {
    check(&["base64", "hello.txt"], b"");
    check(&["base64", "-d"], b"aGVsbG8gZmlsZQo=\n");
}

#[test]
fn ls_a_directory() {
    check(&["ls", "-1", "."], b"");
    check(&["ls", "sub"], b"");
}

#[test]
fn a_missing_file_is_an_error() {
    // Native busybox prints its own name in the message on stderr; stdout is
    // empty and the status is 1 either way.
    check(&["cat", "missing.txt"], b"");
}

#[test]
fn pwd_in_the_sandbox() {
    check_literal(&["pwd"], b"", "/\n", 0);
}

#[test]
fn shell_pipeline() {
    check(
        &["sh", "-c", "echo hi | cat; seq 20 | tail -n 3 | sort -rn"],
        b"",
    );
}

#[test]
fn shell_pipeline_through_an_exec() {
    // gzip and gunzip are not shell builtins: each is a fork, an execve of
    // the applet symlink, and a wait.
    check(&["sh", "-c", "gzip -c hello.txt | gunzip -c"], b"");
}

#[test]
fn shell_command_substitution_and_subshell() {
    check(
        &[
            "sh",
            "-c",
            "x=$(cat hello.txt); echo \"got $x\"; (exit 5); echo $?; (cd sub; ls)",
        ],
        b"",
    );
}

#[test]
fn shell_reads_a_pipe_line_by_line() {
    check(
        &["sh", "-c", "seq 3 | while read i; do echo \"l$i\"; done"],
        b"",
    );
}

#[test]
fn a_writer_with_no_reader_ends_quietly() {
    // `yes` fills the pipe, `head` leaves, and SIGPIPE ends `yes` without a
    // complaint on stderr or a non-zero status for the pipeline.
    check(&["sh", "-c", "yes | head -n 2; echo status=$?"], b"");
}

#[test]
fn shell_redirections_and_file_management() {
    check(
        &[
            "sh",
            "-c",
            "cd work && mkdir tmpd && echo abc > tmpd/f && mv tmpd/f tmpd/g && cat tmpd/g && ls tmpd && rm -r tmpd && ls tmpd 2>/dev/null; echo done",
        ],
        b"",
    );
}

#[test]
fn child_exit_status_reaches_the_parent() {
    check(
        &[
            "sh",
            "-c",
            "sh -c 'exit 7'; echo $?; false | true; echo $?; true | false; echo $?",
        ],
        b"",
    );
}

#[test]
fn a_pipeline_larger_than_the_pipe_buffer() {
    // 100 KiB through a 64 KiB pipe: the producer blocks on a full pipe and
    // the consumer on an empty one, several times over.
    check(&["sh", "-c", "cat large.txt | wc -c"], b"");
}
