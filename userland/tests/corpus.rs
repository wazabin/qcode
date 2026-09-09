//! Runs the freestanding C corpus under `tests/corpus/` through the
//! environment, interpreted and with the JIT, checking output and status.
//!
//! The programs are built with the host `gcc` at test time; without one the
//! tests skip. They use no libc — SSE is not lifted by the emulator, which
//! rules out static glibc — so each carries its own `_start` and raw syscall
//! wrappers (`corpus/sys.h`).

use std::path::{Path, PathBuf};
use std::process::Command;

use qcode_userland::fs::Stdio;
use qcode_userland::{Config, Process, ProcessExit};

const CFLAGS: &[&str] = &[
    "-static",
    "-nostdlib",
    "-nostartfiles",
    "-O2",
    "-mno-sse",
    "-mno-sse2",
    "-mno-mmx",
    "-fno-tree-vectorize",
    "-fno-stack-protector",
    "-ffreestanding",
    "-fno-tree-loop-distribute-patterns",
    "-fno-asynchronous-unwind-tables",
];

/// A generous cap: the corpus programs retire a few hundred thousand
/// operations at most.
const BUDGET: u64 = 200_000_000;

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus")
}

/// Compiles `name.c`; `None` when there is no compiler to do it with.
fn build(name: &str, pie: bool) -> Option<PathBuf> {
    let out_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("corpus");
    std::fs::create_dir_all(&out_dir).unwrap();
    let out = out_dir.join(if pie {
        format!("{name}-pie")
    } else {
        name.to_owned()
    });
    let mut cmd = Command::new("gcc");
    cmd.args(CFLAGS);
    if pie {
        cmd.args(["-fPIE", "-static-pie"]);
    } else {
        cmd.arg("-no-pie");
    }
    cmd.arg("-o")
        .arg(&out)
        .arg(corpus_dir().join(format!("{name}.c")));
    let status = match cmd.status() {
        Ok(status) => status,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("skipping {name}: gcc is not installed");
            return None;
        }
        Err(e) => panic!("cannot run gcc: {e}"),
    };
    assert!(status.success(), "gcc failed on {name}.c");
    Some(out)
}

struct Outcome {
    stdout: String,
    exit: ProcessExit,
}

fn run(elf: &Path, jit: bool, args: &[&str], envp: &[&str], root: Option<&Path>) -> Outcome {
    let image = std::fs::read(elf).unwrap();
    let mut argv = vec![elf.display().to_string()];
    argv.extend(args.iter().map(|s| s.to_string()));
    let config = Config {
        argv,
        envp: envp.iter().map(|s| s.to_string()).collect(),
        jit,
        trace: false,
        root: root.map(Path::to_path_buf),
        stdio: Stdio::Captured,
        exe_path: elf.display().to_string(),
    };
    let mut process = Process::new(&image, config).expect("the image loads");
    let exit = process.run(BUDGET);
    Outcome {
        stdout: String::from_utf8_lossy(process.files.stdout()).into_owned(),
        exit,
    }
}

/// Asserts the program prints `stdout` and exits with `code`, both ways.
fn check(
    name: &str,
    pie: bool,
    args: &[&str],
    envp: &[&str],
    root: Option<&Path>,
    stdout: &str,
    code: i32,
) {
    let Some(elf) = build(name, pie) else { return };
    for jit in [false, true] {
        let outcome = run(&elf, jit, args, envp, root);
        let strategy = if jit { "jit" } else { "interpreter" };
        match outcome.exit {
            ProcessExit::Exited(got) => assert_eq!(
                got, code,
                "{name} ({strategy}) exit status; stdout:\n{}",
                outcome.stdout
            ),
            other => panic!(
                "{name} ({strategy}) did not exit: {other:?}; stdout:\n{}",
                outcome.stdout
            ),
        }
        assert_eq!(outcome.stdout, stdout, "{name} ({strategy}) stdout");
    }
}

#[test]
fn hello() {
    check("hello", false, &[], &[], None, "hi\n", 0);
}

#[test]
fn args_env_auxv_and_exit_status() {
    let Some(elf) = build("args", false) else {
        return;
    };
    let expected = format!(
        "argc=3\nargv[0]={}\nargv[1]=one\nargv[2]=two words\nFOO is bar\nnenv=2\npagesz=4096\nentry ok\nrandom ok\nphdr ok\n",
        elf.display()
    );
    for jit in [false, true] {
        let outcome = run(
            &elf,
            jit,
            &["one", "two words"],
            &["PATH=/bin", "FOO=bar"],
            None,
        );
        assert!(
            matches!(outcome.exit, ProcessExit::Exited(3)),
            "{:?}",
            outcome.exit
        );
        assert_eq!(outcome.stdout, expected);
    }
}

#[test]
fn brk() {
    check("brk", false, &[], &[], None, "brk ok\n", 0);
}

#[test]
fn mmap_mprotect_munmap() {
    check("mmap", false, &[], &[], None, "mmap ok\n", 0);
}

#[test]
fn files_under_a_sandbox_root() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("root");
    std::fs::create_dir_all(root.join("data")).unwrap();
    std::fs::write(root.join("data/input.txt"), "file contents\n").unwrap();
    std::fs::write(root.join("data/other.txt"), "").unwrap();
    check(
        "file",
        false,
        &[],
        &[],
        Some(&root),
        "size=14 regular=1\nfile contents\nfile\nentries=4 input seen\nexe ok\ncwd ok\n",
        0,
    );
}

#[test]
fn thread_local_storage_via_fs_base() {
    check("tls", false, &[], &[], None, "tls ok\n", 0);
}

#[test]
fn uname_and_identity() {
    check(
        "uname",
        false,
        &[],
        &[],
        None,
        "Linux x86_64\npid>0=1 uid=1000\n",
        0,
    );
}

#[test]
fn startup_syscalls_cpuid_and_rdtsc() {
    check(
        "misc",
        false,
        &[],
        &[],
        None,
        "cpuid: GenuineIntel\nmisc ok\n",
        0,
    );
}

#[test]
fn static_pie() {
    check("pie", true, &[], &[], None, "pie counter=6\nbase ok\n", 0);
}

#[test]
fn a_fault_is_reported_not_fatal() {
    let Some(elf) = build("fault", false) else {
        return;
    };
    for jit in [false, true] {
        let outcome = run(&elf, jit, &[], &[], None);
        assert_eq!(outcome.stdout, "about to fault\n");
        let ProcessExit::Crashed(crash) = outcome.exit else {
            panic!("expected a crash, got {:?}", outcome.exit);
        };
        assert_eq!(crash.signal, Some(11), "{crash}");
        assert!(crash.reason.contains("0x10"), "{crash}");
        assert!(crash.pc.is_some(), "{crash}");
        let rendered = crash.to_string();
        assert!(
            rendered.contains("crashed at") && rendered.contains("rsp="),
            "{rendered}"
        );
    }
}

#[test]
fn budget_exhaustion_is_resumable() {
    let Some(elf) = build("hello", false) else {
        return;
    };
    let image = std::fs::read(&elf).unwrap();
    let config = Config {
        stdio: Stdio::Captured,
        ..Config::default()
    };
    let mut process = Process::new(&image, config).unwrap();
    assert!(matches!(process.run(1), ProcessExit::Budget));
    assert!(matches!(process.run(BUDGET), ProcessExit::Exited(0)));
    assert_eq!(process.files.stdout(), b"hi\n");
}
