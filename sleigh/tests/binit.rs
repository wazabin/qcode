//! Replays Binit/Aegis x86-64 instruction states through the QCode SLEIGH
//! lifter and emulator. The database-backed tests are ignored by default.
mod engine {
    use std::any::Any;
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::fs::File;
    use std::io::BufWriter;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
    use postgres::{Client, NoTls};
    use serde::Serialize;

    use qcode::{
        context::Context,
        value::{insn::Mnemonic, varnode::register::RegisterId},
    };
    use qcode_emulator::{Emulator, EmulatorError, EmulatorErrorKind};
    use sleigh::{CompiledSpec, Decoder, Opcode, SymbolKind, Varnode};
    use sleigh_precompile::x64;
    use wazabin_qcode_sleigh::SleighLifter;

    /// Flag table: (name, bit-mask in RFLAGS word)
    const FLAGS: &[(&str, u64)] = &[
        ("CF", 0x001),
        ("PF", 0x004),
        ("AF", 0x010),
        ("ZF", 0x040),
        ("SF", 0x080),
        ("DF", 0x400),
        ("OF", 0x800),
    ];
    const FIRE_START: usize = 0x6666_6666_1000;
    const MEM0_ADDR: u64 = 0x6666_6601_0100;
    const MEM1_ADDR: u64 = 0x6666_6601_0200;
    const MEM0_PAGE_ADDR: u64 = MEM0_ADDR & !0xfff;
    const MEM0_PAGE_SIZE: usize = 0x1000;
    /// Fixed raw byte transport rooted at MEM0_ADDR. It covers FXSAVE's full
    /// legacy image and the existing mem1 word at offset 0x100.
    const SCRATCH_MEMORY_SIZE: usize = 512;

/// Where binit places a bit-index instruction's memory operand within the
/// scratch window, so a signed `index s>> 3` reaches seeded bytes either side.
/// Must match `BIT_INDEX_OPERAND_OFFSET` in binit's `make_test_cases.py`.
const BIT_INDEX_OPERAND_OFFSET: usize = 256;
    const MEMORY_WORDS: &[(&str, u64)] = &[("mem0_value", MEM0_ADDR), ("mem1_value", MEM1_ADDR)];
    const MAX_EMULATED_STEPS: usize = 10_000;
    const SCALAR_REGISTERS: &[&str] = &[
        "RAX", "RBX", "RCX", "RDX", "RSI", "RDI", "R8", "R9", "RBP", "RSP",
    ];
    /// MMX is the low-64 view of the eight physical x87 slots. The SLEIGH
    /// model intentionally has no independent MM register varnodes.
    const X87_PHYSICAL_REGISTERS: usize = 8;
    const CSV_FIELDNAMES: &[&str] = &[
        "tool",
        "test_case_id",
        "state_index",
        "instruction",
        "opcode",
        "initial_state_json",
        "expected_outcome",
        "actual_outcome",
        "expected_exception",
        "actual_exception",
        "expected_state_json",
        "actual_state_json",
        "diff_json",
        "ignored_flags",
        "mismatch_reason",
    ];

    /// Looks up a register in the exact precompiled x86-64 specification used
    /// to lift the instruction. Register IDs are stable within that spec.
    fn x64_register(name: &str) -> Option<RegisterId> {
        x64::spec()
            .registers()
            .find(|register| register.name().eq_ignore_ascii_case(name))
            .map(|register| register.id)
    }

    /// Decodes and lifts the complete hardware instruction stream.  Binit's
    /// ordinary rows contain one instruction, but a bounded (at most 15-byte)
    /// stream is needed to make restored x87 full tags observable through a
    /// following save instruction.
    fn lift_x64(bytes: &[u8], address: u64) -> Result<Context<'static>, String> {
        let spec: &CompiledSpec = x64::spec();
        let lifter = SleighLifter::new(spec);
        let decode_context = spec.new_context();
        let decoder = Decoder::new(spec);
        let mut context = lifter.new_context();
        let function = context.anon_function();
        let mut offset = 0;
        while offset < bytes.len() {
            let instruction = decoder
                .decode_one(address + offset as u64, &bytes[offset..], &decode_context)
                .map_err(|error| format!("decode failed at byte {offset}: {error}"))?;
            let length = instruction.len();
            if length == 0 {
                return Err(format!("decoded zero-length instruction at byte {offset}"));
            }
            lifter
                .lift_instruction(&mut context, &instruction, Some(function))
                .map_err(|error| format!("lift failed at byte {offset}: {error}"))?;
            offset += length;
        }
        Ok(context)
    }

    fn parse_hex_bytes(s: &str) -> Option<Vec<u8>> {
        if s.is_empty() || s.len() % 2 != 0 {
            return None;
        }
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
            .collect()
    }

    fn instruction_addr(bytes: &[u8]) -> usize {
        FIRE_START
            .checked_sub(bytes.len())
            .expect("instruction is larger than the fire page address")
    }

    /// Measures which SLEIGH constructors are reached by Binit's concrete
    /// encodings. A test case is counted once, independently of how many input
    /// states Aegis generated for it.
    fn collect_constructor_coverage() -> Result<ConstructorCoverage, String> {
        let spec = x64::spec();
        let table_names: Vec<_> = spec
            .symbols()
            .filter(|symbol| matches!(symbol.kind, SymbolKind::Table))
            .map(|symbol| symbol.name.to_owned())
            .collect();
        let table_totals: BTreeMap<_, _> = table_names
            .into_iter()
            .filter_map(|name| {
                spec.table(&name)
                    .map(|table| (name, table.constructor_count()))
            })
            .collect();
        let total_constructors = table_totals.values().sum();
        let mut client = Client::connect(&db_dsn(), NoTls).map_err(|error| error.to_string())?;
        let rows = client
            .query(
                "SELECT instruction, opcode FROM test_cases ORDER BY id",
                &[],
            )
            .map_err(|error| error.to_string())?;

        let mut witnesses = BTreeMap::new();
        let mut decoded_test_cases = 0;
        let mut invalid_opcodes = 0;
        let mut decode_failures = 0;
        for row in &rows {
            let instruction_text: String = row.get(0);
            let opcode: String = row.get(1);
            let Some(bytes) = parse_hex_bytes(&opcode) else {
                invalid_opcodes += 1;
                continue;
            };
            let context = spec.new_context();
            match Decoder::new(spec).decode_one(0, &bytes, &context) {
                Ok(instruction) => {
                    decoded_test_cases += 1;
                    for constructor in instruction.constructor_matches() {
                        let table = constructor.table().name().to_owned();
                        let index = constructor.index();
                        witnesses
                            .entry((table, index))
                            .or_insert_with(|| ConstructorWitness {
                                index,
                                opcode: opcode.clone(),
                                instruction: instruction_text.clone(),
                            });
                    }
                }
                Err(_) => decode_failures += 1,
            }
        }

        let tables: Vec<_> = table_totals
            .into_iter()
            .map(|(table, total_constructors)| {
                let witnesses: Vec<_> = witnesses
                    .iter()
                    .filter(|((covered_table, _), _)| covered_table == &table)
                    .map(|(_, witness)| witness.clone())
                    .collect();
                let covered_constructors = witnesses.len();
                TableCoverage {
                    table,
                    total_constructors,
                    covered_constructors,
                    percentage: if total_constructors == 0 {
                        0.0
                    } else {
                        covered_constructors as f64 * 100.0 / total_constructors as f64
                    },
                    witnesses,
                }
            })
            .collect();
        let covered_constructors = tables.iter().map(|table| table.covered_constructors).sum();

        Ok(ConstructorCoverage {
            architecture: "x86-64",
            total_constructors,
            covered_constructors,
            percentage: if total_constructors == 0 {
                0.0
            } else {
                covered_constructors as f64 * 100.0 / total_constructors as f64
            },
            test_cases: rows.len(),
            decoded_test_cases,
            invalid_opcodes,
            decode_failures,
            tables,
        })
    }

    fn hex_encode(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// Applies a small x86-oriented mutation to an existing corpus encoding.
    fn mutate_opcode(seed: &[u8], rng: &mut XorShift64) -> Vec<u8> {
        const PREFIXES: &[u8] = &[
            0x26, 0x2e, 0x36, 0x3e, 0x64, 0x65, 0x66, 0x67, 0xf0, 0xf2, 0xf3,
        ];

        let mut bytes = seed.to_vec();
        match rng.index(5) {
            // Change a whole byte, which efficiently explores opcode, ModRM,
            // SIB, displacement, and immediate fields without knowing the
            // instruction's syntax.
            0 => {
                let index = rng.index(bytes.len());
                bytes[index] = rng.next() as u8;
            }
            // A one-bit change is useful around opcode extension selectors.
            1 => {
                let index = rng.index(bytes.len());
                bytes[index] ^= 1 << rng.index(8);
            }
            // Prefixes are constructors in their own right in the x86 spec.
            2 if bytes.len() < 15 => {
                bytes.insert(0, PREFIXES[rng.index(PREFIXES.len())]);
            }
            // Some forms are prefixes of longer encodings.
            3 if bytes.len() > 1 => {
                bytes.remove(rng.index(bytes.len()));
            }
            // Probe a trailing immediate/displacement byte.
            _ if bytes.len() < 15 => bytes.push(rng.next() as u8),
            _ => {
                let index = rng.index(bytes.len());
                bytes[index] = rng.next() as u8;
            }
        }
        bytes
    }

    /// Mutates Binit opcode encodings and retains only decodable instructions
    /// that cover a constructor absent from the database corpus. The resulting
    /// JSON is intentionally Binit-ready (`instruction` plus `opcode`) but is
    /// not inserted automatically: state generation and Aegis execution remain
    /// the authority for admitting a candidate to the shared database.
    fn generate_constructor_candidates() -> Result<(), String> {
        let coverage = collect_constructor_coverage()?;
        let mut covered = coverage.covered_keys();
        let spec = x64::spec();
        let mut client = Client::connect(&db_dsn(), NoTls).map_err(|error| error.to_string())?;
        let rows = client
            .query(
                "SELECT instruction, opcode FROM test_cases ORDER BY id",
                &[],
            )
            .map_err(|error| error.to_string())?;

        let mut existing = HashSet::new();
        let mut seeds = Vec::new();
        for row in &rows {
            let instruction: String = row.get(0);
            let opcode: String = row.get(1);
            let Some(bytes) = parse_hex_bytes(&opcode) else {
                continue;
            };
            if existing.insert(bytes.clone()) {
                seeds.push((instruction, bytes));
            }
        }
        if seeds.is_empty() {
            return Err("Binit has no valid opcode encodings to mutate".to_string());
        }

        let budget = std::env::var("PCODE_SLEIGH_CANDIDATE_BUDGET")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()
            .map_err(|error| format!("invalid PCODE_SLEIGH_CANDIDATE_BUDGET: {error}"))?
            .unwrap_or(100_000);
        let mut rng = XorShift64(0x9e37_79b9_7f4a_7c15);
        let mut candidates = Vec::new();
        for _ in 0..budget {
            let (_, seed) = &seeds[rng.index(seeds.len())];
            let mutated = mutate_opcode(seed, &mut rng);
            let context = spec.new_context();
            let Ok(instruction) = Decoder::new(spec).decode_one(0, &mutated, &context) else {
                continue;
            };
            let canonical = instruction.bytes().to_vec();
            if canonical.is_empty() || !existing.insert(canonical.clone()) {
                continue;
            }
            let constructor_path: Vec<_> = instruction
                .constructor_matches()
                .map(|constructor| ConstructorLocation {
                    table: constructor.table().name().to_owned(),
                    index: constructor.index(),
                })
                .collect();
            let newly_covered: Vec<_> = constructor_path
                .iter()
                .filter(|constructor| {
                    !covered.contains(&(constructor.table.clone(), constructor.index))
                })
                .cloned()
                .collect();
            if newly_covered.is_empty() {
                continue;
            }
            for constructor in &newly_covered {
                covered.insert((constructor.table.clone(), constructor.index));
            }
            candidates.push(ConstructorCandidate {
                opcode: hex_encode(&canonical),
                instruction: instruction
                    .display()
                    .unwrap_or_else(|error| format!("<unrenderable: {error}>")),
                newly_covered,
                constructor_path,
            });
        }

        let path = std::env::var_os("PCODE_SLEIGH_CANDIDATES_OUTPUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("candidates/sleigh_constructor_candidates.json"));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let file = File::create(&path).map_err(|error| error.to_string())?;
        serde_json::to_writer_pretty(BufWriter::new(file), &candidates)
            .map_err(|error| error.to_string())?;
        let new_constructors: usize = candidates
            .iter()
            .map(|candidate| candidate.newly_covered.len())
            .sum();
        eprintln!(
            "[sleigh-coverage] retained {} candidates covering {} new constructors after {budget} mutations: {}",
            candidates.len(),
            new_constructors,
            path.display(),
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // PostgreSQL-based corpus runner (x86db)
    // -------------------------------------------------------------------------

    /// DB-native state. Ordinary registers use signed JSON-compatible words;
    /// x87 stack entries retain their raw 80-bit little-endian encodings.
    #[derive(Clone, Debug)]
    struct DbState {
        regs: HashMap<String, i64>,
        f80: HashMap<String, u128>,
        scratch_memory: Option<[u8; SCRATCH_MEMORY_SIZE]>,
    }

    const X87_CONTROL_FIELDS: &[(&str, &str)] = &[
        ("x87_control", "FPUControlWord"),
        ("x87_status", "FPUStatusWord"),
        ("x87_tag", "FPUTagWord"),
        ("x87_opcode", "FPULastInstructionOpcode"),
        ("x87_ip", "FPUInstructionPointer"),
        ("x87_dp", "FPUDataPointer"),
    ];

    fn parse_f80(value: &str) -> Option<u128> {
        let value = value.strip_prefix("0x").unwrap_or(value);
        if value.len() != 20 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return None;
        }
        let mut result = 0u128;
        for index in 0..10 {
            let byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).ok()?;
            result |= u128::from(byte) << (index * 8);
        }
        Some(result)
    }

    fn format_f80(value: u128) -> String {
        (0..10)
            .map(|index| format!("{:02x}", (value >> (index * 8)) as u8))
            .collect()
    }

    fn parse_scratch_memory(value: &str) -> Option<[u8; SCRATCH_MEMORY_SIZE]> {
        if value.len() != SCRATCH_MEMORY_SIZE * 2
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return None;
        }
        let mut bytes = [0; SCRATCH_MEMORY_SIZE];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).ok()?;
        }
        Some(bytes)
    }

    fn format_scratch_memory(value: &[u8; SCRATCH_MEMORY_SIZE]) -> String {
        value.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn x87_top(status: u64) -> usize {
        ((status >> 11) & 7) as usize
    }

    /// Binit's x87 transport is canonical physical R0..R7 only. Hand-written
    /// logical fixtures must use binit.db.logical_x87_to_physical_state before
    /// reaching this replay boundary.
    fn x87_physical_slot_index(name: &str) -> Option<usize> {
        name.strip_prefix("x87_r")?
            .parse::<usize>()
            .ok()
            .filter(|&index| index < X87_PHYSICAL_REGISTERS)
    }

    fn x87_space(ctx: &Context<'_>) -> Result<qcode::space::SpaceId, String> {
        ctx.shared
            .named_spaces
            .get("x87")
            .copied()
            .ok_or_else(|| "x86-64 SLEIGH spec has no x87 physical-file space".to_string())
    }

    fn write_x87_slot(
        emu: &mut Emulator<'_>,
        ctx: &Context<'_>,
        slot: usize,
        value: u128,
    ) -> Result<(), String> {
        let mut bytes = [0; 10];
        bytes.copy_from_slice(&value.to_le_bytes()[..10]);
        emu.write_memory(x87_space(ctx)?, (slot * 10) as u64, &bytes)
            .map_err(|error| error.to_string())
    }

    fn read_x87_slot(
        emu: &mut Emulator<'_>,
        ctx: &Context<'_>,
        slot: usize,
    ) -> Result<u128, String> {
        let bytes = emu
            .read_memory(x87_space(ctx)?, (slot * 10) as u64, 10)
            .map_err(|error| error.to_string())?;
        let mut wide = [0; 16];
        wide[..10].copy_from_slice(&bytes);
        Ok(u128::from_le_bytes(wide))
    }

    fn mmx_index(name: &str) -> Option<usize> {
        name.strip_prefix("mm")
            .or_else(|| name.strip_prefix("MM"))
            .and_then(|index| index.parse::<usize>().ok())
            .filter(|&index| index < X87_PHYSICAL_REGISTERS)
    }

    fn abridged_physical_to_full_tag(tag: u8) -> u16 {
        (0..X87_PHYSICAL_REGISTERS).fold(0, |full, physical| {
            full | if tag & (1 << physical) == 0 {
                3 << (physical * 2)
            } else {
                0
            }
        })
    }

    fn full_to_abridged_physical_tag(tag: u16) -> u8 {
        (0..X87_PHYSICAL_REGISTERS).fold(0, |abridged, physical| {
            abridged | u8::from((tag >> (physical * 2)) & 3 != 3) << physical
        })
    }

    fn x87_control_register(field: &str) -> Option<&'static str> {
        X87_CONTROL_FIELDS
            .iter()
            .find_map(|&(name, register)| (name == field).then_some(register))
    }

    struct DbStateResult {
        initial: DbState,
        /// None = hardware exception (skipped during emulation).
        final_state: Option<DbState>,
    }

    struct DbTestCase {
        id: i64,
        instruction_id: i32,
        instruction: String,
        opcode: String,
        undefined_flags: HashSet<String>,
        states: Vec<DbStateResult>,
    }

    /// One Binit encoding that reaches a constructor.
    #[derive(Debug, Clone, Serialize)]
    struct ConstructorWitness {
        index: usize,
        opcode: String,
        instruction: String,
    }

    /// Coverage details for one SLEIGH table.
    #[derive(Debug, Serialize)]
    struct TableCoverage {
        table: String,
        total_constructors: usize,
        covered_constructors: usize,
        percentage: f64,
        /// One concrete Binit encoding for every covered constructor.
        witnesses: Vec<ConstructorWitness>,
    }

    /// Constructor coverage of Binit's encoded x86-64 corpus.
    ///
    /// Each decoded instruction contributes its root constructor and all
    /// constructors reached recursively through operand tables.
    #[derive(Debug, Serialize)]
    struct ConstructorCoverage {
        architecture: &'static str,
        total_constructors: usize,
        covered_constructors: usize,
        percentage: f64,
        test_cases: usize,
        decoded_test_cases: usize,
        invalid_opcodes: usize,
        decode_failures: usize,
        tables: Vec<TableCoverage>,
    }

    /// A constructor location in a compiled specification.
    #[derive(Debug, Clone, Serialize)]
    struct ConstructorLocation {
        table: String,
        index: usize,
    }

    /// A valid instruction encoding that reaches at least one constructor not
    /// reached by the database corpus at the start of a guided search.
    #[derive(Debug, Serialize)]
    struct ConstructorCandidate {
        opcode: String,
        instruction: String,
        newly_covered: Vec<ConstructorLocation>,
        constructor_path: Vec<ConstructorLocation>,
    }

    /// Small deterministic PRNG for reproducible corpus mutation without a
    /// runtime dependency on `rand`.
    struct XorShift64(u64);

    impl XorShift64 {
        fn next(&mut self) -> u64 {
            let mut value = self.0;
            value ^= value << 13;
            value ^= value >> 7;
            value ^= value << 17;
            self.0 = value;
            value
        }

        fn index(&mut self, upper_bound: usize) -> usize {
            debug_assert!(upper_bound > 0);
            (self.next() as usize) % upper_bound
        }
    }

    impl ConstructorCoverage {
        fn print(&self) {
            eprintln!(
                "[sleigh-coverage] {} constructors: {}/{} ({:.2}%); {} of {} Binit cases decoded ({} invalid opcodes, {} decode failures)",
                self.architecture,
                self.covered_constructors,
                self.total_constructors,
                self.percentage,
                self.decoded_test_cases,
                self.test_cases,
                self.invalid_opcodes,
                self.decode_failures,
            );
            for table in self
                .tables
                .iter()
                .filter(|table| table.covered_constructors > 0)
            {
                eprintln!(
                    "[sleigh-coverage]   {:<32} {}/{} ({:.2}%)",
                    table.table,
                    table.covered_constructors,
                    table.total_constructors,
                    table.percentage,
                );
            }
        }

        fn covered_keys(&self) -> HashSet<(String, usize)> {
            self.tables
                .iter()
                .flat_map(|table| {
                    table
                        .witnesses
                        .iter()
                        .map(|witness| (table.table.clone(), witness.index))
                })
                .collect()
        }

        fn write_json_if_requested(&self) -> Result<(), String> {
            let Some(path) = std::env::var_os("PCODE_SLEIGH_COVERAGE_OUTPUT") else {
                return Ok(());
            };
            let path = PathBuf::from(path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
            let file = File::create(&path).map_err(|error| error.to_string())?;
            serde_json::to_writer_pretty(BufWriter::new(file), self)
                .map_err(|error| error.to_string())?;
            eprintln!("[sleigh-coverage] wrote {}", path.display());
            Ok(())
        }
    }

    #[derive(Debug)]
    struct DbMismatch {
        instruction_id: i32,
        test_case_id: i64,
        state_index: usize,
        instruction: String,
        opcode: String,
        initial_state: DbState,
        expected_state: DbState,
        actual_state: Option<DbState>,
        diff: serde_json::Value,
        actual_outcome: &'static str,
        mismatch_reason: &'static str,
        ignored_flags: Vec<String>,
    }

    #[derive(Serialize)]
    struct MismatchCsvRow {
        tool: &'static str,
        test_case_id: i64,
        state_index: usize,
        instruction: String,
        opcode: String,
        initial_state_json: String,
        expected_outcome: &'static str,
        actual_outcome: &'static str,
        expected_exception: &'static str,
        actual_exception: &'static str,
        expected_state_json: String,
        actual_state_json: String,
        diff_json: String,
        ignored_flags: String,
        mismatch_reason: &'static str,
    }

    struct MismatchCsv {
        writer: csv::Writer<File>,
        failed_instruction_ids: HashSet<i32>,
        mismatch_count: usize,
        /// Written mismatches bucketed by `mismatch_reason` (e.g. "state_mismatch",
        /// "unsupported", "fixture_limitation") so the summary can separate genuine
        /// emulation discrepancies from unmodeled instructions.
        reason_counts: HashMap<&'static str, usize>,
    }

    #[derive(Debug)]
    enum RunError {
        Emulator(EmulatorError),
        StepLimit,
    }

    impl std::fmt::Display for RunError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Emulator(error) => error.fmt(f),
                Self::StepLimit => write!(
                    f,
                    "did not leave the instruction within {MAX_EMULATED_STEPS} qcode steps"
                ),
            }
        }
    }

    fn db_dsn() -> String {
        std::env::var("X86DB_DSN")
            .unwrap_or_else(|_| "postgresql://x86db:x86db@localhost:5432/x86db".to_string())
    }

    fn load_db_test_case(
        client: &mut Client,
        tc_id: i64,
        instruction: &str,
        opcode: &str,
        instruction_id: i32,
    ) -> DbTestCase {
        // Query 1: undefined flags.
        let undef_rows = client
            .query(
                "SELECT flag FROM instruction_undefined_flags WHERE instruction_id = $1",
                &[&instruction_id],
            )
            .expect("failed to query undefined flags");
        let undefined_flags: HashSet<String> =
            undef_rows.iter().map(|r| r.get::<_, String>(0)).collect();

        // Query 2: initial_states jsonb array from test_cases.
        let tc_row = client
            .query_one(
                "SELECT initial_states::text FROM test_cases WHERE id = $1",
                &[&tc_id],
            )
            .expect("failed to query test_case initial_states");
        let initial_states_str: String = tc_row.get(0);
        let initial_states_json: serde_json::Value =
            serde_json::from_str(&initial_states_str).unwrap_or(serde_json::Value::Array(vec![]));
        let initial_states_arr = initial_states_json.as_array().cloned().unwrap_or_default();

        // Query 3: final states from test_results.
        let result_rows = client
            .query(
                "SELECT state_index, exception_kind, final_state::text \
                 FROM test_results \
                 WHERE test_case_id = $1 \
                 ORDER BY state_index",
                &[&tc_id],
            )
            .expect("failed to query test results");

        // Build map: state_index → (exception_kind, final_state jsonb).
        let mut result_map: HashMap<i32, (Option<String>, Option<serde_json::Value>)> =
            HashMap::new();
        for row in &result_rows {
            let idx: i32 = row.get(0);
            let exception_kind: Option<String> = row.get(1);
            let final_state: Option<serde_json::Value> = row
                .get::<_, Option<String>>(2)
                .and_then(|s| serde_json::from_str(&s).ok());
            result_map.insert(idx, (exception_kind, final_state));
        }

        fn parse_state(json: &serde_json::Value) -> DbState {
            let mut regs = HashMap::new();
            let mut f80 = HashMap::new();
            let mut scratch_memory = None;
            if let Some(object) = json.as_object() {
                assert!(
                    !(object.contains_key("scratch_memory")
                        && (object.contains_key("mem0_value")
                            || object.contains_key("mem1_value"))),
                    "scratch_memory cannot be mixed with mem0_value or mem1_value"
                );
                for (name, value) in object {
                    let is_legacy_logical = name
                        .strip_prefix("x87_st")
                        .and_then(|index| index.parse::<usize>().ok())
                        .is_some_and(|index| index < X87_PHYSICAL_REGISTERS);
                    assert!(
                        !is_legacy_logical,
                        "Binit state uses legacy logical {name}; convert it with logical_x87_to_physical_state"
                    );
                    if name == "scratch_memory" {
                        scratch_memory = value.as_str().and_then(parse_scratch_memory);
                        assert!(scratch_memory.is_some(), "invalid scratch_memory encoding");
                    } else if x87_physical_slot_index(name).is_some() {
                        if let Some(value) = value.as_str().and_then(parse_f80) {
                            f80.insert(name.clone(), value);
                        }
                    } else if let Some(value) =
                        value.as_i64().or_else(|| value.as_u64().map(|n| n as i64))
                    {
                        regs.insert(name.clone(), value);
                    }
                }
            }
            DbState {
                regs,
                f80,
                scratch_memory,
            }
        }

        // Combine initial states with final states.
        let states = initial_states_arr
            .iter()
            .enumerate()
            .map(|(idx, init_json)| {
                let initial = parse_state(init_json);
                let final_state = result_map.get(&(idx as i32)).and_then(|(exc, fin)| {
                    if exc.is_some() {
                        None // hardware exception — skip
                    } else {
                        fin.as_ref().map(parse_state)
                    }
                });
                DbStateResult {
                    initial,
                    final_state,
                }
            })
            .collect();

        DbTestCase {
            id: tc_id,
            instruction_id,
            instruction: instruction.to_string(),
            opcode: opcode.to_string(),
            undefined_flags,
            states,
        }
    }

    impl DbMismatch {
        fn backend_error(
            tc: &DbTestCase,
            state_index: usize,
            pair: &DbStateResult,
            expected_state: &DbState,
            error: impl ToString,
        ) -> Self {
            Self::new(
                tc,
                state_index,
                pair,
                expected_state,
                None,
                serde_json::json!({ "backend_error": error.to_string() }),
                "backend_error",
                "backend_error",
            )
        }

        fn state_mismatch(
            tc: &DbTestCase,
            state_index: usize,
            pair: &DbStateResult,
            expected_state: &DbState,
            actual_state: DbState,
            diff: serde_json::Value,
        ) -> Self {
            Self::new(
                tc,
                state_index,
                pair,
                expected_state,
                Some(actual_state),
                diff,
                "normal",
                "state_mismatch",
            )
        }

        fn unsupported(
            tc: &DbTestCase,
            state_index: usize,
            pair: &DbStateResult,
            expected_state: &DbState,
            operation: &str,
        ) -> Self {
            Self::new(
                tc,
                state_index,
                pair,
                expected_state,
                None,
                serde_json::json!({ "unsupported_pcode_op": operation }),
                "unsupported",
                "unsupported",
            )
        }

        fn fixture_limitation(
            tc: &DbTestCase,
            state_index: usize,
            pair: &DbStateResult,
            expected_state: &DbState,
            reason: impl ToString,
        ) -> Self {
            Self::new(
                tc,
                state_index,
                pair,
                expected_state,
                None,
                serde_json::json!({ "fixture_limitation": reason.to_string() }),
                "fixture_limitation",
                "fixture_limitation",
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn new(
            tc: &DbTestCase,
            state_index: usize,
            pair: &DbStateResult,
            expected_state: &DbState,
            actual_state: Option<DbState>,
            diff: serde_json::Value,
            actual_outcome: &'static str,
            mismatch_reason: &'static str,
        ) -> Self {
            let mut ignored_flags: Vec<_> = tc.undefined_flags.iter().cloned().collect();
            ignored_flags.sort();
            Self {
                instruction_id: tc.instruction_id,
                test_case_id: tc.id,
                state_index,
                instruction: tc.instruction.clone(),
                opcode: tc.opcode.clone(),
                initial_state: pair.initial.clone(),
                expected_state: expected_state.clone(),
                actual_state,
                diff,
                actual_outcome,
                mismatch_reason,
                ignored_flags,
            }
        }

        fn into_csv_row(self) -> MismatchCsvRow {
            MismatchCsvRow {
                tool: "wazabin",
                test_case_id: self.test_case_id,
                state_index: self.state_index,
                instruction: self.instruction,
                opcode: self.opcode,
                initial_state_json: format_state_json(Some(&self.initial_state)),
                expected_outcome: "normal",
                actual_outcome: self.actual_outcome,
                expected_exception: "",
                actual_exception: "",
                expected_state_json: format_state_json(Some(&self.expected_state)),
                actual_state_json: format_state_json(self.actual_state.as_ref()),
                diff_json: self.diff.to_string(),
                ignored_flags: self.ignored_flags.join(","),
                mismatch_reason: self.mismatch_reason,
            }
        }
    }

    fn format_state_json(state: Option<&DbState>) -> String {
        let Some(state) = state else {
            return "null".to_string();
        };
        let mut normalized = serde_json::Map::new();
        let mut names: Vec<_> = state.regs.keys().collect();
        names.sort();
        for name in names {
            let value = state.regs[name] as u64;
            if name == "flag" {
                let flags = FLAGS
                    .iter()
                    .map(|&(flag_name, mask)| {
                        (
                            flag_name.to_string(),
                            serde_json::Value::from(u8::from(value & mask != 0)),
                        )
                    })
                    .collect();
                normalized.insert(name.clone(), serde_json::Value::Object(flags));
            } else {
                normalized.insert(
                    name.clone(),
                    serde_json::Value::String(format!("{value:#x}")),
                );
            }
        }
        let mut f80_names: Vec<_> = state.f80.keys().collect();
        f80_names.sort();
        for name in f80_names {
            normalized.insert(
                name.clone(),
                serde_json::Value::String(format_f80(state.f80[name])),
            );
        }
        if let Some(scratch_memory) = &state.scratch_memory {
            normalized.insert(
                "scratch_memory".to_string(),
                serde_json::Value::String(format_scratch_memory(scratch_memory)),
            );
        }
        serde_json::Value::Object(normalized).to_string()
    }

    impl MismatchCsv {
        fn new(path: &Path) -> Self {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("failed to create CSV output directory");
            }
            let file = File::create(path).expect("failed to create mismatch CSV");
            let mut writer = csv::WriterBuilder::new()
                .has_headers(false)
                .from_writer(file);
            writer
                .write_record(CSV_FIELDNAMES)
                .expect("failed to write mismatch CSV header");
            writer.flush().expect("failed to flush mismatch CSV header");
            Self {
                writer,
                failed_instruction_ids: HashSet::new(),
                mismatch_count: 0,
                reason_counts: HashMap::new(),
            }
        }

        fn contains(&self, instruction_id: i32) -> bool {
            self.failed_instruction_ids.contains(&instruction_id)
        }

        fn write_mismatch(&mut self, mismatch: DbMismatch) -> bool {
            if !self.failed_instruction_ids.insert(mismatch.instruction_id) {
                return false;
            }
            let reason = mismatch.mismatch_reason;
            self.writer
                .serialize(mismatch.into_csv_row())
                .expect("failed to write mismatch CSV row");
            self.writer.flush().expect("failed to flush mismatch CSV");
            self.mismatch_count += 1;
            *self.reason_counts.entry(reason).or_default() += 1;
            true
        }
    }

    fn panic_message(payload: Box<dyn Any + Send>) -> String {
        payload
            .downcast_ref::<&str>()
            .map(|message| (*message).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic payload".to_string())
    }

    fn run_to_instruction_boundary(
        emu: &mut Emulator<'_>,
        end_address: u64,
    ) -> Result<(), RunError> {
        for _ in 0..MAX_EMULATED_STEPS {
            if emu.block().address().is_some_and(|address| address >= end_address) {
                return Ok(());
            }
            let Some(insn) = emu.insn() else {
                return Err(RunError::StepLimit);
            };
            if matches!(
                insn.mnemonic(),
                Mnemonic::Call(_)
                    | Mnemonic::CallInd(_)
                    | Mnemonic::BranchInd(_)
                    | Mnemonic::Return(_)
            ) {
                return Ok(());
            }
            emu.step().map_err(RunError::Emulator)?;
        }
        Err(RunError::StepLimit)
    }

    fn snapshot_state(
        emu: &mut Emulator<'_>,
        ctx: &qcode::context::Context<'_>,
        include_mem0: bool,
        include_mem1: bool,
        include_scratch_memory: bool,
        include_x87: bool,
    ) -> Result<DbState, String> {
        let mut regs = HashMap::new();
        let mut f80 = HashMap::new();
        for &name in SCALAR_REGISTERS {
            let id = x64_register(name).ok_or_else(|| format!("unknown register {name}"))?;
            let value = emu
                .read_register(id)
                .ok_or_else(|| format!("could not read register {name}"))?;
            regs.insert(name.to_ascii_lowercase(), value as i64);
        }
        for physical in 0..X87_PHYSICAL_REGISTERS {
            let value = read_x87_slot(emu, ctx, physical)?;
            regs.insert(format!("mm{physical}"), value as i64);
        }

        let mut rflags = 0;
        for &(flag_name, mask) in FLAGS {
            let id = x64_register(flag_name).ok_or_else(|| format!("unknown flag {flag_name}"))?;
            let value = emu
                .read_register(id)
                .ok_or_else(|| format!("could not read flag {flag_name}"))?;
            if value != 0 {
                rflags |= mask;
            }
        }
        regs.insert("flag".to_string(), rflags as i64);
        if let Some(rip) = emu.block().address() {
            regs.insert("rip".to_string(), rip as i64);
        }

        if include_x87 {
            for &(field, name) in X87_CONTROL_FIELDS {
                let id =
                    x64_register(name).ok_or_else(|| format!("unknown x87 register {name}"))?;
                let mut value = emu
                    .read_register(id)
                    .ok_or_else(|| format!("could not read x87 register {name}"))?;
                if field == "x87_tag" {
                    value = u64::from(full_to_abridged_physical_tag(value as u16));
                }
                regs.insert(field.to_string(), value as i64);
            }
            regs.insert(
                "x87_top".to_string(),
                x87_top(regs.get("x87_status").copied().unwrap_or(0) as u64) as i64,
            );
            for physical in 0..X87_PHYSICAL_REGISTERS {
                let value = read_x87_slot(emu, ctx, physical)?;
                f80.insert(format!("x87_r{physical}"), value);
            }
        }

        if include_scratch_memory {
            let bytes = emu
                .inspect_memory(ctx.shared.default_space, MEM0_ADDR, SCRATCH_MEMORY_SIZE)
                .ok_or_else(|| format!("could not read scratch memory at {MEM0_ADDR:#x}"))?;
            let scratch_memory: [u8; SCRATCH_MEMORY_SIZE] = bytes
                .try_into()
                .map_err(|_| "scratch memory has unexpected length".to_string())?;
            return Ok(DbState {
                regs,
                f80,
                scratch_memory: Some(scratch_memory),
            });
        }
        for (name, address, include) in [
            ("mem0_value", MEM0_ADDR, include_mem0),
            ("mem1_value", MEM1_ADDR, include_mem1),
        ] {
            if !include {
                continue;
            }
            let value = emu
                .inspect_memory(ctx.shared.default_space, address, 8)
                .and_then(|bytes| bytes.try_into().ok())
                .map(u64::from_le_bytes)
                .ok_or_else(|| format!("could not read memory at {address:#x}"))?;
            regs.insert(name.to_string(), value as i64);
        }
        Ok(DbState {
            regs,
            f80,
            scratch_memory: None,
        })
    }

    /// Returns a fixture-limitation reason when the instruction in this state
    /// accesses memory outside the harness's modeled 8-byte `mem0` window. Such
    /// divergences reflect the harness seeding only 8 bytes (and zeroing the rest
    /// of the page) rather than a wazabin bug — the lifted qcode is correct.
    fn fixture_limitation_reason(tc: &DbTestCase, pair: &DbStateResult) -> Option<&'static str> {
        let mnemonic = tc.instruction.split_whitespace().next().unwrap_or("");
        // CMPXCHG16B compares/writes a 16-byte operand; only the low 8 bytes are
        // modeled, so the unmodeled high 8 bytes drive the divergence.
        if mnemonic.eq_ignore_ascii_case("cmpxchg16b") {
            return Some("128-bit memory operand exceeds modeled 8-byte window");
        }
        // BT/BTC/BTR/BTS with a memory operand and a register bit index access the
        // byte at base + (index s>> 3), which can fall outside the modeled window.
        // Binit seeds these forms through the 512-byte `scratch_memory` transport
        // with the operand at its centre, so the whole reachable range is modelled
        // and the case is a real result; a case still carrying only the 8-byte
        // `mem0_value` word predates that and cannot be compared.
        if matches!(mnemonic, "bt" | "btc" | "btr" | "bts") && tc.instruction.contains("ptr") {
            let bit_index = tc
                .instruction
                .rsplit(',')
                .next()
                .map(str::trim)
                .and_then(|reg| pair.initial.regs.get(&reg.to_ascii_lowercase()).copied());
            if let Some(value) = bit_index {
                let byte_offset = value >> 3; // signed, matches SLEIGH `s>> 3`
                // Offsets are relative to the operand, which binit places at
                // BIT_INDEX_OPERAND_OFFSET within the window; the widest operand
                // reads eight bytes from there.
                let modeled = if pair.initial.scratch_memory.is_some() {
                    let centre = BIT_INDEX_OPERAND_OFFSET as i64;
                    -centre..(SCRATCH_MEMORY_SIZE as i64 - centre - 8)
                } else {
                    0..8
                };
                if !modeled.contains(&byte_offset) {
                    return Some("bit index addresses memory outside modeled window");
                }
            }
        }
        None
    }

    /// SHLD/SHRD leave their destination operand undefined when the shift count
    /// exceeds the operand bit-width (Intel SDM). This is only reachable for
    /// 16-bit operands, whose count masks to at most 31.
    enum UndefinedShldShrdDestination {
        Register(String),
        Memory,
    }

    fn shld_shrd_undefined_dest(
        tc: &DbTestCase,
        pair: &DbStateResult,
    ) -> Option<UndefinedShldShrdDestination> {
        let insn = &tc.instruction;
        let mnemonic = insn.split_whitespace().next().unwrap_or("");
        if !matches!(mnemonic, "shld" | "shrd") {
            return None;
        }
        let operands: Vec<&str> = insn[mnemonic.len()..].split(',').map(str::trim).collect();
        if operands.len() != 3 {
            return None;
        }
        // Only 16-bit destinations can carry a (5-bit-masked) count above their width.
        let is_word_destination = canonical_reg16(operands[0]).is_some()
            || operands[0].trim_start().starts_with("word ptr");
        if !is_word_destination {
            return None;
        }
        let raw_count = if operands[2].eq_ignore_ascii_case("cl") {
            (pair.initial.regs.get("rcx").copied().unwrap_or(0) as u64) & 0xff
        } else {
            parse_signed_imm(operands[2])?
        };
        if (raw_count & 0x1f) <= 16 {
            return None;
        }
        canonical_reg16(operands[0])
            .map(UndefinedShldShrdDestination::Register)
            .or_else(|| {
                operands[0]
                    .contains("ptr")
                    .then_some(UndefinedShldShrdDestination::Memory)
            })
    }

    /// Maps a 16-bit register operand to its canonical 64-bit name (lowercase),
    /// or `None` if the operand isn't a 16-bit register.
    fn canonical_reg16(operand: &str) -> Option<String> {
        let lower = operand.to_ascii_lowercase();
        let name = match lower.as_str() {
            "ax" => "rax",
            "bx" => "rbx",
            "cx" => "rcx",
            "dx" => "rdx",
            "si" => "rsi",
            "di" => "rdi",
            "bp" => "rbp",
            "sp" => "rsp",
            other => {
                return other
                    .strip_prefix('r')
                    .and_then(|rest| rest.strip_suffix('w'))
                    .and_then(|n| n.parse::<u8>().ok())
                    .filter(|n| (8..=15).contains(n))
                    .map(|n| format!("r{n}"));
            }
        };
        Some(name.to_string())
    }

    /// Parses a possibly-negative immediate operand (`-0x1`, `0x7f`, `5`) into the
    /// raw bit pattern used as the shift count.
    fn parse_signed_imm(s: &str) -> Option<u64> {
        let (neg, body) = s.strip_prefix('-').map_or((false, s), |rest| (true, rest));
        let value = if let Some(hex) = body.strip_prefix("0x") {
            i64::from_str_radix(hex, 16).ok()?
        } else {
            body.parse::<i64>().ok()?
        };
        Some(if neg {
            value.wrapping_neg() as u64
        } else {
            value as u64
        })
    }

    // DbMismatch is a large record; box it so the common `Ok` return stays small.
    fn run_db_case(tc: &DbTestCase) -> Result<(), Box<DbMismatch>> {
        let bytes = parse_hex_bytes(&tc.opcode)
            .unwrap_or_else(|| panic!("{}: invalid opcode hex '{}'", tc.instruction, tc.opcode));

        // The opcode bytes are identical across every state of a test case, so the
        // disassembly + lifting only depends on `bytes` and the load address. Do it
        // once here instead of redoing it for all ~150 states — this lifting pipeline
        // (sleigh walker + pcode expansion + qcode context) dominates the profile.
        // Lifting errors are attributed to the first state that would be emulated.
        let Some((first_index, first_pair)) = tc
            .states
            .iter()
            .enumerate()
            .find(|(_, pair)| pair.final_state.is_some())
        else {
            return Ok(()); // every state is a hardware exception — nothing to emulate
        };
        let first_final = first_pair
            .final_state
            .as_ref()
            .expect("find() guaranteed a final state is present");

        let instruction_addr = instruction_addr(&bytes);
        let ctx = catch_unwind(AssertUnwindSafe(|| {
            lift_x64(&bytes, instruction_addr as u64)
        }))
        .map_err(|payload| {
            DbMismatch::backend_error(
                tc,
                first_index,
                first_pair,
                first_final,
                format!("lifting panicked: {}", panic_message(payload)),
            )
        })?
        .map_err(|error| {
            DbMismatch::backend_error(tc, first_index, first_pair, first_final, error)
        })?;

        for (state_index, pair) in tc.states.iter().enumerate() {
            let final_state = match &pair.final_state {
                None => continue, // hardware exception — skip
                Some(s) => s,
            };
            let include_mem0 = pair.initial.regs.contains_key("mem0_value")
                || final_state.regs.contains_key("mem0_value");
            let include_mem1 = pair.initial.regs.contains_key("mem1_value")
                || final_state.regs.contains_key("mem1_value");
            let include_scratch_memory =
                pair.initial.scratch_memory.is_some() || final_state.scratch_memory.is_some();
            let include_x87 = !pair.initial.f80.is_empty()
                || !final_state.f80.is_empty()
                || pair.initial.regs.contains_key("x87_top")
                || final_state.regs.contains_key("x87_top")
                || X87_CONTROL_FIELDS.iter().any(|&(field, _)| {
                    pair.initial.regs.contains_key(field) || final_state.regs.contains_key(field)
                });
            // Mismatches that stem from the harness's modeling limits (rather than a
            // wazabin bug) are reclassified so they are recorded distinctly from real
            // state mismatches.
            let fixture_reason = fixture_limitation_reason(tc, pair);
            // When set, this SHLD/SHRD state has count > operand size, so both the
            // destination register and all flags are architecturally undefined.
            let shld_shrd_undefined = shld_shrd_undefined_dest(tc, pair);
            let classify = |actual_state: DbState, diff: serde_json::Value| {
                Box::new(match fixture_reason {
                    Some(reason) => {
                        DbMismatch::fixture_limitation(tc, state_index, pair, final_state, reason)
                    }
                    None => DbMismatch::state_mismatch(
                        tc,
                        state_index,
                        pair,
                        final_state,
                        actual_state,
                        diff,
                    ),
                })
            };

            let mut emu = Emulator::from_address(&ctx, instruction_addr as u64);

            emu.write_memory(
                ctx.shared.default_space,
                MEM0_PAGE_ADDR,
                &[0; MEM0_PAGE_SIZE],
            )
            .map_err(|error| {
                DbMismatch::backend_error(tc, state_index, pair, final_state, error)
            })?;

            if let Some(scratch_memory) = &pair.initial.scratch_memory {
                emu.write_memory(ctx.shared.default_space, MEM0_ADDR, scratch_memory)
                    .map_err(|error| {
                        DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                    })?;
            }

            for &(memory_name, address) in MEMORY_WORDS {
                if !pair.initial.regs.contains_key(memory_name)
                    && !final_state.regs.contains_key(memory_name)
                {
                    continue;
                }
                let value = pair.initial.regs.get(memory_name).copied().unwrap_or(0) as u64;
                emu.write_memory(ctx.shared.default_space, address, &value.to_le_bytes())
                    .map_err(|error| {
                        DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                    })?;
            }

            // Aegis executes from CpuState::zero(). Seed the same scalar and
            // physical x87/MMX baseline because emulator storage is lazy.
            for &name in SCALAR_REGISTERS {
                let id = x64_register(name).expect("scalar register is present in the x64 spec");
                emu.set_register(id, 0).map_err(|error| {
                    DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                })?;
            }
            emu.write_memory(
                x87_space(&ctx).map_err(|error| {
                    DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                })?,
                0,
                &[0; X87_PHYSICAL_REGISTERS * 10],
            )
            .map_err(|error| {
                DbMismatch::backend_error(tc, state_index, pair, final_state, error)
            })?;

            // Set initial general-purpose registers.
            for (name, &raw) in &pair.initial.regs {
                if name == "flag"
                    || name == "rip"
                    || mmx_index(name).is_some()
                    || name == "x87_top"
                    || x87_control_register(name).is_some()
                    || MEMORY_WORDS
                        .iter()
                        .any(|&(memory_name, _)| name == memory_name)
                {
                    continue;
                }
                let upper = name.to_ascii_uppercase();
                let Some(id) = x64_register(&upper) else {
                    panic!(
                        "{}[{state_index}]: unknown register {upper}",
                        tc.instruction
                    )
                };
                emu.set_register(id, raw as u64).map_err(|error| {
                    DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                })?;
            }

            if include_x87 {
                let initial_top =
                    x87_top(pair.initial.regs.get("x87_status").copied().unwrap_or(0) as u64);
                if let Some(&explicit_top) = pair.initial.regs.get("x87_top") {
                    if explicit_top as usize != initial_top {
                        return Err(Box::new(DbMismatch::backend_error(
                            tc,
                            state_index,
                            pair,
                            final_state,
                            format!(
                                "x87_top ({explicit_top}) conflicts with x87_status.TOP ({initial_top})"
                            ),
                        )));
                    }
                }
                for &(field, register) in X87_CONTROL_FIELDS {
                    let Some(&raw) = pair.initial.regs.get(field) else {
                        continue;
                    };
                    let id = x64_register(register).ok_or_else(|| {
                        DbMismatch::backend_error(
                            tc,
                            state_index,
                            pair,
                            final_state,
                            format!("x87 register {register} missing from x64 spec"),
                        )
                    })?;
                    let value = if field == "x87_tag" {
                        u64::from(abridged_physical_to_full_tag(raw as u8))
                    } else {
                        raw as u64
                    };
                    emu.set_register(id, value).map_err(|error| {
                        DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                    })?;
                }
                for (name, &value) in &pair.initial.f80 {
                    let Some(physical) = x87_physical_slot_index(name) else {
                        return Err(Box::new(DbMismatch::backend_error(
                            tc,
                            state_index,
                            pair,
                            final_state,
                            format!("invalid physical x87 state key {name}"),
                        )));
                    };
                    write_x87_slot(&mut emu, &ctx, physical, value).map_err(|error| {
                        DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                    })?;
                }
            }

            // MMX is the low-64 view of physical R0..R7 and can be present in
            // states that do not otherwise request x87 comparison.
            for (name, &raw) in &pair.initial.regs {
                let Some(physical) = mmx_index(name) else {
                    continue;
                };
                let mut slot = read_x87_slot(&mut emu, &ctx, physical).map_err(|error| {
                    DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                })?;
                slot = (slot & !u128::from(u64::MAX)) | u128::from(raw as u64);
                write_x87_slot(&mut emu, &ctx, physical, slot).map_err(|error| {
                    DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                })?;
            }

            // Set initial flags from the "flag" RFLAGS word.
            let rflags = pair.initial.regs.get("flag").copied().unwrap_or(0) as u64;
            for &(flag_name, mask) in FLAGS {
                let bit = if rflags & mask != 0 { 1u64 } else { 0u64 };
                if let Some(id) = x64_register(flag_name) {
                    emu.set_register(id, bit).map_err(|error| {
                        DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                    })?;
                }
            }

            // Keep looping instruction semantics bounded, for example REP forms
            // with very large counters.
            catch_unwind(AssertUnwindSafe(|| {
                run_to_instruction_boundary(&mut emu, instruction_addr as u64 + bytes.len() as u64)
            }))
                .map_err(|payload| {
                    DbMismatch::backend_error(
                        tc,
                        state_index,
                        pair,
                        final_state,
                        format!("emulation panicked: {}", panic_message(payload)),
                    )
                })?
                .map_err(|error| match error {
                    RunError::Emulator(EmulatorError {
                        kind: EmulatorErrorKind::UnsupportedPCodeOp(operation),
                        ..
                    }) => DbMismatch::unsupported(tc, state_index, pair, final_state, &operation),
                    error => DbMismatch::backend_error(tc, state_index, pair, final_state, error),
                })?;

            // Compare final general-purpose registers and the control-flow stop RIP.
            for (name, &raw) in &final_state.regs {
                if name == "rip" {
                    let expected = raw as u64;
                    let actual = emu.block().address().ok_or_else(|| {
                        DbMismatch::backend_error(
                            tc,
                            state_index,
                            pair,
                            final_state,
                            "emulator stopped outside an addressable successor block",
                        )
                    })?;
                    if actual != expected {
                        let actual_state = snapshot_state(
                            &mut emu,
                            &ctx,
                            include_mem0,
                            include_mem1,
                            include_scratch_memory,
                            include_x87,
                        )
                        .map_err(|error| {
                            DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                        })?;
                        return Err(classify(
                            actual_state,
                            serde_json::json!({
                                "rip": {
                                    "actual": format!("{actual:#x}"),
                                    "expected": format!("{expected:#x}"),
                                }
                            }),
                        ));
                    }
                    continue;
                }
                if name == "x87_top" {
                    let status_id = x64_register("FPUStatusWord").ok_or_else(|| {
                        DbMismatch::backend_error(
                            tc,
                            state_index,
                            pair,
                            final_state,
                            "x87 status register missing from x64 spec",
                        )
                    })?;
                    let actual = x87_top(emu.read_register(status_id).ok_or_else(|| {
                        DbMismatch::backend_error(
                            tc,
                            state_index,
                            pair,
                            final_state,
                            "could not read x87 status register",
                        )
                    })?);
                    if actual != raw as usize {
                        let actual_state = snapshot_state(
                            &mut emu,
                            &ctx,
                            include_mem0,
                            include_mem1,
                            include_scratch_memory,
                            include_x87,
                        )
                        .map_err(|error| {
                            DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                        })?;
                        return Err(classify(
                            actual_state,
                            serde_json::json!({
                                "x87_top": { "actual": actual, "expected": raw }
                            }),
                        ));
                    }
                    continue;
                }
                if let Some(register) = x87_control_register(name) {
                    let id = x64_register(register).ok_or_else(|| {
                        DbMismatch::backend_error(
                            tc,
                            state_index,
                            pair,
                            final_state,
                            format!("x87 register {register} missing from x64 spec"),
                        )
                    })?;
                    let mut actual = emu.read_register(id).ok_or_else(|| {
                        DbMismatch::backend_error(
                            tc,
                            state_index,
                            pair,
                            final_state,
                            format!("could not read x87 register {register}"),
                        )
                    })?;
                    if name == "x87_tag" {
                        actual = u64::from(full_to_abridged_physical_tag(actual as u16));
                    }
                    if actual != raw as u64 {
                        let actual_state = snapshot_state(
                            &mut emu,
                            &ctx,
                            include_mem0,
                            include_mem1,
                            include_scratch_memory,
                            include_x87,
                        )
                        .map_err(|error| {
                            DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                        })?;
                        return Err(classify(
                            actual_state,
                            serde_json::json!({
                                (name): { "actual": format!("{actual:#x}"), "expected": format!("{:#x}", raw as u64) }
                            }),
                        ));
                    }
                    continue;
                }
                if let Some(physical) = mmx_index(name) {
                    let actual = read_x87_slot(&mut emu, &ctx, physical).map_err(|error| {
                        DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                    })? as u64;
                    if actual != raw as u64 {
                        let actual_state = snapshot_state(
                            &mut emu,
                            &ctx,
                            include_mem0,
                            include_mem1,
                            include_scratch_memory,
                            include_x87,
                        )
                        .map_err(|error| {
                            DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                        })?;
                        return Err(classify(
                            actual_state,
                            serde_json::json!({
                                (name): {
                                    "actual": format!("{actual:#x}"),
                                    "expected": format!("{:#x}", raw as u64),
                                }
                            }),
                        ));
                    }
                    continue;
                }
                if name == "flag"
                    || MEMORY_WORDS
                        .iter()
                        .any(|&(memory_name, _)| name == memory_name)
                {
                    continue;
                }

                // Hacky, but ignore rax for bsf/bsr when input is zero (undefined output).
                if (name == "rax")
                    && (tc.instruction.starts_with("bsf") || tc.instruction.starts_with("bsr"))
                    && (pair.initial.regs.get("rbx").copied().unwrap_or(0) == 0)
                {
                    continue;
                }

                // shld/shrd leave the destination undefined when the shift count
                // exceeds the operand bit-width (only reachable for 16-bit operands).
                if matches!(
                    &shld_shrd_undefined,
                    Some(UndefinedShldShrdDestination::Register(dest)) if dest == name
                ) {
                    continue;
                }

                let upper = name.to_ascii_uppercase();
                let Some(id) = x64_register(&upper) else {
                    continue;
                };

                let expected = raw as u64;
                let actual = emu.read_register(id).ok_or_else(|| {
                    DbMismatch::backend_error(
                        tc,
                        state_index,
                        pair,
                        final_state,
                        format!("could not read register {upper}"),
                    )
                })?;
                if actual != expected {
                    let actual_state = snapshot_state(
                        &mut emu,
                        &ctx,
                        include_mem0,
                        include_mem1,
                        include_scratch_memory,
                        include_x87,
                    )
                    .map_err(|error| {
                        DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                    })?;
                    return Err(classify(
                        actual_state,
                        serde_json::json!({
                            (name): {
                                "actual": format!("{actual:#x}"),
                                "expected": format!("{expected:#x}"),
                            }
                        }),
                    ));
                }
            }

            for (name, &expected) in &final_state.f80 {
                let Some(physical) = x87_physical_slot_index(name) else {
                    return Err(Box::new(DbMismatch::backend_error(
                        tc,
                        state_index,
                        pair,
                        final_state,
                        format!("invalid physical x87 state key {name}"),
                    )));
                };
                let actual = read_x87_slot(&mut emu, &ctx, physical).map_err(|error| {
                    DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                })?;
                if actual != expected {
                    let actual_state = snapshot_state(
                        &mut emu,
                        &ctx,
                        include_mem0,
                        include_mem1,
                        include_scratch_memory,
                        include_x87,
                    )
                    .map_err(|error| {
                        DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                    })?;
                    return Err(classify(
                        actual_state,
                        serde_json::json!({
                            (name): { "actual": format_f80(actual), "expected": format_f80(expected) }
                        }),
                    ));
                }
            }

            if let Some(expected) = &final_state.scratch_memory {
                let actual = emu
                    .inspect_memory(ctx.shared.default_space, MEM0_ADDR, SCRATCH_MEMORY_SIZE)
                    .ok_or_else(|| {
                        DbMismatch::backend_error(
                            tc,
                            state_index,
                            pair,
                            final_state,
                            format!("could not read scratch memory at {MEM0_ADDR:#x}"),
                        )
                    })?;
                if actual != expected {
                    let actual_state = snapshot_state(
                        &mut emu,
                        &ctx,
                        include_mem0,
                        include_mem1,
                        include_scratch_memory,
                        include_x87,
                    )
                    .map_err(|error| {
                        DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                    })?;
                    let changed_offsets: Vec<_> = actual
                        .iter()
                        .zip(expected)
                        .enumerate()
                        .filter_map(|(offset, (actual, expected))| {
                            (actual != expected).then_some(offset)
                        })
                        .collect();
                    return Err(classify(
                        actual_state,
                        serde_json::json!({
                            "scratch_memory": {
                                "changed_offsets": changed_offsets,
                                "expected": format_scratch_memory(expected),
                                "actual": actual.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
                            }
                        }),
                    ));
                }
            }

            for &(memory_name, address) in MEMORY_WORDS {
                let Some(&raw) = final_state.regs.get(memory_name) else {
                    continue;
                };
                if matches!(
                    shld_shrd_undefined,
                    Some(UndefinedShldShrdDestination::Memory)
                ) {
                    continue;
                }
                let actual = emu
                    .inspect_memory(ctx.shared.default_space, address, 8)
                    .and_then(|bytes| bytes.try_into().ok())
                    .map(u64::from_le_bytes)
                    .ok_or_else(|| {
                        DbMismatch::backend_error(
                            tc,
                            state_index,
                            pair,
                            final_state,
                            format!("could not read memory at {address:#x}"),
                        )
                    })?;
                let expected = raw as u64;
                if actual != expected {
                    let actual_state = snapshot_state(
                        &mut emu,
                        &ctx,
                        include_mem0,
                        include_mem1,
                        include_scratch_memory,
                        include_x87,
                    )
                    .map_err(|error| {
                        DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                    })?;
                    return Err(classify(
                        actual_state,
                        serde_json::json!({
                            (memory_name): {
                                "actual": format!("{actual:#x}"),
                                "expected": format!("{expected:#x}"),
                            }
                        }),
                    ));
                }
            }

            // Compare final flags (skip undefined ones).
            let final_rflags = final_state.regs.get("flag").copied().unwrap_or(0) as u64;
            for &(flag_name, mask) in FLAGS {
                if tc.undefined_flags.contains(flag_name) {
                    continue;
                }
                // shld/shrd with count > operand size leave all flags undefined.
                if shld_shrd_undefined.is_some() {
                    continue;
                }
                let expected_bit = if final_rflags & mask != 0 { 1u64 } else { 0u64 };
                let Some(id) = x64_register(flag_name) else {
                    continue;
                };
                let actual = emu.read_register(id).ok_or_else(|| {
                    DbMismatch::backend_error(
                        tc,
                        state_index,
                        pair,
                        final_state,
                        format!("could not read flag {flag_name}"),
                    )
                })?;
                if actual != expected_bit {
                    let actual_state = snapshot_state(
                        &mut emu,
                        &ctx,
                        include_mem0,
                        include_mem1,
                        include_scratch_memory,
                        include_x87,
                    )
                    .map_err(|error| {
                        DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                    })?;
                    return Err(classify(
                        actual_state,
                        serde_json::json!({
                            "flag": {
                                flag_name: {
                                    "actual": actual,
                                    "expected": expected_bit,
                                }
                            }
                        }),
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn run_db_corpus(max_states: Option<usize>) {
        let dsn = db_dsn();

        // Open main-thread connection and load scalar test cases with Aegis results.
        let mut main_client =
            Client::connect(&dsn, NoTls).expect("failed to connect to x86db (is DB running?)");

        // Optionally replay only selected cases. This makes it practical to
        // validate newly admitted encodings without replaying the full corpus.
        let selected_case_ids: Vec<i64> = std::env::var("PCODE_FUZZ_TEST_CASE_IDS")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter(|value| !value.is_empty())
                    .map(|value| value.parse::<i64>())
                    .collect::<Result<_, _>>()
                    .expect("PCODE_FUZZ_TEST_CASE_IDS must be comma-separated integers")
            })
            .unwrap_or_default();

        // The per-case state count (one row in test_results per state) is what makes
        // up the real workload: each case carries ~150 states. Fetch it alongside the
        // case so progress, throughput, and the cap can be reported in states.
        let rows = main_client
            .query(
                "SELECT tc.id, tc.instruction, tc.opcode, tc.instruction_id, \
                        (SELECT count(*) FROM test_results tr WHERE tr.test_case_id = tc.id) \
                 FROM test_cases tc \
                 WHERE EXISTS ( \
                     SELECT 1 FROM test_results tr WHERE tr.test_case_id = tc.id \
                 ) \
                   AND NOT EXISTS ( \
                     SELECT 1 \
                     FROM instruction_tags it \
                     WHERE it.instruction_id = tc.instruction_id AND it.tag = 'avx' \
                 ) \
                   AND (cardinality($1::bigint[]) = 0 OR tc.id = ANY($1)) \
                 ORDER BY tc.id",
                &[&selected_case_ids],
            )
            .expect("failed to load scalar test cases");

        // The replay unit is a *state*, not a case. Cases stay the schedulable chunk
        // because lifting is shared across a case's states, so the cap admits whole
        // cases (in id order) until the requested state budget is reached.
        let max_states = max_states.unwrap_or(usize::MAX) as u64;
        let mut replay_cases: Vec<(i64, String, String, i32, u64)> = Vec::new();
        let mut total_states: u64 = 0;
        for row in &rows {
            let n_states = row.get::<_, i64>(4).max(0) as u64;
            replay_cases.push((row.get(0), row.get(1), row.get(2), row.get(3), n_states));
            total_states += n_states;
            if total_states >= max_states {
                break;
            }
        }

        drop(main_client); // release connection before spawning threads

        if replay_cases.is_empty() {
            eprintln!("[db-fuzz] no scalar test cases with Aegis results");
            return;
        }

        let num_threads = std::env::var("PCODE_FUZZ_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            });

        let bar_style = ProgressStyle::with_template(
            "worker {prefix} {percent:>3}% {bar:40.cyan/blue} {human_pos:>12}/{human_len:7} [{elapsed}<{eta}, {per_sec}]",
        )
        .unwrap()
        .progress_chars("##-");

        let multi =
            MultiProgress::with_draw_target(indicatif::ProgressDrawTarget::stderr_with_hz(5));
        let total_ok = AtomicUsize::new(0);
        let total_err = AtomicUsize::new(0);
        let total_skipped = AtomicUsize::new(0);
        let output = std::env::var_os("PCODE_FUZZ_OUTPUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("candidates/wazabin_mismatches.csv"));
        let mismatches = Mutex::new(MismatchCsv::new(&output));
        let chunk_size = replay_cases.len().div_ceil(num_threads.max(1));

        eprintln!(
            "[db-fuzz] replaying {} states across {} scalar test cases on {} threads",
            total_states,
            replay_cases.len(),
            num_threads
        );

        std::thread::scope(|s| {
            for (worker_id, chunk) in replay_cases.chunks(chunk_size).enumerate() {
                let chunk_states: u64 = chunk.iter().map(|c| c.4).sum();
                let bar = multi.add(ProgressBar::new(chunk_states));
                bar.set_style(bar_style.clone());
                bar.set_prefix(format!("{worker_id}"));
                let dsn = &dsn;
                let total_ok = &total_ok;
                let total_err = &total_err;
                let total_skipped = &total_skipped;
                let mismatches = &mismatches;
                s.spawn(move || {
                    let mut client =
                        Client::connect(dsn, NoTls).expect("worker failed to connect to x86db");
                    for (tc_id, instruction, opcode, instruction_id, n_states) in chunk {
                        if mismatches
                            .lock()
                            .expect("mismatch mutex was poisoned")
                            .contains(*instruction_id)
                        {
                            total_skipped.fetch_add(1, Ordering::Relaxed);
                            bar.inc(*n_states);
                            continue;
                        }
                        let tc = load_db_test_case(
                            &mut client,
                            *tc_id,
                            instruction,
                            opcode,
                            *instruction_id,
                        );
                        match run_db_case(&tc) {
                            Ok(()) => {
                                total_ok.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(mismatch) => {
                                let written = mismatches
                                    .lock()
                                    .expect("mismatch mutex was poisoned")
                                    .write_mismatch(*mismatch);
                                if written {
                                    total_err.fetch_add(1, Ordering::Relaxed);
                                } else {
                                    total_skipped.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                        bar.inc(*n_states);
                    }
                    bar.finish();
                });
            }
        });

        let ok = total_ok.load(Ordering::Relaxed);
        let err = total_err.load(Ordering::Relaxed);
        let skipped = total_skipped.load(Ordering::Relaxed);
        let csv = mismatches
            .into_inner()
            .expect("mismatch mutex was poisoned");
        let mismatch_count = csv.mismatch_count;
        let state_mismatches = csv
            .reason_counts
            .get("state_mismatch")
            .copied()
            .unwrap_or(0);
        let unsupported = csv.reason_counts.get("unsupported").copied().unwrap_or(0);
        eprintln!(
            "[db-fuzz] done: {ok} cases OK, {state_mismatches} errors, {unsupported} unsupported instructions, {skipped} cases skipped after a matching error, {mismatch_count} deduplicated mismatches written to {}",
            output.display()
        );
        if err > 0 && std::env::var_os("PCODE_FUZZ_FAIL_QUIETLY").is_none() {
            panic!("{err} scalar DB emulation cases failed");
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn reports_recursive_constructor_matches() {
            let spec = x64::spec();
            let context = spec.new_context();
            let instruction = Decoder::new(spec)
                .decode_one(0, b"\x48\x89\xd8", &context)
                .unwrap();
            assert!(
                instruction.constructor_matches().count() > 1,
                "a register move should enter operand tables as well as the root instruction table"
            );
        }

        #[test]
        fn lifts_and_emulates_a_register_move() {
            let tc = DbTestCase {
                id: 0,
                instruction_id: 0,
                instruction: "mov rax, rbx".into(),
                opcode: "4889d8".into(),
                undefined_flags: HashSet::new(),
                states: vec![DbStateResult {
                    initial: DbState {
                        regs: HashMap::from([("rbx".into(), 1)]),
                        f80: HashMap::new(),
                        scratch_memory: None,
                    },
                    final_state: Some(DbState {
                        regs: HashMap::from([("rax".into(), 1)]),
                        f80: HashMap::new(),
                        scratch_memory: None,
                    }),
                }],
            };
            run_db_case(&tc).unwrap();
        }

        #[test]
        fn bounded_scratch_memory_is_seeded_and_compared() {
            let mut initial_scratch = [0; SCRATCH_MEMORY_SIZE];
            initial_scratch[32] = 0xa5;
            let mut expected_scratch = initial_scratch;
            expected_scratch[..8].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
            let tc = DbTestCase {
                id: 0,
                instruction_id: 0,
                instruction: "mov qword ptr [rbx], rax".into(),
                opcode: "488903".into(),
                undefined_flags: HashSet::new(),
                states: vec![DbStateResult {
                    initial: DbState {
                        regs: HashMap::from([
                            ("rbx".into(), MEM0_ADDR as i64),
                            ("rax".into(), 0x1122_3344_5566_7788u64 as i64),
                        ]),
                        f80: HashMap::new(),
                        scratch_memory: Some(initial_scratch),
                    },
                    final_state: Some(DbState {
                        regs: HashMap::new(),
                        f80: HashMap::new(),
                        scratch_memory: Some(expected_scratch),
                    }),
                }],
            };
            run_db_case(&tc).unwrap();
        }

        #[test]
        #[ignore = "Requires a running Binit PostgreSQL database."]
        fn binit_undefined_flags_are_explicit_undef_assignments() {
            let mut client = Client::connect(&db_dsn(), NoTls).unwrap();
            let rows = client
                .query(
                    "SELECT tc.id, tc.instruction, tc.opcode, iuf.flag \
                     FROM test_cases tc \
                     JOIN instruction_undefined_flags iuf \
                       ON iuf.instruction_id = tc.instruction_id \
                     ORDER BY tc.id, iuf.flag",
                    &[],
                )
                .unwrap();
            let mut cases: BTreeMap<i64, (String, String, Vec<String>)> = BTreeMap::new();
            for row in rows {
                let id: i64 = row.get(0);
                let instruction: String = row.get(1);
                let opcode: String = row.get(2);
                let flag: String = row.get(3);
                let entry = cases
                    .entry(id)
                    .or_insert_with(|| (instruction, opcode, Vec::new()));
                entry.2.push(flag);
            }
            assert!(
                !cases.is_empty(),
                "the Binit dataset has no instructions with undefined flags"
            );

            let spec = x64::spec();
            let undef_id =
                spec.pcode_ops()
                    .position(|name| name == "undef")
                    .expect("the x86 SLEIGH specification defines undef") as u64;
            // Every Binit case is checked, but report one concise bucket per
            // mnemonic/flag pair: operand permutations otherwise turn a single
            // missing SLEIGH flag write into thousands of duplicate diagnostics.
            let mut missing: BTreeMap<(String, String), Vec<i64>> = BTreeMap::new();
            for (id, (instruction_text, opcode, undefined_flags)) in cases {
                let bytes = parse_hex_bytes(&opcode)
                    .unwrap_or_else(|| panic!("{id} {instruction_text}: invalid opcode {opcode}"));
                let instruction = Decoder::new(spec)
                    .decode_one(FIRE_START as u64, &bytes, &spec.new_context())
                    .unwrap_or_else(|error| panic!("{id} {instruction_text}: {error}"));
                let pcode = instruction
                    .pcode_ops()
                    .unwrap_or_else(|error| panic!("{id} {instruction_text}: {error}"));

                for flag_name in undefined_flags {
                    let flag = spec
                        .registers()
                        .find(|register| register.name().eq_ignore_ascii_case(&flag_name))
                        .unwrap_or_else(|| {
                            panic!("{id} {instruction_text}: unknown undefined flag {flag_name}")
                        });
                    let flag = Varnode::new(flag.space(), flag.offset() as u64, flag.size());
                    if !pcode.ops.iter().any(|op| {
                        op.opcode == Opcode::CallOther
                            && op.output == Some(flag)
                            && op.inputs.len() == 1
                            && op.inputs[0].is_constant()
                            && op.inputs[0].offset == undef_id
                    }) {
                        let mnemonic = instruction_text
                            .split_whitespace()
                            .next()
                            .unwrap_or("<empty>")
                            .to_ascii_uppercase();
                        missing.entry((mnemonic, flag_name)).or_default().push(id);
                    }
                }
            }
            let missing = missing
                .into_iter()
                .map(|((mnemonic, flag), ids)| {
                    let examples = ids
                        .iter()
                        .take(5)
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!(
                        "{mnemonic}: {flag} missing in {} Binit cases (for example IDs {examples})",
                        ids.len()
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                missing.is_empty(),
                "Binit undefined-flag writes missing from SLEIGH:\n{missing}"
            );
        }

        #[test]
        #[ignore = "Requires a running Binit PostgreSQL database with Aegis results."]
        fn binit_constructor_coverage() {
            let coverage = collect_constructor_coverage().unwrap();
            coverage.print();
            coverage.write_json_if_requested().unwrap();
        }

        #[test]
        #[ignore = "Requires a running Binit PostgreSQL database with Aegis results."]
        fn binit_constructor_candidates() {
            generate_constructor_candidates().unwrap();
        }

        #[test]
        #[ignore = "Requires a running Binit PostgreSQL database with Aegis results."]
        fn binit_smoke() {
            let coverage = collect_constructor_coverage().unwrap();
            coverage.print();
            coverage.write_json_if_requested().unwrap();
            run_db_corpus(Some(10));
        }

        #[test]
        #[ignore = "Requires a running Binit PostgreSQL database with Aegis results."]
        fn binit_full() {
            run_db_corpus(None);
        }
    }
}
