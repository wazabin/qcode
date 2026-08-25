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
    use sleigh::{CompiledSpec, Decoder, SymbolKind};
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
    const MEMORY_WORDS: &[(&str, u64)] = &[("mem0_value", MEM0_ADDR), ("mem1_value", MEM1_ADDR)];
    const MAX_EMULATED_STEPS: usize = 10_000;
    const SCALAR_REGISTERS: &[&str] = &[
        "RAX", "RBX", "RCX", "RDX", "RSI", "RDI", "R8", "R9", "RBP", "RSP",
    ];
    /// MMX aliases x87 physical registers, but the Binit protocol exposes its
    /// eight 64-bit MMX views. Aegis restores/captures that shared file with
    /// FXSAVE/FXRSTOR around each one-instruction test.
    const MMX_REGISTERS: &[&str] = &["MM0", "MM1", "MM2", "MM3", "MM4", "MM5", "MM6", "MM7"];
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

    /// Decodes and lifts one instruction using the embedded x86-64 SLEIGH
    /// specification. Keeping the compiled spec in `sleigh-precompile` makes
    /// corpus replay practical: source compilation is never on the hot path.
    fn lift_x64(bytes: &[u8], address: u64) -> Result<Context<'static>, String> {
        let spec: &CompiledSpec = x64::spec();
        let lifter = SleighLifter::new(spec);
        let decode_context = spec.new_context();
        let instruction = Decoder::new(spec)
            .decode_one(address, bytes, &decode_context)
            .map_err(|error| format!("decode failed: {error}"))?;
        let mut context = lifter.new_context();
        lifter
            .lift_instruction(&mut context, &instruction, None)
            .map_err(|error| format!("lift failed: {error}"))?;
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

    /// DB-native state: register/flag values keyed by name (lowercase).
    /// The "flag" key holds the raw RFLAGS word.
    #[derive(Clone, Debug)]
    struct DbState {
        regs: HashMap<String, i64>,
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
            let regs = json
                .as_object()
                .map(|obj| {
                    obj.iter()
                        .filter_map(|(k, v)| {
                            v.as_i64()
                                .or_else(|| v.as_u64().map(|n| n as i64))
                                .map(|n| (k.clone(), n))
                        })
                        .collect()
                })
                .unwrap_or_default();
            DbState { regs }
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

    fn run_to_instruction_boundary(emu: &mut Emulator<'_>) -> Result<(), RunError> {
        let entry_address = emu.block().address();
        for _ in 0..MAX_EMULATED_STEPS {
            if emu
                .block()
                .address()
                .is_some_and(|address| Some(address) != entry_address)
            {
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
    ) -> Result<DbState, String> {
        let mut regs = HashMap::new();
        for &name in SCALAR_REGISTERS.iter().chain(MMX_REGISTERS) {
            let id = x64_register(name).ok_or_else(|| format!("unknown register {name}"))?;
            let value = emu
                .read_register(id)
                .ok_or_else(|| format!("could not read register {name}"))?;
            regs.insert(name.to_ascii_lowercase(), value as i64);
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
        Ok(DbState { regs })
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
        if matches!(mnemonic, "bt" | "btc" | "btr" | "bts") && tc.instruction.contains("ptr") {
            let bit_index = tc
                .instruction
                .rsplit(',')
                .next()
                .map(str::trim)
                .and_then(|reg| pair.initial.regs.get(&reg.to_ascii_lowercase()).copied());
            if let Some(value) = bit_index {
                let byte_offset = value >> 3; // signed, matches SLEIGH `s>> 3`
                if !(0..8).contains(&byte_offset) {
                    return Some("bit index addresses memory outside modeled window");
                }
            }
        }
        None
    }

    /// SHLD/SHRD leave the destination operand undefined when the shift count
    /// exceeds the operand bit-width (Intel SDM). This is only reachable for
    /// 16-bit operands, whose count masks to at most 31. Returns the destination
    /// register (canonical 64-bit, lowercase) whose comparison should be skipped.
    fn shld_shrd_undefined_dest(tc: &DbTestCase, pair: &DbStateResult) -> Option<String> {
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
        let dest = canonical_reg16(operands[0])?;
        let raw_count = if operands[2].eq_ignore_ascii_case("cl") {
            (pair.initial.regs.get("rcx").copied().unwrap_or(0) as u64) & 0xff
        } else {
            parse_signed_imm(operands[2])?
        };
        if (raw_count & 0x1f) > 16 {
            Some(dest)
        } else {
            None
        }
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

            // Aegis executes from CpuState::zero(). Seed the same scalar baseline
            // because qcode's emulator stores register bytes lazily.
            for &name in SCALAR_REGISTERS.iter().chain(MMX_REGISTERS) {
                let id =
                    x64_register(name).expect("scalar/MMX register is present in the x64 spec");
                emu.set_register(id, 0).map_err(|error| {
                    DbMismatch::backend_error(tc, state_index, pair, final_state, error)
                })?;
            }

            // Set initial general-purpose registers.
            for (name, &raw) in &pair.initial.regs {
                if name == "flag"
                    || name == "rip"
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
            catch_unwind(AssertUnwindSafe(|| run_to_instruction_boundary(&mut emu)))
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
                if shld_shrd_undefined.as_deref() == Some(name.as_str()) {
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
                    let actual_state = snapshot_state(&mut emu, &ctx, include_mem0, include_mem1)
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

            for &(memory_name, address) in MEMORY_WORDS {
                let Some(&raw) = final_state.regs.get(memory_name) else {
                    continue;
                };
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
                    let actual_state = snapshot_state(&mut emu, &ctx, include_mem0, include_mem1)
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
                    let actual_state = snapshot_state(&mut emu, &ctx, include_mem0, include_mem1)
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
                    },
                    final_state: Some(DbState {
                        regs: HashMap::from([("rax".into(), 1)]),
                    }),
                }],
            };
            run_db_case(&tc).unwrap();
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
