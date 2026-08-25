-- Small, deterministic Binit/Aegis corpus for CI.
--
-- The instruction encodings and states use Binit's x86-64 execution layout.
-- The rows cover register transfers and RIP-relative memory access without
-- requiring KVM or Aegis during ordinary CI runs.

BEGIN;

TRUNCATE instructions RESTART IDENTITY CASCADE;

INSERT INTO instructions (id, name) VALUES
    (1, 'MOV');

INSERT INTO test_cases (id, instruction, opcode, instruction_id, initial_states) VALUES
    (
        1,
        'mov rax, rbx',
        '4889d8',
        1,
        '[{"rax":0,"rbx":81985529216486895,"flag":0}]'::jsonb
    ),
    (
        2,
        'mov rax, qword ptr [0x666666010100]',
        '488b0500f19aff',
        1,
        '[{"mem0_value":1311768467294899695,"flag":0}]'::jsonb
    ),
    (
        3,
        'mov qword ptr [0x666666010100], rbx',
        '48891d00f19aff',
        1,
        '[{"rbx":81985529216486895,"mem0_value":0,"flag":0}]'::jsonb
    );

INSERT INTO test_results (test_case_id, state_index, exception_kind, final_state) VALUES
    (1, 0, NULL, '{"rax":81985529216486895,"flag":0}'::jsonb),
    (2, 0, NULL, '{"rax":1311768467294899695,"flag":0}'::jsonb),
    (3, 0, NULL, '{"mem0_value":81985529216486895,"flag":0}'::jsonb);

SELECT setval(pg_get_serial_sequence('instructions', 'id'), 1, true);
SELECT setval(pg_get_serial_sequence('test_cases', 'id'), 3, true);
SELECT setval(pg_get_serial_sequence('test_results', 'id'), 3, true);

COMMIT;
