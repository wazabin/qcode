//! Measures what checked binding costs on the VM's fetch path.
//!
//! Every on-demand lift binds the VM's context and address index as a
//! `LiftTarget`, which checks the index's provenance — a comparison of the
//! context's revision, O(functions) — before lowering. This prints that cost
//! in isolation, for modules of growing function count, next to the whole
//! per-instruction translation cost the VM reports, so the share is visible.
//!
//! ```sh
//! cargo run --release -p wazabin-qcode-sleigh --example bind-overhead
//! ```

use std::time::{Duration, Instant};

use qcode::{
    address_index::AddressIndex,
    lift::LiftTarget,
    value::{BasicBlock, FunctionBody},
};
use qcode_vm::{Vm, VmMemory, perm};
use wazabin_qcode_sleigh::{SleighLifter, vm_source::SleighCodeSource};

fn per_call(total: Duration, calls: u32) -> String {
    format!("{:>8.1} ns", total.as_nanos() as f64 / f64::from(calls))
}

fn main() {
    let spec = sleigh_precompile::x64::spec();
    let lifter = SleighLifter::new(spec).with_flat_control_flow();

    println!("checked binding alone, by module size (index current):");
    println!("{:>10} {:>8} {:>12} {:>14} {:>16}", "functions", "blocks", "bind_indexed", "bind(current)", "bind(refresh)");
    for &functions in &[1usize, 10, 100, 1_000, 10_000] {
        let mut ctx = lifter.new_context();
        let mut addresses = AddressIndex::analyze(&ctx);
        let mut host = None;
        for i in 0..functions {
            let address = 0x10_0000 + (i as u64) * 0x100;
            let function =
                FunctionBody::make_at_addr_indexed(&mut ctx, &mut addresses, address, None).id;
            // Ten addressed blocks per function, so refresh has something to scan.
            for b in 1..=10u64 {
                BasicBlock::make(&mut ctx, function)
                    .with_address_indexed(&mut addresses, address + b * 8);
            }
            host.get_or_insert(function);
        }
        let host = host.unwrap();
        let blocks = ctx.block_ids().len();
        assert!(addresses.is_current(&ctx));

        const CALLS: u32 = 20_000;
        let started = Instant::now();
        for _ in 0..CALLS {
            let target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, host).unwrap();
            std::hint::black_box(target.function());
        }
        let indexed = started.elapsed();

        let started = Instant::now();
        for _ in 0..CALLS {
            let target = LiftTarget::bind(&mut ctx, &mut addresses, host).unwrap();
            std::hint::black_box(target.function());
        }
        let current = started.elapsed();

        // A binding that has to rebuild: the index is left behind before each.
        let refresh_calls = (CALLS / 100).max(10);
        let started = Instant::now();
        for _ in 0..refresh_calls {
            addresses.clear();
            let target = LiftTarget::bind(&mut ctx, &mut addresses, host).unwrap();
            std::hint::black_box(target.function());
        }
        let refresh = started.elapsed();

        println!(
            "{functions:>10} {blocks:>8} {:>12} {:>14} {:>16}",
            per_call(indexed, CALLS),
            per_call(current, CALLS),
            per_call(refresh, refresh_calls)
        );
    }

    // The VM fetch path: a long straight-line sequence, each instruction
    // discovered once, so `decode_lift` is dominated by first-time lifts.
    println!();
    println!("VM fetch path, straight-line guest code lifted once per instruction:");
    for &instructions in &[1_000u32, 10_000] {
        let mut code = Vec::new();
        for i in 0..instructions {
            // mov eax, imm32 — five bytes, a distinct immediate each time.
            code.push(0xb8);
            code.extend_from_slice(&i.to_le_bytes());
        }
        code.push(0xf4); // hlt: loops in place, ending discovery.
        let source = SleighCodeSource::new(spec);
        let ctx = source.new_context();
        let mut memory = VmMemory::new();
        memory
            .mmu
            .write_unchecked(0x1000, &code, perm::READ | perm::EXEC);
        memory.mmu.map(0x20000, 0x2000, perm::RW_INIT).unwrap();
        let mut vm = Vm::at_address(ctx, 0x1000, source, memory).expect("the entry decodes");
        let started = Instant::now();
        vm.run(u64::from(instructions) * 8);
        let elapsed = started.elapsed();
        let stats = &vm.stats;
        println!(
            "{instructions:>6} instructions: {} lifts, decode+lift {:>8.2} µs/lift, fetch {:>6.2} µs/lift, wall {:.1} ms",
            stats.lifts,
            stats.decode_lift.as_nanos() as f64 / stats.lifts as f64 / 1e3,
            stats.fetch.as_nanos() as f64 / stats.lifts as f64 / 1e3,
            elapsed.as_secs_f64() * 1e3
        );
    }
}
