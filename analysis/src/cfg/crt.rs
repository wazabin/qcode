//! Narrow C runtime startup recognizers.

use qcode::{
    address_index::AddressIndex,
    context::Context,
    discovery::{Discovery, FunctionDiscoveryReason},
    value::{
        FunctionBody, LiteralRef, ValueId,
        insn::{Call, CallInd, Mnemonic},
    },
};

use crate::{Pass, PipelineEnv};

#[derive(Default)]
pub struct DiscoverLibcMain;

impl Pass for DiscoverLibcMain {
    const NAME: &'static str = "discover_libc_main";

    fn description(&self) -> &'static str {
        "Discover the glibc __libc_start_main main argument"
    }

    fn run(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        _targets: &[qcode::value::FunctionId],
    ) -> Result<crate::ModulePassOutcome, String> {
        Ok(
            crate::ModulePassOutcome::module_if(discover_libc_main(ctx, env))
                .preserving_global::<crate::AddressAnalysis>(),
        )
    }
}

crate::register_module_pass!(DiscoverLibcMain);

pub fn discover_libc_main(ctx: &mut Context, env: &PipelineEnv) -> bool {
    let Some(entry) = ctx.primary_entrypoint() else {
        return false;
    };

    let Some(main_reg) = env
        .cfg
        .abi
        .int_args
        .first()
        .and_then(|reg| reg.widths.iter().max_by_key(|(width, _)| *width))
        .map(|(_, varnode)| ValueId::Varnode(*varnode))
    else {
        return false;
    };

    let addresses = AddressIndex::analyze(ctx);
    let Some((main, source_addr)) = find_libc_main_arg(ctx, &addresses, entry, main_reg) else {
        return false;
    };

    let mut changed = false;

    // `main` is passed to `__libc_start_main` as a pointer argument, not called
    // directly, so it never appears as a derived callee of `entry`. Record the
    // `entry → main` call-graph edge synthetically so the call graph links them.
    // Keyed by address, this is independent of whether we (below) or the symbol
    // table materialize `main`, and it is idempotent across analyze rounds.
    if let Some(entry_id) = addresses.function_at(entry) {
        changed |= ctx.shared.values.add_synthetic_callee(entry_id, main);
    }

    // Materialize and name `main` only if nothing lives there yet: a function may
    // already exist at `main` (a prior round's discovery) or the name `main` may
    // be taken (e.g. from the symbol table), in which case naming ours `main`
    // would collide on the unique-name invariant.
    if addresses.function_at(main).is_none() && FunctionBody::from_name(ctx, "main").is_none() {
        changed |= ctx.discover(
            Discovery::function(main)
                .with_function_reason(FunctionDiscoveryReason::CrtMain)
                .with_name(Some("main".to_string()))
                .from_addr(source_addr),
        );
    }

    changed
}

fn find_libc_main_arg(
    ctx: &Context<'_>,
    addresses: &AddressIndex,
    entry: u64,
    main_reg: ValueId,
) -> Option<(u64, u64)> {
    let function = FunctionBody::from_id(ctx, addresses.function_at(entry)?);
    let mut last_main = None;

    for block in function.blocks() {
        for insn in block.instructions() {
            match insn.mnemonic() {
                Mnemonic::Store(store) if store.ptr.qualify(insn.id.func) == main_reg => {
                    last_main = store.src.qualify(insn.id.func).as_literal().map(|id| {
                        (
                            LiteralRef::from_id(ctx, id).value(),
                            insn.address().unwrap_or(entry),
                        )
                    });
                }
                Mnemonic::Call(Call { .. }) | Mnemonic::CallInd(CallInd { .. }) => {
                    return last_main;
                }
                _ => {}
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{testing::TestContext, value::FunctionBody};

    use crate::{ArchConfig, CallingConvention, GpReg};

    #[test]
    fn discovers_named_main_from_primary_entrypoint_argument() {
        let mut tc = TestContext::new();
        tc.ctx.set_primary_entrypoint(Some(0x1000));
        let start = FunctionBody::make_at_addr(&mut tc.ctx, 0x1000, Some("start".into())).id;

        {
            let block = { tc.ctx.get_or_make_block(0x1000, start) };
            FunctionBody::from_id_mut(&mut tc.ctx, start)
                .set_root(block)
                .unwrap();
            let mut builder = tc.ctx.builder(block);
            let main = builder.shr().get_const(0x2000, 8);
            builder.push_store(main, ValueId::Varnode(tc.r0), tc.reg_space);
            let target = builder.shr().get_const(0x3000, 8);
            builder.push_call_ind(target);
        }

        let cfg = ArchConfig {
            stack_pointer: qcode::value::RegisterId::from(0usize),
            dead_flag_regs: Vec::new(),
            abi: CallingConvention {
                int_args: vec![GpReg {
                    widths: vec![(8, tc.r0)],
                }],
                ..CallingConvention::default()
            },
            os: qcode::context::TargetOs::Unknown,
            bitness: 64,
        };
        let env = PipelineEnv::from_parts(cfg, tc.r3);

        assert!(
            DiscoverLibcMain
                .run(&mut tc.ctx, &env, &[])
                .unwrap()
                .changed()
        );
        let discoveries = tc.ctx.discoveries().collect::<Vec<_>>();
        assert_eq!(discoveries.len(), 1);
        assert_eq!(discoveries[0].target, 0x2000);
        assert_eq!(discoveries[0].name.as_deref(), Some("main"));
    }

    #[test]
    fn records_entry_to_main_call_graph_edge() {
        let mut tc = TestContext::new();
        tc.ctx.set_primary_entrypoint(Some(0x1000));
        let start = FunctionBody::make_at_addr(&mut tc.ctx, 0x1000, Some("start".into())).id;

        {
            let block = { tc.ctx.get_or_make_block(0x1000, start) };
            FunctionBody::from_id_mut(&mut tc.ctx, start)
                .set_root(block)
                .unwrap();
            let mut builder = tc.ctx.builder(block);
            let main = builder.shr().get_const(0x2000, 8);
            builder.push_store(main, ValueId::Varnode(tc.r0), tc.reg_space);
            let target = builder.shr().get_const(0x3000, 8);
            builder.push_call_ind(target);
        }

        let cfg = ArchConfig {
            stack_pointer: qcode::value::RegisterId::from(0usize),
            dead_flag_regs: Vec::new(),
            abi: CallingConvention {
                int_args: vec![GpReg {
                    widths: vec![(8, tc.r0)],
                }],
                ..CallingConvention::default()
            },
            os: qcode::context::TargetOs::Unknown,
            bitness: 64,
        };
        let env = PipelineEnv::from_parts(cfg, tc.r3);

        assert!(
            DiscoverLibcMain
                .run(&mut tc.ctx, &env, &[])
                .unwrap()
                .changed()
        );

        let entry_id = AddressIndex::analyze(&tc.ctx).function_at(0x1000).unwrap();
        // The synthetic edge is keyed by `main`'s address.
        assert!(
            tc.ctx
                .shared
                .values
                .synthetic_callees_of(entry_id)
                .any(|addr| addr == 0x2000)
        );

        // It does not surface as a callee until a function exists at `main`...
        assert!(
            crate::CallGraph::analyze(&tc.ctx)
                .callees(entry_id)
                .is_empty()
        );

        // ...and once one does, `entry → main` shows up in the call graph.
        let main_id = FunctionBody::make_at_addr(&mut tc.ctx, 0x2000, Some("main".into())).id;
        assert_eq!(
            crate::CallGraph::analyze(&tc.ctx).callees(entry_id),
            vec![main_id]
        );

        // Re-running is idempotent: the edge already exists, so nothing changes.
        assert!(
            !DiscoverLibcMain
                .run(&mut tc.ctx, &env, &[])
                .unwrap()
                .changed()
        );
    }
}
