//! Narrow C runtime startup recognizers.

use qcode::{
    context::Context,
    discovery::{Discovery, FunctionDiscoveryReason},
    value::{
        Function, LiteralRef, ValueId,
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

    fn run(&self, ctx: &mut Context, env: &PipelineEnv) -> Result<bool, String> {
        Ok(discover_libc_main(ctx, env))
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
    let Some((main, source_addr)) = find_libc_main_arg(ctx, entry, main_reg) else {
        return false;
    };
    // Idempotent across analyze rounds: once the discovery has materialized a
    // function at `main` (named or not), don't re-emit it every round.
    if Function::from_addr(ctx, main).is_some() {
        return false;
    }
    // A `main` already exists elsewhere (e.g. from the symbol table). Naming our
    // target `main` too would collide on the unique-name invariant, so bail.
    if Function::from_name(ctx, "main").is_some() {
        return false;
    }

    ctx.discover(
        Discovery::function(main)
            .with_function_reason(FunctionDiscoveryReason::CrtMain)
            .with_name(Some("main".to_string()))
            .from_addr(source_addr),
    )
}

fn find_libc_main_arg(ctx: &Context<'_>, entry: u64, main_reg: ValueId) -> Option<(u64, u64)> {
    let function = Function::from_addr(ctx, entry)?;
    let mut last_main = None;

    for block in function.blocks() {
        for insn in block.instructions() {
            match insn.mnemonic() {
                Mnemonic::Store(store) if store.ptr == main_reg => {
                    last_main = store.src.as_literal().map(|id| {
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
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{BasicBlock, Function},
    };

    use crate::{ArchConfig, CallingConvention, GpReg};

    #[test]
    fn discovers_named_main_from_primary_entrypoint_argument() {
        let mut tc = TestContext::new();
        tc.ctx.set_primary_entrypoint(Some(0x1000));
        Function::make_at_addr(&mut tc.ctx, 0x1000, Some("start".into()));

        {
            let block = tc.ctx.get_or_make_block(0x1000);
            Function::from_addr_mut(&mut tc.ctx, 0x1000)
                .unwrap()
                .set_root(block)
                .unwrap();
            let mut builder = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, block));
            let main = builder.context_mut().get_const(0x2000, 8).id();
            builder.push_store(main, ValueId::Varnode(tc.r0), tc.reg_space);
            let target = builder.context_mut().get_const(0x3000, 8).id();
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
        };
        let env = PipelineEnv {
            cfg,
            sp_varnode: tc.r3,
        };

        assert!(DiscoverLibcMain.run(&mut tc.ctx, &env).unwrap());
        let discoveries = tc.ctx.discoveries().collect::<Vec<_>>();
        assert_eq!(discoveries.len(), 1);
        assert_eq!(discoveries[0].target, 0x2000);
        assert_eq!(discoveries[0].name.as_deref(), Some("main"));
    }
}
