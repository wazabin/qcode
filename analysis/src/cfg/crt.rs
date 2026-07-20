//! Narrow C runtime startup recognizers.

use qcode::{
    address_index::AddressIndex,
    context::Context,
    discovery::{Discovery, FunctionDiscoveryReason},
    space::{Space, SpaceType},
    value::{
        FunctionBody, LiteralRef, ValueId,
        insn::{Call, CallInd, Mnemonic},
    },
};
use rustc_hash::FxHashMap;

use crate::{Pass, PipelineEnv, gvn::affine::precompute_forms};

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

    let main_reg = env
        .cfg
        .abi
        .int_args
        .first()
        .and_then(|reg| reg.widths.iter().max_by_key(|(width, _)| *width))
        .map(|(_, varnode)| ValueId::Varnode(*varnode));
    let pointer_width = usize::from(env.cfg.bitness) / 8;
    if main_reg.is_none() && pointer_width != 4 {
        return false;
    }

    let addresses = AddressIndex::analyze(ctx);
    let Some((main, source_addr)) =
        find_libc_main_arg(ctx, &addresses, entry, main_reg, pointer_width)
    else {
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
    main_reg: Option<ValueId>,
    pointer_width: usize,
) -> Option<(u64, u64)> {
    let function = FunctionBody::from_id(ctx, addresses.function_at(entry)?);
    let numbering = precompute_forms(qcode::value::ModuleView::new(ctx), function.id);
    let mut last_main = None;
    let mut stack_stores = Vec::new();
    let mut known_literals = FxHashMap::default();

    for block in function.blocks() {
        for insn in block.instructions() {
            match insn.mnemonic() {
                Mnemonic::Store(store)
                    if main_reg
                        .is_some_and(|main_reg| store.ptr.qualify(insn.id.func) == main_reg) =>
                {
                    let src = store.src.qualify(insn.id.func);
                    last_main = resolve_literal(ctx, &numbering, &known_literals, src)
                        .map(|value| (value, insn.address().unwrap_or(entry)));
                }
                Mnemonic::Store(store)
                    if main_reg.is_none() && store.space == ctx.shared.default_space =>
                {
                    let src = store.src.qualify(insn.id.func);
                    stack_stores.push((
                        store.ptr.qualify(insn.id.func),
                        resolve_literal(ctx, &numbering, &known_literals, src),
                        insn.address().unwrap_or(entry),
                    ));
                }
                Mnemonic::Call(call) if is_libc_start_main(ctx, call) => {
                    if main_reg.is_none() {
                        let result =
                            find_stack_main_arg(ctx, &numbering, &stack_stores, pointer_width);
                        log::debug!(
                            "i386 CRT main at call {:#x}: examined {} stack stores, result {result:?}",
                            insn.address().unwrap_or(entry),
                            stack_stores.len(),
                        );
                        return result;
                    }
                    return last_main;
                }
                // Preserve the historical fallback for an indirect startup
                // call, but skip unrelated direct calls such as i386's
                // get-PC thunk before `__libc_start_main`.
                Mnemonic::CallInd(CallInd { .. }) => {
                    if main_reg.is_none() {
                        return find_stack_main_arg(ctx, &numbering, &stack_stores, pointer_width);
                    }
                    return last_main;
                }
                _ => {}
            }

            // Clean lifted IR writes architectural registers through the
            // register space. Track literal-valued writes so a subsequent
            // `push eax` can be treated like a literal stack store.
            if let Mnemonic::Store(store) = insn.mnemonic()
                && store.space.shared().is_some_and(|space| {
                    matches!(Space::from_id(ctx, space).ty, SpaceType::Register)
                })
            {
                let ptr = store.ptr.qualify(insn.id.func);
                let src = store.src.qualify(insn.id.func);
                if let Some(value) = resolve_literal(ctx, &numbering, &known_literals, src) {
                    known_literals.insert(ptr, value);
                } else {
                    known_literals.remove(&ptr);
                }
            }
            if let Mnemonic::Load(load) = insn.mnemonic()
                && load.space.shared().is_some_and(|space| {
                    matches!(Space::from_id(ctx, space).ty, SpaceType::Register)
                })
            {
                let result = ValueId::Instruction(insn.id);
                let ptr = load.ptr.qualify(insn.id.func);
                if let Some(value) = known_literals.get(&ptr).copied() {
                    known_literals.insert(result, value);
                } else {
                    known_literals.remove(&result);
                }
            }
        }
    }

    None
}

fn is_libc_start_main(ctx: &Context<'_>, call: &Call) -> bool {
    call.target.real().is_some_and(|target| {
        FunctionBody::from_id(ctx, target).name().split('@').next() == Some("__libc_start_main")
    })
}

/// Recover i386's first cdecl argument. Immediately before a lifted call the
/// final pointer-sized RAM store is the synthetic return address at `[SP]`; the
/// first argument is the literal stored at `[SP + 4]`. Compare affine pointer
/// forms instead of relying on temporary identities so realigned startup stacks
/// such as glibc's `_start` are handled naturally.
fn find_stack_main_arg(
    ctx: &Context<'_>,
    numbering: &crate::gvn::affine::Numbering,
    stores: &[(ValueId, Option<u64>, u64)],
    pointer_width: usize,
) -> Option<(u64, u64)> {
    let (return_ptr, _, _) = stores.last()?;
    let view = qcode::value::ModuleView::new(ctx);
    let (return_base, return_offset) = numbering
        .base_offset(view, *return_ptr)
        .unwrap_or((*return_ptr, 0));

    stores[..stores.len() - 1]
        .iter()
        .rev()
        .find_map(|(ptr, value, source_addr)| {
            let (base, offset) = numbering.base_offset(view, *ptr).unwrap_or((*ptr, 0));
            // Clean pre-SSA x86 IR represents every push as a RAM store through
            // the same mutable ESP varnode. After register rewriting/GVN those
            // stores become distinct affine pointers one word apart. Accept
            // both representations; iteration from the end selects the first
            // argument store immediately before the return-address store.
            let same_mutable_sp = ptr == return_ptr;
            let adjacent_affine =
                base == return_base && offset == return_offset + pointer_width as i64;
            if !same_mutable_sp && !adjacent_affine {
                return None;
            }
            Some(((*value)?, *source_addr))
        })
}

fn resolve_literal(
    ctx: &Context<'_>,
    numbering: &crate::gvn::affine::Numbering,
    known_literals: &FxHashMap<ValueId, u64>,
    value: ValueId,
) -> Option<u64> {
    value
        .as_literal()
        .map(|literal| LiteralRef::from_id(ctx, literal).value())
        .or_else(|| numbering.constant_value(value))
        .or_else(|| known_literals.get(&value).copied())
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
            assume_calling_convention: false,
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
            assume_calling_convention: false,
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

    #[test]
    fn discovers_i386_main_from_first_stack_argument() {
        let mut tc = TestContext::new();
        tc.ctx.set_primary_entrypoint(Some(0x1000));
        let start = FunctionBody::make_at_addr(&mut tc.ctx, 0x1000, Some("start".into())).id;
        let libc_start_main = FunctionBody::make(&mut tc.ctx, "__libc_start_main".into())
            .unwrap()
            .id;

        {
            let block = { tc.ctx.get_or_make_block(0x1000, start) };
            FunctionBody::from_id_mut(&mut tc.ctx, start)
                .set_root(block)
                .unwrap();
            let ram = tc.ctx.shared.default_space;
            let reg_space = tc.reg_space;
            let main_reg = ValueId::Varnode(tc.r2);
            let mut builder = tc.ctx.builder(block);
            let stack_base = ValueId::Varnode(tc.r1);
            let four = builder.shr().get_const(4, 4);
            let eight = builder.shr().get_const(8, 4);
            let main_slot = builder.push_sub(stack_base, four).id();
            let return_slot = builder.push_sub(stack_base, eight).id();
            let main = builder.shr().get_const(0x2000, 4);
            builder.push_store(main, main_reg, reg_space);
            builder.push_store(main_reg, main_slot, ram);
            let return_address = builder.shr().get_const(0x1010, 4);
            builder.push_store(return_address, return_slot, ram);
            builder.push_call(libc_start_main);
        }

        let cfg = ArchConfig {
            stack_pointer: qcode::value::RegisterId::from(0usize),
            dead_flag_regs: Vec::new(),
            abi: CallingConvention::default(),
            os: qcode::context::TargetOs::Unknown,
            bitness: 32,
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
}
