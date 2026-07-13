use qcode::{
    builder::Builder,
    context::Context,
    space::{SpaceId, SpaceType},
    types::TypeId,
    value::{
        BasicBlock, FunctionBody, FunctionId, Value, ValueId, Varnode, VarnodeId, insn::Mnemonic,
    },
};

use rustc_hash::FxHashSet;

use crate::{Pass, PipelineEnv};

use super::{add_input, address_taken_set, append_outputs, called_function_set};

fn is_register(ctx: &Context, vn: VarnodeId) -> bool {
    matches!(Varnode::from_id(ctx, vn).space().ty, SpaceType::Register)
}

// ===========================================================================
// Register channel (runs early, before mem2reg — see ARGPROMOTE_REGISTERS.md)
// ===========================================================================
//
// Unlike the RAM channel above, register effects are functionalized by a purely
// syntactic scan of the lifted body: a register *loaded* is an input, a register
// *stored* is an output. No shadow space is needed — every register access
// converts, so the body keeps operating on real register space and the *next*
// mem2reg run SSA-promotes it into a pure value function. Inputs become by-value
// params seeded into register space at entry; outputs are returned as a flat
// positional aggregate (slot i ↔ output register i) the caller replays.

/// A function's register interface, recovered by [`scan_register_effects`].
#[derive(Debug)]
pub(crate) struct RegisterEffects {
    /// Registers the body loads — each becomes a by-value input parameter
    /// (over-approximated: a written-first register yields a dead param that
    /// mem2reg/DCE prune). Sorted by `(address, size)` for a deterministic
    /// param/argument order shared with the caller rewrite.
    pub(crate) inputs: Vec<VarnodeId>,
    /// Registers the body stores, canonicalized to the coarsest register per
    /// overlap group so the caller's replay is order-independent. Sorted.
    pub(crate) outputs: Vec<VarnodeId>,
}

/// The byte interval `(space, start, end)` a register varnode occupies.
fn reg_interval(ctx: &Context, vn: VarnodeId) -> (SpaceId, i64, i64) {
    let v = Varnode::from_id(ctx, vn);
    let start = v.address();
    (v.space().id, start, start + v.size() as i64)
}

/// `true` if two register varnodes occupy overlapping bytes of the same space.
fn regs_overlap(ctx: &Context, a: VarnodeId, b: VarnodeId) -> bool {
    let (sa, a0, a1) = reg_interval(ctx, a);
    let (sb, b0, b1) = reg_interval(ctx, b);
    sa == sb && a0 < b1 && b0 < a1
}

/// Collapse a set of register varnodes into the coarsest register per overlap
/// group. Register files nest (AL ⊂ AX ⊂ EAX ⊂ RAX), so each overlap group has a
/// unique member whose interval contains the rest; for *outputs*, loading that
/// register at the return reads the merged final state of every sub-write, and
/// for *inputs* one seed of it covers every overlapping read. Returns `None` if
/// some group has no single covering register (partial overlap with no cover) —
/// that function is left on the conservative path.
fn canonicalize_to_coarsest(ctx: &Context, regs: &[VarnodeId]) -> Option<Vec<VarnodeId>> {
    // Connected components under `regs_overlap` (tiny N, so O(N²) is fine).
    let mut group_of: Vec<usize> = (0..regs.len()).collect();
    for i in 0..regs.len() {
        for j in (i + 1)..regs.len() {
            if regs_overlap(ctx, regs[i], regs[j]) {
                let (gi, gj) = (group_of[i], group_of[j]);
                if gi != gj {
                    for g in &mut group_of {
                        if *g == gj {
                            *g = gi;
                        }
                    }
                }
            }
        }
    }

    let mut coarse: Vec<VarnodeId> = Vec::new();
    for g in 0..regs.len() {
        let members: Vec<VarnodeId> = (0..regs.len())
            .filter(|&i| group_of[i] == g)
            .map(|i| regs[i])
            .collect();
        if members.is_empty() {
            continue; // not a group representative
        }
        // The cover must contain every member's interval.
        let cover = members.iter().copied().find(|&m| {
            let (_, m0, m1) = reg_interval(ctx, m);
            members.iter().all(|&o| {
                let (_, o0, o1) = reg_interval(ctx, o);
                m0 <= o0 && m1 >= o1
            })
        })?;
        if !coarse.contains(&cover) {
            coarse.push(cover);
        }
    }
    Some(coarse)
}

/// Why a function is not register-pure (cannot be functionalized by
/// [`argpromote_registers`]). Mirrors the gating in [`try_promote_registers`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegPurityReason {
    /// External function — no body to functionalize.
    External,
    /// Lifted but has no root block (empty body).
    NoBody,
    /// Address-taken: reachable by an indirect call this pass can't rewrite.
    AddressTaken,
    /// Writes no registers, so there is nothing to functionalize.
    NoRegisterWrites,
    /// A register-write overlap group has no single covering register.
    NonCanonicalRegisters,
    /// No direct callers to thread by-value inputs / replayed outputs through.
    NoCallers,
}

impl RegPurityReason {
    /// One-line human-readable explanation, for display in the GUI / headless dump.
    pub fn describe(self) -> &'static str {
        match self {
            RegPurityReason::External => "external function (no body to functionalize)",
            RegPurityReason::NoBody => "no function body",
            RegPurityReason::AddressTaken => {
                "address-taken (reachable by indirect calls this pass can't rewrite)"
            }
            RegPurityReason::NoRegisterWrites => "writes no registers (nothing to functionalize)",
            RegPurityReason::NonCanonicalRegisters => {
                "register writes don't canonicalize to a coarsest register"
            }
            RegPurityReason::NoCallers => "no direct callers to thread inputs/outputs through",
        }
    }
}

/// Report whether `fid` is eligible to be functionalized into a register-pure
/// function (`Ok`) or, if not, the gating reason (`Err`). A function whose
/// [`FunctionBody::is_pure_reg`] is already set is necessarily `Ok`; this is the
/// source of the "why not" shown for the rest.
///
/// Builds the whole-program address-taken and called-function sets on every call.
/// A caller querying many functions in a loop (e.g. the GUI loader) should build
/// them once with [`RegPurityGates`] and use [`RegPurityGates::purity`] to avoid
/// an O(functions × instructions) rescan.
pub fn reg_purity(ctx: &Context, fid: FunctionId) -> Result<(), RegPurityReason> {
    RegPurityGates::compute(ctx).purity(ctx, fid)
}

/// The two whole-program gates a [`reg_purity`] query consults — the set of
/// address-taken functions and the set of direct call targets — each an
/// O(instructions) scan. Building them once amortizes both across a per-function
/// classification loop (the GUI loader), turning O(functions × instructions) into
/// O(instructions + functions). Both are invariant under register promotion (it
/// threads data values and rewrites interfaces but adds no `ValueId::Function`
/// operand and no `Call.target` edge), so a single instance is valid for the
/// whole loop.
pub struct RegPurityGates {
    address_taken: FxHashSet<FunctionId>,
    called: FxHashSet<FunctionId>,
}

impl RegPurityGates {
    /// Compute both gates for `ctx` in two O(instructions) passes.
    pub fn compute(ctx: &Context) -> Self {
        Self {
            address_taken: address_taken_set(ctx),
            called: called_function_set(ctx),
        }
    }

    /// Classify `fid` against the precomputed gates — see [`reg_purity`].
    pub fn purity(&self, ctx: &Context, fid: FunctionId) -> Result<(), RegPurityReason> {
        reg_purity_with(ctx, fid, &self.address_taken, &self.called)
    }
}

/// [`reg_purity`] with the whole-program gating sets supplied by the caller, so a
/// per-function loop builds them once instead of rescanning all instructions per
/// function. `address_taken` and `called` must be
/// [`address_taken_set`] / [`called_function_set`] over the same `ctx`.
fn reg_purity_with(
    ctx: &Context,
    fid: FunctionId,
    address_taken: &FxHashSet<FunctionId>,
    called: &FxHashSet<FunctionId>,
) -> Result<(), RegPurityReason> {
    let f = FunctionBody::from_id(ctx, fid);
    if f.is_external() {
        return Err(RegPurityReason::External);
    }
    if f.root().is_none() {
        return Err(RegPurityReason::NoBody);
    }
    if address_taken.contains(&fid) {
        return Err(RegPurityReason::AddressTaken);
    }
    scan_register_effects(ctx, fid)?;
    if !called.contains(&fid) {
        return Err(RegPurityReason::NoCallers);
    }
    Ok(())
}

/// Scan `fid` for register reads (inputs) and writes (outputs). Returns `Err`
/// when there is no register write (nothing to functionalize) or an output
/// overlap group has no single covering register.
pub(crate) fn scan_register_effects(
    ctx: &Context,
    fid: FunctionId,
) -> Result<RegisterEffects, RegPurityReason> {
    let mut loaded: Vec<VarnodeId> = Vec::new();
    let mut stored: Vec<VarnodeId> = Vec::new();
    for block in FunctionBody::from_id(ctx, fid).blocks() {
        for insn in block.iter() {
            match insn.mnemonic() {
                Mnemonic::Load(l) => {
                    if let qcode::value::LocalValueId::Varnode(vn) = l.ptr
                        && is_register(ctx, vn)
                        && !loaded.contains(&vn)
                    {
                        loaded.push(vn);
                    }
                }
                Mnemonic::Store(s) => {
                    if let qcode::value::LocalValueId::Varnode(vn) = s.ptr
                        && is_register(ctx, vn)
                        && !stored.contains(&vn)
                    {
                        stored.push(vn);
                    }
                }
                _ => {}
            }
        }
    }
    if stored.is_empty() {
        return Err(RegPurityReason::NoRegisterWrites);
    }
    let mut outputs =
        canonicalize_to_coarsest(ctx, &stored).ok_or(RegPurityReason::NonCanonicalRegisters)?;

    // The rewritten body reads not only the originally-loaded registers but also
    // every output (the return write-set loads each one). An output written on
    // only some paths is therefore read-before-write at a return on a no-write
    // path; seeding it from an input param makes that read the caller's incoming
    // value (replayed back as a no-op), keeping callee params and caller args in
    // sync. Always-written outputs just yield a dead seed that DCE prunes.
    let mut read_set = loaded;
    for &o in &outputs {
        if !read_set.contains(&o) {
            read_set.push(o);
        }
    }
    let mut inputs =
        canonicalize_to_coarsest(ctx, &read_set).ok_or(RegPurityReason::NonCanonicalRegisters)?;

    let key = |ctx: &Context, vn: &VarnodeId| {
        let v = Varnode::from_id(ctx, *vn);
        (v.address(), v.size())
    };
    inputs.sort_by_key(|vn| key(ctx, vn));
    outputs.sort_by_key(|vn| key(ctx, vn));
    Ok(RegisterEffects { inputs, outputs })
}

/// Functionalize every eligible function's register effects (see
/// [`try_promote_registers`]). Returns `true` if anything changed.
pub fn argpromote_registers(ctx: &mut Context) -> bool {
    let mut changed = false;
    // Gate every function on the two whole-program predicates via sets built once
    // instead of a per-function rescan: address-taken (stable — promotion adds no
    // `ValueId::Function` operands) and has-a-direct-caller (stable — promotion
    // rewrites interfaces but adds/removes no `Call.target` edges). See
    // [`super::address_taken_set`] / [`super::called_function_set`].
    let address_taken = super::address_taken_set(ctx);
    let called = super::called_function_set(ctx);
    for fid in ctx.function_ids() {
        if try_promote_registers(ctx, &address_taken, &called, fid) {
            changed = true;
        }
    }
    changed
}

/// Per-output-register metadata: `(register, size, space, name)`. The name is the
/// register's own (e.g. `eax`) or a positional `output{n}` fallback.
fn output_meta(ctx: &Context, regs: &[VarnodeId]) -> Vec<(VarnodeId, usize, SpaceId, String)> {
    regs.iter()
        .map(|&r| {
            let v = Varnode::from_id(ctx, r);
            let name = v
                .name()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("output{}", usize::from(r) + 1));
            (r, v.size(), v.space().id, name)
        })
        .collect()
}

fn try_promote_registers(
    ctx: &mut Context,
    address_taken: &FxHashSet<FunctionId>,
    called: &FxHashSet<FunctionId>,
    fid: FunctionId,
) -> bool {
    let f = FunctionBody::from_id(ctx, fid);
    if f.is_external() || f.root().is_none() {
        return false;
    }
    // Closed-world: an address-taken function may be reached by an indirect call
    // this pass cannot find and rewrite, leaving a caller on the old register ABI.
    if address_taken.contains(&fid) {
        return false;
    }
    let Ok(eff) = scan_register_effects(ctx, fid) else {
        return false;
    };

    // Only direct callers can be rewritten to pass inputs / replay outputs; with
    // none, rewriting the callee would leave it expecting params nobody provides.
    // `called` is the whole-program set of direct call targets (see
    // [`super::called_function_set`]), an O(1) lookup instead of a per-function rescan.
    if !called.contains(&fid) {
        return false;
    }

    rewrite_registers(ctx, fid, &eff);

    // The body now reads its registers only through by-value params and returns
    // every write through the aggregate write-set: it is a pure value function.
    // `dead_signature` keys on this flag to trim dead params / returned fields.
    FunctionBody::from_id_mut(ctx, fid).set_pure_reg(true);
    true
}

/// Precomputed `(register, size, space, name, type)` for one input register.
type InputMeta = (VarnodeId, usize, SpaceId, Option<String>, Option<TypeId>);

/// Rewrite `fid` for its register effects `eff`, in both directions: thread one
/// by-value input param per input register (seeded into the register file, loaded
/// fresh at every caller) and append the outputs as a flat positional write-set
/// replayed at every caller. Leaves `is_pure_reg` for [`try_promote_registers`] to
/// set — factored out so unit tests can drive the rewrite without the gating.
pub(crate) fn rewrite_registers(ctx: &mut Context, fid: FunctionId, eff: &RegisterEffects) {
    // --- inputs: one by-value param per input register --------------------------
    // The body reads its live-in registers through params the next mem2reg run
    // SSA-promotes. Precompute `(register, size, space, name, type)` before the
    // first mutable borrow.
    let input_meta: Vec<InputMeta> = eff
        .inputs
        .iter()
        .map(|&r| {
            let v = Varnode::from_id(&*ctx, r);
            (
                r,
                v.size(),
                v.space().id,
                v.name().map(str::to_owned),
                // Inherit a global varnode type override (e.g. `FS_OFFSET` typed
                // `PtrTo<TEB>` by `windows_teb_seed`) so the by-value entry param
                // carries the ambient register's richer type, not `Int(size)`.
                ctx.stored_type_of(ValueId::Varnode(r)),
            )
        })
        .collect();
    for (r, size, space, name, ty) in input_meta {
        add_input(
            ctx,
            fid,
            size,
            // Name the param after its register so the calling convention binds it
            // from the register file (the emulator's `seed_entry_params` keys on it).
            name,
            Some(ValueId::Varnode(r)),
            ty,
            space,
            move |_b| ValueId::Varnode(r),
            move |ctx, call_id, block| {
                let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, block));
                b.set_insert_point_before(call_id);
                b.push_load::<false>(ValueId::Varnode(r), size, space).id()
            },
        );
    }

    // --- outputs: a flat positional write-set, one field per output register ----
    // The register channel is the first to touch the return slot, so it *sets*
    // (`append = false`); a later RAM round appends its memory pairs after these.
    let outputs = output_meta(ctx, &eff.outputs);
    append_outputs(
        ctx,
        fid,
        &outputs,
        false,
        |_i, (_, _, _, name)| vec![name.clone()],
        |b, &(r, size, space, _)| vec![b.push_load::<false>(ValueId::Varnode(r), size, space).id()],
        |b, &(r, _, space, _), ext| {
            b.push_store(ext[0], ValueId::Varnode(r), space);
        },
    );
}

#[derive(Default)]
pub struct ArgPromoteRegisters;

impl Pass for ArgPromoteRegisters {
    const NAME: &'static str = "argpromote_registers";
    fn description(&self) -> &'static str {
        "Functionalize register side effects into a returned write-set (runs early)"
    }
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        Ok(argpromote_registers(ctx))
    }
}

crate::register_module_pass!(ArgPromoteRegisters);
