//! Rename ordinal-only PE imports to their real export names.
//!
//! Some Windows DLLs (classically `OLEAUT32.dll`) are imported by ordinal, so
//! the PE import table carries no symbol name and the loader names the stub
//! `ORDINAL <n>`. That placeholder has no C prototype, so
//! [`external_sigs`](super::external_sig) cannot give it a signature and the
//! GUI shows it unresolved.
//!
//! This pass consults a user-maintained [`cabi::OrdinalMap`] (`ordinal_map.toml`)
//! and, using the binary's per-import DLL, renames each `ORDINAL <n>` stub to
//! its real name. It must run **before** `external_sigs` so the renamed stubs
//! get prototype signatures like any other named import; everything downstream
//! (doc links, the GUI list) then follows from the real name.
//!
//! Requires the live binary handle (for the per-import DLL); a no-op on
//! headless/textual runs, on binaries with no ordinal imports, and when no
//! `ordinal_map.toml` maps the DLL/ordinal.

use cabi::OrdinalMap;
use qcode::{
    context::Context,
    value::{FunctionBody, FunctionId, Renameable},
};

use crate::{Pass, PipelineEnv};

#[derive(Default)]
pub struct ResolveOrdinals;

impl Pass for ResolveOrdinals {
    const NAME: &'static str = "resolve_ordinals";

    fn description(&self) -> &'static str {
        "Rename ordinal-only PE imports (e.g. OLEAUT32) to their real export names"
    }

    fn run(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
    ) -> Result<crate::ModulePassOutcome, String> {
        // The per-import DLL is only available from the live binary handle.
        let Some(binary) = env.binary.as_ref() else {
            return Ok(crate::ModulePassOutcome::default());
        };
        let map = OrdinalMap::load();
        if map.is_empty() {
            return Ok(crate::ModulePassOutcome::default());
        }

        let mut renamed = Vec::new();
        for &id in targets {
            let f = FunctionBody::from_id(ctx, id);
            if !f.is_external() {
                continue;
            }
            let Some(ordinal) = parse_ordinal(f.name()) else {
                continue;
            };
            let Some(addr) = f.address() else {
                continue;
            };
            let Some(dll) = binary.import_library(addr) else {
                continue;
            };
            let Some(real) = map.resolve(dll, ordinal).map(str::to_owned) else {
                continue;
            };
            // A duplicate name (another import already carries it) fails the
            // unique rename; skip that one rather than abort the pass.
            let mut f = FunctionBody::from_id_mut(ctx, id);
            if f.rename(std::borrow::Cow::Owned(real)).is_ok() {
                // Record the by-ordinal origin on the interface so it survives
                // the rename (and snapshots); the GUI reads it back.
                f.set_import_ordinal(Some(ordinal));
                renamed.push(id);
            }
        }

        // Renaming changes only display names, not the CFG or address index, so
        // both whole-program analyses survive.
        Ok(crate::ModulePassOutcome::functions(renamed)
            .preserving_global::<crate::CallGraphAnalysis>()
            .preserving_global::<crate::AddressAnalysis>())
    }
}

crate::register_module_pass!(ResolveOrdinals);

/// The ordinal in a synthesized `ORDINAL <n>` import name (goblin's placeholder
/// for a PE import brought in by ordinal only), or `None` for a normal name.
fn parse_ordinal(name: &str) -> Option<u16> {
    name.strip_prefix("ORDINAL ")?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::parse_ordinal;

    #[test]
    fn parses_ordinal_placeholder() {
        assert_eq!(parse_ordinal("ORDINAL 2"), Some(2));
        assert_eq!(parse_ordinal("ORDINAL 65535"), Some(65535));
        assert_eq!(parse_ordinal("SysAllocString"), None);
        assert_eq!(parse_ordinal("ORDINAL 70000"), None); // overflows u16
        assert_eq!(parse_ordinal("printf@plt"), None);
    }
}
