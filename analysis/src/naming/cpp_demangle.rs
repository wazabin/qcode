//! Demangle C++ function names — the **easy example pass**.
//!
//! This is the smallest possible [`FunctionPass`]: it reads the function's name,
//! tries to demangle it as an Itanium C++ symbol, and renames the function if that
//! succeeds. It's the one to read (and copy) when writing a new pass.
//!
//! It also shows why [`FunctionPass`] requires [`Default`]: the pass precomputes
//! its [`DemangleOptions`] once in [`Default::default`] and reuses them on every
//! call, rather than rebuilding them per function. Registration is a single
//! [`inventory::submit!`] at the bottom of the file — nothing central to edit.

use std::borrow::Cow;

use cpp_demangle::DemangleOptions;

use crate::{ContextView, FunctionBody, FunctionPass};

pub struct CppDemangle {
    /// Precomputed in `Default`: rendered names omit parameter lists
    /// (`space::foo` rather than `space::foo(int, bool, char)`).
    options: DemangleOptions,
}

impl Default for CppDemangle {
    fn default() -> Self {
        Self {
            options: DemangleOptions::new().no_params(),
        }
    }
}

impl FunctionPass for CppDemangle {
    const NAME: &'static str = "cpp_demangle";

    fn description(&self) -> &'static str {
        "Demangle C++ function names"
    }

    // Returns true if a change was made, so it can be used in a `repeat_until` stage if needed.
    // This pass doesn't need to be, but it's good practice to track changes in case you later add more functionality
    fn run<'str>(
        &self,
        f: &mut FunctionBody<'_, 'str>,
        m: ContextView<'_, 'str>,
    ) -> Result<bool, String> {
        // Read the function's own name, decide the demangled form, then buffer
        // the rename.
        let demangled: Option<String> = {
            let name = f.self_ref(m).name().to_string();
            // ELF symbol-version suffixes (`foo@@GLIBCXX_3.4`, `foo@CXXABI_1.3`)
            // aren't part of the Itanium mangling, and `Symbol::new` rejects them
            // as not well-formed. Strip from the first `@` — a mangled name never
            // contains one — so versioned imports still demangle.
            let mangled = name.split('@').next().unwrap_or(&name);
            cpp_demangle::Symbol::new(mangled)
                .ok()
                // A symbol that parses but won't render: leave it untouched.
                .and_then(|sym| sym.demangle_with_options(&self.options).ok())
        };

        match demangled {
            Some(demangled) => {
                // Buffered; the driver applies it (uniquified) at the barrier.
                f.effects_mut().rename_self(Cow::Owned(demangled));
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

crate::register_function_pass!(CppDemangle);

#[cfg(test)]
mod tests {
    use qcode::context::Context;
    use qcode::value::{Function, Renameable};
    use qcode_macro::qcode;

    use super::*;
    use crate::test_util::run_function_pass;

    #[test]
    fn unmangled_name_is_not_changed() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                fn foo:
                    <entry>
                        return at 0x1000;
            "
        );

        run_function_pass::<CppDemangle>(&mut ctx, foo).unwrap();

        assert_eq!(Function::from_id(&ctx, foo).name(), "foo");
    }

    #[test]
    fn mangled_name_gets_demangled() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                fn _ZN5space3fooEibc:
                    <entry>
                        return at 0x1000;
            "
        );

        run_function_pass::<CppDemangle>(&mut ctx, _ZN5space3fooEibc).unwrap();

        assert_eq!(
            Function::from_id(&ctx, _ZN5space3fooEibc).name(),
            "space::foo"
        );
    }

    #[test]
    fn versioned_symbol_gets_demangled() {
        // ELF imports often carry a `@@VERSION` suffix that isn't part of the
        // Itanium mangling; it must be stripped before demangling. `@@` can't be
        // a qcode identifier, so set the versioned name through the API.
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                fn placeholder:
                    <entry>
                        return at 0x1000;
            "
        );
        Function::from_id_mut(&mut ctx, placeholder)
            .rename("_ZNSsixEj@@GLIBCXX_3.4".into())
            .unwrap();

        run_function_pass::<CppDemangle>(&mut ctx, placeholder).unwrap();

        assert_eq!(
            Function::from_id(&ctx, placeholder).name(),
            "std::string::operator[]"
        );
    }
}
