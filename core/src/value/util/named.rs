use std::borrow::Cow;

use crate::{
    context::Context,
    error::{Error, ErrorTy, Result},
    value::ValueId,
};

pub trait Named {
    fn name(&self) -> Option<&str>;
}

/// Attempts to set the name in the context reverse map
pub fn update_context_name<'str>(
    id: ValueId,
    ctx: &mut Context<'str>,
    name: Cow<'str, str>,
    old_name: Option<&str>,
) -> Result<'str, ()> {
    if let Some(existing_id) = ctx.get_named(&name) {
        if existing_id != id {
            Err(Error::spanless(ErrorTy::DuplicateName(name.to_string())))
        } else {
            Ok(())
        }
    } else {
        ctx.update_name(name, id, old_name)
    }
}

pub trait Renameable<'str, 'ctx>: Named {
    fn rename(&mut self, name: Cow<'str, str>) -> Result<'str, ()>;

    fn with_name(mut self, name: Cow<'str, str>) -> Result<'str, Self>
    where
        Self: Sized,
    {
        self.rename(name)?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::Context,
        error::{Error, ErrorTy},
        value::{Varnode, util::base_ref::WithCtx},
    };

    #[test]
    fn test_update_name() {
        let mut ctx = Context::new();
        let space_id = ctx.default_space;
        let mut var = Varnode::make(&mut ctx, 0, 1, space_id);

        assert_eq!(var.ctx().get_named("var"), None);

        var.rename("var".into()).unwrap();

        assert_eq!(var.ctx().get_named("var"), Some(var.id()));
        assert_eq!(var.name(), Some("var"));
    }

    #[test]
    fn test_update_to_same_name() {
        let mut ctx = Context::new();
        let space_id = ctx.default_space;
        let mut var = Varnode::make(&mut ctx, 0, 1, space_id);

        assert_eq!(var.ctx().get_named("var"), None);

        var.rename("var".into()).unwrap();

        assert_eq!(var.ctx().get_named("var"), Some(var.id()));
        assert_eq!(var.name(), Some("var"));

        var.rename("var".into()).unwrap();

        assert_eq!(var.ctx().get_named("var"), Some(var.id()));
        assert_eq!(var.name(), Some("var"));
    }

    #[test]
    fn test_update_name_conflict() {
        let mut ctx = Context::new();
        let space_id = ctx.default_space;
        let mut var1 = Varnode::make(&mut ctx, 0, 1, space_id);
        var1.rename("var".into()).unwrap();

        let mut var2 = Varnode::make(&mut ctx, 1, 1, space_id);
        assert_eq!(
            var2.rename("var".into()),
            Err(Error::spanless(ErrorTy::DuplicateName("var".into())))
        );
    }
}
