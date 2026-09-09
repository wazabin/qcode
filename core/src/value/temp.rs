//! Function-local temporary spaces and values.
//!
//! These arenas are introduced before producers migrate away from shared
//! [`Varnode`](crate::value::Varnode) storage. Body storage uses the local IDs;
//! immutable module/API boundaries use the function-qualified IDs.

use std::{borrow::Cow, fmt::Display, marker::PhantomData};

use jstd::Identifier;

use crate::value::{ModuleView, QCodeView, Value, ValueId, util::named::Named};

/// Function-local temporary-space index.
#[derive(Identifier)]
pub struct LocalTempSpaceId(usize);

crate::composite_id!(TempSpaceId, LocalTempSpaceId);

/// Function-local temporary-value index.
#[derive(Identifier)]
pub struct LocalTempId(usize);

crate::composite_id!(TempId, LocalTempId);

/// A body-owned memory space used only for temporary values.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TempSpace {
    pub(crate) name: Option<Box<str>>,
    pub(crate) word_size: usize,
    pub(crate) addr_size: usize,
}

impl TempSpace {
    pub fn new(name: Option<&str>, word_size: usize, addr_size: usize) -> Self {
        Self {
            name: name.map(Box::from),
            word_size,
            addr_size,
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn word_size(&self) -> usize {
        self.word_size
    }

    pub fn addr_size(&self) -> usize {
        self.addr_size
    }
}

/// A body-owned temporary memory value.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Temp<'str> {
    pub(crate) name: Option<Cow<'str, str>>,
    pub(crate) label: Option<u32>,
    pub(crate) address: i64,
    pub(crate) size: usize,
    pub(crate) space: LocalTempSpaceId,
}

impl<'str> Temp<'str> {
    pub fn new(address: i64, size: usize, space: LocalTempSpaceId) -> Self {
        Self {
            name: None,
            label: None,
            address,
            size,
            space,
        }
    }

    /// Attaches the function-local name registered when this temporary is
    /// inserted into its owning body.
    pub fn with_name(mut self, name: Cow<'str, str>) -> Self {
        self.name = Some(name);
        self
    }
}

/// Immutable temporary-space reference over any [`QCodeView`] provider.
#[derive(Clone, Copy)]
pub struct TempSpaceRef<'str, 'ctx, R = ModuleView<'ctx, 'str>> {
    pub id: TempSpaceId,
    view: R,
    marker: PhantomData<&'ctx &'str ()>,
}

impl<'str, 'ctx, R> TempSpaceRef<'str, 'ctx, R> {
    pub fn new(view: R, id: TempSpaceId) -> Self {
        Self {
            id,
            view,
            marker: PhantomData,
        }
    }
}

impl<'str: 'ctx, 'ctx, R> TempSpaceRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn inner(self) -> &'ctx TempSpace {
        self.view.temp_space(self.id)
    }

    pub fn name(self) -> Option<&'ctx str> {
        self.inner().name.as_deref()
    }

    pub fn word_size(self) -> usize {
        self.inner().word_size
    }

    pub fn addr_size(self) -> usize {
        self.inner().addr_size
    }
}

impl<'str: 'ctx, 'ctx, R> Display for TempSpaceRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name().unwrap_or("temp"))
    }
}

/// Immutable temporary-value reference over any [`QCodeView`] provider.
#[derive(Clone, Copy)]
pub struct TempRef<'str, 'ctx, R = ModuleView<'ctx, 'str>> {
    pub id: TempId,
    pub(in crate::value) view: R,
    marker: PhantomData<&'ctx &'str ()>,
}

impl<'str, 'ctx, R> TempRef<'str, 'ctx, R> {
    pub fn new(view: R, id: TempId) -> Self {
        Self {
            id,
            view,
            marker: PhantomData,
        }
    }
}

impl<'str: 'ctx, 'ctx, R> TempRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn inner(self) -> &'ctx Temp<'str> {
        self.view.temp(self.id)
    }

    pub fn name(self) -> Option<&'ctx str> {
        self.inner().name.as_deref()
    }

    pub fn label(self) -> Option<u32> {
        self.inner().label
    }

    pub fn address(self) -> i64 {
        self.inner().address
    }

    pub fn size(self) -> usize {
        self.inner().size
    }

    pub fn space(self) -> TempSpaceRef<'str, 'ctx, R> {
        TempSpaceRef::new(
            self.view,
            TempSpaceId::new(self.id.func, self.inner().space),
        )
    }

    pub fn memory_space(self) -> crate::space::MemorySpaceId {
        crate::space::MemorySpaceId::Temp(self.space().id)
    }
}

impl<'str: 'ctx, 'ctx, R> Display for TempRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = self.name() {
            f.write_str(name)
        } else if let Some(label) = self.label() {
            write!(f, "v{label}")
        } else {
            write!(f, "[{}]:{} {}", self.space(), self.size(), self.address())
        }
    }
}

impl<'str: 'ctx, 'ctx, R> Named for TempRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn name(&self) -> Option<&str> {
        TempRef::name(*self)
    }
}

impl<'str: 'ctx, 'ctx, R> Value<'str, 'ctx> for TempRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn id(&self) -> ValueId {
        ValueId::Temp(self.id)
    }

    fn size(&self) -> usize {
        TempRef::size(*self)
    }
}
