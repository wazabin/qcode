//! Function-local debug names, kept as what they derive from.
//!
//! Nearly every name a body holds is one of two derived kinds: a load named
//! after its register — `rax`, then `rax_1`, `rax_2`, … as the lift meets
//! the register again — and a block labelled with its address in hex. A
//! body holds millions of them, so a [`Name`] is eight bytes: a
//! [`BaseId`] into the body's interned bases plus a suffix, rendered as
//! `base` or `base_<n>`, and a block at an address with no name of its own
//! renders as the address and holds no name at all. Minting the next name
//! of a base is a slot in a vector; registering, resolving and growing the
//! table never hash a suffixed string.
//!
//! The uniqueness rule is the one a flat table of strings would give: a
//! taken bare name gets the first free `base_<n>` with `n ≥ 1`, in
//! creation order, and a name explicitly given as `base_<n>` takes that
//! slot. A block's address label counts as taken against an explicit name
//! spelled the same way, and the other way round.

use std::borrow::Cow;
use std::num::NonZeroU32;

use jstd::Identifier;
use rustc_hash::FxHashMap as HashMap;

use crate::error::{Error, ErrorTy, Result};

/// A base a body interns: `rax`, `tmp`, `loop`.
#[derive(Identifier)]
pub struct BaseId(u32);

/// A function-local name: a [base](BaseId) and a suffix, rendered `base`
/// for suffix `0` and `base_<suffix>` otherwise. Meaningful only with the
/// body that interned the base.
#[derive(Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Name {
    /// The base's index plus one, so an `Option<Name>` is still eight bytes.
    base: NonZeroU32,
    suffix: u32,
}

impl Name {
    pub(crate) fn new(base: BaseId, suffix: u32) -> Self {
        let raw: usize = base.into();
        let base = NonZeroU32::new(raw as u32 + 1).expect("a base index fits");
        Self { base, suffix }
    }

    pub fn base(self) -> BaseId {
        BaseId::from(self.base.get() as usize - 1)
    }

    pub fn suffix(self) -> u32 {
        self.suffix
    }
}

impl std::fmt::Debug for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Name({:?}, {})", self.base(), self.suffix)
    }
}

/// The names of one base, by suffix.
#[derive(Clone, Debug)]
struct Suffixes<Id> {
    /// The address this base spells, when it is one: a block labelled with
    /// it excludes the bare name, and minting checks that per name.
    hex: Option<u64>,
    /// The value holding `base_<n>` at index `n`; index `0` is the bare
    /// base.
    taken: Vec<Option<Id>>,
    /// A lower bound on the first free suffix `≥ 1`, so minting resumes
    /// instead of rescanning; lowered when a suffix is freed.
    hint: u32,
    /// How many names of this base are in the far map instead of `taken`,
    /// because their suffix was far past its end when they were registered.
    far: u32,
}

impl<Id> Suffixes<Id> {
    fn new(base: &str) -> Self {
        Self {
            hex: hex_value(base),
            taken: Vec::new(),
            hint: 1,
            far: 0,
        }
    }
}

/// How far past its end a suffix may extend a base's vector rather than go
/// to the far map.
const NEAR: usize = 4096;

/// A body's name → value table. See the [module documentation](self).
#[derive(Clone, Debug)]
pub(crate) struct LocalNames<Id> {
    bases: Vec<Box<str>>,
    index: HashMap<Box<str>, BaseId>,
    /// Parallel to `bases`.
    suffixes: Vec<Suffixes<Id>>,
    /// Explicit names whose suffix is far past their base's vector.
    far: HashMap<(BaseId, u32), Id>,
    /// The bases that spell an address, by the address: what a block's
    /// label has to be free of, without building the spelling.
    hex_bases: HashMap<u64, BaseId>,
    /// The blocks labelled with an address and no name, by address.
    labels: HashMap<u64, Id>,
}

impl<Id> Default for LocalNames<Id> {
    fn default() -> Self {
        Self {
            bases: Vec::new(),
            index: HashMap::default(),
            suffixes: Vec::new(),
            far: HashMap::default(),
            hex_bases: HashMap::default(),
            labels: HashMap::default(),
        }
    }
}

impl<Id: Copy + Eq> LocalNames<Id> {
    /// The base `base`, interned.
    pub(crate) fn intern(&mut self, base: &str) -> BaseId {
        if let Some(&id) = self.index.get(base) {
            return id;
        }
        let id = BaseId::from(self.bases.len());
        let suffixes = Suffixes::new(base);
        if let Some(address) = suffixes.hex {
            self.hex_bases.insert(address, id);
        }
        self.bases.push(base.into());
        self.index.insert(base.into(), id);
        self.suffixes.push(suffixes);
        id
    }

    /// The base `base`, if it is interned.
    pub(crate) fn base(&self, base: &str) -> Option<BaseId> {
        self.index.get(base).copied()
    }

    /// The name `text` spells: `base_<n>` with a canonical `n` is the
    /// base's `n`th name, anything else a bare base. Interns the base.
    pub(crate) fn parse(&mut self, text: &str) -> Name {
        let (base, suffix) = split_suffix(text);
        Name::new(self.intern(base), suffix)
    }

    /// Renders `name`: borrowed for a bare base, built for a suffixed one.
    pub(crate) fn render(&self, name: Name) -> Cow<'_, str> {
        let base = &self.bases[usize::from(name.base())];
        match name.suffix {
            0 => Cow::Borrowed(base),
            suffix => Cow::Owned(suffixed(base, suffix)),
        }
    }

    /// The value holding the name `text` spells, or labelled with the
    /// address it spells, if any.
    pub(crate) fn get(&self, text: &str) -> Option<Id> {
        let (base, suffix) = split_suffix(text);
        if let Some(base) = self.base(base)
            && let Some(id) = self.holder(Name::new(base, suffix))
        {
            return Some(id);
        }
        if suffix == 0
            && let Some(address) = hex_value(text)
        {
            return self.labels.get(&address).copied();
        }
        None
    }

    /// The value holding `name`, if any.
    fn holder(&self, name: Name) -> Option<Id> {
        let suffixes = &self.suffixes[usize::from(name.base())];
        match suffixes.taken.get(name.suffix as usize) {
            Some(Some(id)) => Some(*id),
            _ if suffixes.far > 0 => self.far.get(&(name.base(), name.suffix)).copied(),
            _ => None,
        }
    }

    /// Whether `name` is taken, by a value or by a label spelled like it.
    fn is_taken(&self, name: Name) -> bool {
        let suffixes = &self.suffixes[usize::from(name.base())];
        match suffixes.taken.get(name.suffix as usize) {
            Some(Some(_)) => return true,
            _ if suffixes.far > 0 && self.far.contains_key(&(name.base(), name.suffix)) => {
                return true;
            }
            _ => {}
        }
        name.suffix == 0
            && suffixes
                .hex
                .is_some_and(|address| self.labels.contains_key(&address))
    }

    fn duplicate(&self, name: Name) -> Error {
        Error::spanless(ErrorTy::DuplicateName(self.render(name).into_owned()))
    }

    /// Registers `name` for `id`. Errors if the name is taken.
    pub(crate) fn register(&mut self, name: Name, id: Id) -> Result<()> {
        if self.is_taken(name) {
            return Err(self.duplicate(name));
        }
        let suffixes = &mut self.suffixes[usize::from(name.base())];
        let at = name.suffix as usize;
        if at < suffixes.taken.len() + NEAR {
            if at >= suffixes.taken.len() {
                suffixes.taken.resize(at + 1, None);
            }
            suffixes.taken[at] = Some(id);
        } else {
            suffixes.far += 1;
            self.far.insert((name.base(), name.suffix), id);
        }
        Ok(())
    }

    /// Registers `id` under `base` or, if that is taken, the first free
    /// `base_<n>`, and returns the name it got. What a lift does for every
    /// value it names.
    pub(crate) fn register_unique(&mut self, base: BaseId, id: Id) -> Name {
        let bare = Name::new(base, 0);
        if !self.is_taken(bare) {
            let taken = &mut self.suffixes[usize::from(base)].taken;
            if taken.is_empty() {
                taken.push(None);
            }
            taken[0] = Some(id);
            return bare;
        }
        let Self { suffixes, far, .. } = self;
        let suffixes = &mut suffixes[usize::from(base)];
        let mut suffix = suffixes.hint.max(1);
        while suffixes
            .taken
            .get(suffix as usize)
            .is_some_and(Option::is_some)
            || (suffixes.far > 0 && far.contains_key(&(base, suffix)))
        {
            suffix += 1;
        }
        let at = suffix as usize;
        if at >= suffixes.taken.len() {
            suffixes.taken.resize(at + 1, None);
        }
        suffixes.taken[at] = Some(id);
        suffixes.hint = suffix + 1;
        Name::new(base, suffix)
    }

    /// The name `name` would become if registered now: itself when free,
    /// else the base's first free `base_<n>`. Registers nothing.
    pub(crate) fn peek_unique(&self, name: Name) -> Name {
        if !self.is_taken(name) {
            return name;
        }
        let base = name.base();
        let suffixes = &self.suffixes[usize::from(base)];
        let mut suffix = suffixes.hint.max(1);
        while self.holder(Name::new(base, suffix)).is_some() {
            suffix += 1;
        }
        Name::new(base, suffix)
    }

    /// Forgets `name`, so the next mint of its base may reuse the suffix.
    pub(crate) fn forget(&mut self, name: Name) {
        let suffixes = &mut self.suffixes[usize::from(name.base())];
        suffixes.hint = suffixes.hint.min(name.suffix);
        match suffixes.taken.get_mut(name.suffix as usize) {
            Some(slot) => {
                *slot = None;
            }
            None => {
                if self.far.remove(&(name.base(), name.suffix)).is_some() {
                    suffixes.far -= 1;
                }
            }
        }
    }

    /// Labels `id`, a block at `address` with no name, with the address.
    /// Errors if a name spelled like the address is taken: the block then
    /// needs a name of its own.
    pub(crate) fn label(&mut self, address: u64, id: Id) -> Result<()> {
        // The spelling is not built: a base that spells the address is
        // listed under it.
        if let Some(&base) = self.hex_bases.get(&address)
            && self.holder(Name::new(base, 0)).is_some()
        {
            return Err(Error::spanless(ErrorTy::DuplicateName(hex_name(address))));
        }
        match self.labels.entry(address) {
            std::collections::hash_map::Entry::Occupied(_) => {
                Err(Error::spanless(ErrorTy::DuplicateName(hex_name(address))))
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(id);
                Ok(())
            }
        }
    }

    /// Forgets the label at `address`.
    pub(crate) fn forget_label(&mut self, address: u64) {
        self.labels.remove(&address);
    }

    /// Every name and the value holding it, labels rendered, in no
    /// particular order.
    pub(crate) fn entries(&self) -> impl Iterator<Item = (Cow<'_, str>, Id)> + '_ {
        let named = self
            .suffixes
            .iter()
            .enumerate()
            .flat_map(|(base, suffixes)| {
                let text = &self.bases[base];
                suffixes
                    .taken
                    .iter()
                    .enumerate()
                    .filter_map(move |(suffix, id)| {
                        id.map(|id| {
                            let name = match suffix {
                                0 => Cow::Borrowed(text.as_ref()),
                                suffix => Cow::Owned(suffixed(text, suffix as u32)),
                            };
                            (name, id)
                        })
                    })
            });
        let far = self.far.iter().map(|(&(base, suffix), &id)| {
            (
                Cow::Owned(suffixed(&self.bases[usize::from(base)], suffix)),
                id,
            )
        });
        let labels = self
            .labels
            .iter()
            .map(|(&address, &id)| (Cow::Owned(hex_name(address)), id));
        named.chain(far).chain(labels)
    }

    /// Forgets every name and label, keeping the bases and the capacity.
    pub(crate) fn clear(&mut self) {
        // The bases, and so what each spells, are kept.
        for suffixes in &mut self.suffixes {
            suffixes.taken.clear();
            suffixes.hint = 1;
            suffixes.far = 0;
        }
        self.far.clear();
        self.labels.clear();
    }

    /// The interned bases, for serializing a body.
    pub(crate) fn bases(&self) -> &[Box<str>] {
        &self.bases
    }

    /// A table over `bases` holding nothing yet, for a body being
    /// deserialized, whose names are then registered from its values.
    pub(crate) fn from_bases(bases: Vec<Box<str>>) -> Self {
        let index = bases
            .iter()
            .enumerate()
            .map(|(id, base)| (base.clone(), BaseId::from(id)))
            .collect();
        let suffixes: Vec<Suffixes<Id>> = bases.iter().map(|base| Suffixes::new(base)).collect();
        let hex_bases = suffixes
            .iter()
            .enumerate()
            .filter_map(|(id, suffixes)| suffixes.hex.map(|address| (address, BaseId::from(id))))
            .collect();
        Self {
            bases,
            index,
            suffixes,
            far: HashMap::default(),
            hex_bases,
            labels: HashMap::default(),
        }
    }
}

/// `base_<suffix>`, without the formatting machinery.
fn suffixed(base: &str, suffix: u32) -> String {
    let mut digits = [0u8; 10];
    let mut at = digits.len();
    let mut rest = suffix;
    loop {
        at -= 1;
        digits[at] = b'0' + (rest % 10) as u8;
        rest /= 10;
        if rest == 0 {
            break;
        }
    }
    let mut name = String::with_capacity(base.len() + 1 + digits.len() - at);
    name.push_str(base);
    name.push('_');
    for &digit in &digits[at..] {
        name.push(char::from(digit));
    }
    name
}

/// The base and suffix `text` spells: `tmp_7` is `("tmp", 7)`; a name with
/// no `_<digits>` tail, or one whose digits are not how a suffix renders
/// (`tmp_07`, `tmp_0`), is a bare base.
pub(crate) fn split_suffix(text: &str) -> (&str, u32) {
    let bytes = text.as_bytes();
    let mut at = bytes.len();
    let mut suffix: u64 = 0;
    let mut scale: u64 = 1;
    while at > 0 && bytes[at - 1].is_ascii_digit() && scale <= 10_000_000_000 {
        at -= 1;
        suffix += u64::from(bytes[at] - b'0') * scale;
        scale *= 10;
    }
    let digits = bytes.len() - at;
    let canonical = digits > 0
        && at > 1
        && bytes[at - 1] == b'_'
        && bytes[at] != b'0'
        && u32::try_from(suffix).is_ok();
    if canonical {
        (&text[..at - 1], suffix as u32)
    } else {
        (text, 0)
    }
}

/// `value` in lowercase hex, as a block at that address is labelled.
pub(crate) fn hex_name(value: u64) -> String {
    let mut digits = [0u8; 16];
    let mut at = digits.len();
    let mut rest = value;
    loop {
        at -= 1;
        digits[at] = b"0123456789abcdef"[(rest & 0xf) as usize];
        rest >>= 4;
        if rest == 0 {
            break;
        }
    }
    let mut name = String::with_capacity(digits.len() - at);
    for &digit in &digits[at..] {
        name.push(char::from(digit));
    }
    name
}

/// The address `text` labels — lowercase hex, no leading zero, at most
/// sixteen digits — if it is one.
fn hex_value(text: &str) -> Option<u64> {
    let bytes = text.as_bytes();
    if bytes.is_empty() || bytes.len() > 16 || (bytes.len() > 1 && bytes[0] == b'0') {
        return None;
    }
    bytes.iter().try_fold(0u64, |value, &byte| {
        let digit = match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            _ => return None,
        };
        Some((value << 4) | u64::from(digit))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_is_eight_bytes_optional() {
        assert_eq!(std::mem::size_of::<Option<Name>>(), 8);
    }

    #[test]
    fn suffixes_split_as_they_render() {
        assert_eq!(split_suffix("tmp_7"), ("tmp", 7));
        assert_eq!(split_suffix("tmp_07"), ("tmp_07", 0));
        assert_eq!(split_suffix("tmp_0"), ("tmp_0", 0));
        assert_eq!(split_suffix("tmp"), ("tmp", 0));
        assert_eq!(split_suffix("_7"), ("_7", 0));
        assert_eq!(split_suffix("a_b_12"), ("a_b", 12));
        assert_eq!(split_suffix("x_99999999999"), ("x_99999999999", 0));
        assert_eq!(hex_value("2100"), Some(0x2100));
        assert_eq!(hex_value("02100"), None);
        assert_eq!(hex_value("0"), Some(0));
    }

    /// The table agrees with one flat map of strings, whichever tier a
    /// name lands in.
    #[test]
    fn the_table_behaves_as_one_map() {
        let mut table: LocalNames<u32> = LocalNames::default();
        let register = |table: &mut LocalNames<u32>, text: &str, id: u32| {
            let name = table.parse(text);
            table.register(name, id)
        };
        let mint = |table: &mut LocalNames<u32>, base: BaseId, id: u32| {
            let name = table.register_unique(base, id);
            table.render(name).into_owned()
        };
        for (text, id) in [("rax", 2), ("rax_2", 3), ("rax_007", 4), ("rax_9000", 5)] {
            register(&mut table, text, id).unwrap();
            assert!(register(&mut table, text, 99).is_err(), "{text} twice");
            assert_eq!(table.get(text), Some(id), "{text}");
        }
        table.label(0x2100, 1).unwrap();
        assert!(table.label(0x2100, 9).is_err());
        assert_eq!(table.get("2100"), Some(1));
        assert!(
            register(&mut table, "2100", 9).is_err(),
            "a label is a name"
        );
        assert_eq!(table.get("02100"), None);
        assert_eq!(table.get("rax_7"), None);
        assert_eq!(table.get("rax_1"), None);
        // Minting skips every taken suffix, whichever tier holds it.
        let rax = table.intern("rax");
        assert_eq!(mint(&mut table, rax, 6), "rax_1");
        assert_eq!(mint(&mut table, rax, 7), "rax_3");
        let rbx = table.intern("rbx");
        assert_eq!(mint(&mut table, rbx, 8), "rbx");
        let hex = table.intern("2100");
        assert_eq!(mint(&mut table, hex, 8), "2100_1");
        assert_eq!(table.get("2100_1"), Some(8));
        // Freeing lowers the hint; a far suffix is skipped when reached.
        let rax_2 = table.parse("rax_2");
        table.forget(rax_2);
        assert_eq!(table.get("rax_2"), None);
        assert_eq!(mint(&mut table, rax, 9), "rax_2");
        for id in 10..9010u32 {
            assert_ne!(
                mint(&mut table, rax, id),
                "rax_9000",
                "the far suffix is taken"
            );
        }
        assert_eq!(table.get("rax_9000"), Some(5));
        // Every name registered is listed exactly once.
        let mut names: Vec<String> = table.entries().map(|(name, _)| name.into_owned()).collect();
        let listed = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), listed);
        assert_eq!(listed, 4 + 1 + 3 + 1 + 9000);
        // Forgetting the bare name frees it for the next mint; forgetting
        // the label frees the spelling.
        let bare = table.parse("rax");
        table.forget(bare);
        assert_eq!(mint(&mut table, rax, 1), "rax");
        table.forget_label(0x2100);
        assert_eq!(table.get("2100"), None);
        register(&mut table, "2100", 9).unwrap();
        assert!(table.label(0x2100, 1).is_err(), "a name is a label");
    }
}
