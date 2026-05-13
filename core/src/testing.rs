use crate::{
    context::Context,
    space::{Space, SpaceId, SpaceType},
    value::{
        Renameable,
        varnode::{Varnode, VarnodeId},
    },
};

/// A minimal [`Context`] with a register space and a small set of named varnodes,
/// intended for use in unit tests that need register-like storage without depending
/// on any specific architecture.
///
/// Layout of the register space:
///
/// ```text
/// offset: 0        1        2        3        4        5        6        7
///         ├────────────────────────────────────────────────────────────────┤  r0 (8 bytes)
///         ├────────────────────────┤                                          r0_lo32 (4 bytes)
///         ├────────┤                                                           r0_lo16 (2 bytes)
///         ├───┤                                                                r0_byte0 (1 byte)
///                  ├───┤                                                       r0_byte1 (1 byte)
/// offset: 8       16       24
///         ├───────┤                                                            r1 (8 bytes)
///                  ├───────┤                                                   r2 (8 bytes)
///                           ├───────┤                                          r3 (8 bytes)
/// ```
pub struct TestContext {
    pub ctx: Context<'static>,

    /// The ID of the register space.
    pub reg_space: SpaceId,

    /// 8-byte registers at non-overlapping offsets.
    pub r0: VarnodeId,
    pub r1: VarnodeId,
    pub r2: VarnodeId,
    pub r3: VarnodeId,

    /// Sub-registers of `r0`, useful for testing partial-overlap scenarios.
    pub r0_lo32: VarnodeId, // 4 bytes at offset 0
    pub r0_lo16: VarnodeId,  // 2 bytes at offset 0
    pub r0_byte0: VarnodeId, // 1 byte at offset 0
    pub r0_byte1: VarnodeId, // 1 byte at offset 1
}

impl TestContext {
    pub fn new() -> Self {
        let mut ctx = Context::new();

        let mut reg_space_def = Space::new(Some("register"), 1, 4);
        reg_space_def.ty = SpaceType::Register;
        let reg_space = ctx.spaces.push(reg_space_def);
        ctx.named_spaces.insert(Box::from("register"), reg_space);

        let make = |ctx: &mut Context<'static>, offset: i64, size: usize, name: &'static str| {
            Varnode::make(ctx, offset, size, reg_space)
                .with_name(name.into())
                .expect("register names are unique")
                .id
        };

        let r0 = make(&mut ctx, 0, 8, "r0");
        let r1 = make(&mut ctx, 8, 8, "r1");
        let r2 = make(&mut ctx, 16, 8, "r2");
        let r3 = make(&mut ctx, 24, 8, "r3");

        let r0_lo32 = make(&mut ctx, 0, 4, "r0_lo32");
        let r0_lo16 = make(&mut ctx, 0, 2, "r0_lo16");
        let r0_byte0 = make(&mut ctx, 0, 1, "r0_byte0");
        let r0_byte1 = make(&mut ctx, 1, 1, "r0_byte1");

        Self {
            ctx,
            reg_space,
            r0,
            r1,
            r2,
            r3,
            r0_lo32,
            r0_lo16,
            r0_byte0,
            r0_byte1,
        }
    }
}

impl Default for TestContext {
    fn default() -> Self {
        Self::new()
    }
}
