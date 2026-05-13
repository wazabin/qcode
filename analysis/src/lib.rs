pub mod cfg_simplify;
pub use cfg_simplify::simplify_cfg;

pub mod dead_load;
pub use dead_load::{dead_load_insns, remove_dead_load_insns_block};

pub mod dce;
pub use dce::{dead_insns, remove_dead_insns};

pub mod alias;
pub use alias::{AliasResult, alias_analysis};

pub mod gvn;
pub use gvn::{gvn, gvn_function};

pub mod mem2reg;
pub use mem2reg::mem2reg;
