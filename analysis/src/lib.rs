pub mod cfg_simplify;
pub use cfg_simplify::simplify_cfg;

pub mod dead_store;
pub use dead_store::{dead_reg_insns, remove_dead_reg_insns};

pub mod dce;
pub use dce::{dead_insns, remove_dead_insns};

pub mod alias;
pub use alias::{AliasResult, alias_analysis};

pub mod gvn;
pub use gvn::gvn;
