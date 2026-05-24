//! Function transforms: autograd (`value_and_grad`, `grad`, `vjp`, `jvp`),
//! custom-VJP overrides, gradient checkpointing, and bulk eval / async-eval.
//!
//! Mirrors `mlx-swift`'s `MLX.Transforms` (`Transforms.swift`,
//! `Transforms+Eval.swift`, `Transforms+Grad.swift`, `Transforms+Internal.swift`)
//! and `mlx.core.{value_and_grad,grad,vjp,jvp,custom_function,custom_vjp,
//! checkpoint,eval,async_eval}` on the Python side.
//!
//! ## API surface (foundation chunk)
//!
//! This chunk lands the foundational [`Closure`] wrapper + crate-private
//! marshalling helpers. Subsequent commits layer the autograd / custom /
//! checkpoint / eval transforms on top.
//!
//! ## Threading
//!
//! Like the rest of mlxrs, `Closure` and the (later-landed) `impl Fn`
//! callables are `!Send` + `!Sync` (they own [`crate::Array`] handles
//! transitively through the trampoline's closure, and mlx's evaluator is
//! single-threaded — see `crate::array::Array` for the rationale). The Rust
//! callable passed in (`F: Fn(&[Array]) -> Result<Vec<Array>>`) is required
//! `+ 'static` so it can outlive the construction scope and be invoked from
//! mlx-c.

pub mod closure;

pub use closure::Closure;
