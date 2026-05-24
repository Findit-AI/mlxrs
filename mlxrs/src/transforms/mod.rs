//! Function transforms: autograd (`value_and_grad`, `grad`, `vjp`, `jvp`),
//! custom-VJP overrides, gradient checkpointing, and bulk eval / async-eval.
//!
//! Mirrors `mlx-swift`'s `MLX.Transforms` (`Transforms.swift`,
//! `Transforms+Eval.swift`, `Transforms+Grad.swift`, `Transforms+Internal.swift`)
//! and `mlx.core.{value_and_grad,grad,vjp,jvp,custom_function,custom_vjp,
//! checkpoint,eval,async_eval}` on the Python side.
//!
//! ## API surface (custom-VJP chunk)
//!
//! - [`crate::transforms::closure::Closure`] — RAII wrapper over
//!   `mlx_closure` (foundation).
//! - [`crate::transforms::autograd::value_and_grad`] /
//!   [`crate::transforms::autograd::grad`] / [`crate::transforms::autograd::vjp`]
//!   / [`crate::transforms::autograd::jvp`] — autograd.
//! - [`crate::transforms::custom::custom_vjp`] /
//!   [`crate::transforms::custom::custom_function`] — wrap a forward function
//!   with a user-defined backward (cotangent) function, overriding the
//!   autograd-derived VJP.
//! - [`crate::transforms::eval::eval`] /
//!   [`crate::transforms::eval::async_eval`] — bulk eval / async-eval.
//!
//! Checkpoint lands in the next chunk.

pub mod autograd;
pub mod closure;
pub mod custom;
pub mod eval;

pub use autograd::{grad, jvp, value_and_grad, vjp};
pub use closure::Closure;
pub use custom::{custom_function, custom_vjp};
pub use eval::{async_eval, eval};
