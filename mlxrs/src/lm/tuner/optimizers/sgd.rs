//! [`SGD`] — stochastic gradient descent with optional (Nesterov) momentum,
//! weight decay, and dampening.
//!
//! Mirrors Python `mlx.optimizers.SGD`
//! (`mlx/python/mlx/optimizers/optimizers.py:230..=294`) and Swift `SGD`
//! (`mlx-swift/Source/MLXOptimizers/Optimizers.swift`).
//!
//! Update formula (`optimizers.py:231..=294`):
//!
//! ```text
//! if weight_decay != 0:  g = g + weight_decay * w
//! if momentum <= 0:      w_new = w - lr * g                 (vanilla SGD)
//! else:                  v = momentum * v
//!                        v += (1 - dampening) * g     if dampening > 0
//!                        v += g                       otherwise
//!                        update = g + momentum * v    if nesterov
//!                                  v                  otherwise
//!                        w_new = w - lr * update
//! ```
//!
//! Per-parameter state: a single velocity `Array` keyed by parameter name.

use std::collections::HashMap;

use crate::{
  Array, Result,
  error::Error,
  lm::{
    load::Weights,
    tuner::optimizers::base::{LearningRate, Optimizer, zeros_like, zeros_like_map},
  },
  ops::arithmetic,
};

/// Stochastic gradient descent.
///
/// Mirrors Python `mlx.optimizers.SGD`
/// (`mlx/python/mlx/optimizers/optimizers.py:230..=294`).
pub struct SGD {
  /// Learning rate `λ` (or a step-driven schedule producing the same).
  pub learning_rate: LearningRate,
  /// Momentum coefficient `µ`. Default Python: `0.0` (vanilla SGD).
  pub momentum: f32,
  /// Weight decay (L2 penalty). Default Python: `0.0`.
  pub weight_decay: f32,
  /// Dampening `τ`. Default Python: `0.0`.
  pub dampening: f32,
  /// Enable Nesterov momentum. Default Python: `false`. Requires
  /// `momentum > 0` and `dampening == 0` (checked at construction).
  pub nesterov: bool,
  /// 1-based step counter, incremented at the top of every
  /// [`SGD::apply_gradients`] call (matches Python).
  step_count: usize,
  /// Last resolved learning rate after schedule eval (for
  /// [`Optimizer::learning_rate`]).
  current_lr: f32,
  /// Per-parameter velocity state `v` (Python `state["v"]`).
  state: HashMap<String, Array>,
}

impl SGD {
  /// Construct an [`SGD`] optimizer. Mirrors Python `SGD.__init__`
  /// (`optimizers.py:248..=266`).
  ///
  /// Errors with [`Error::Backend`] if `nesterov && (momentum <= 0 ||
  /// dampening != 0)` — same precondition as the Python `ValueError`.
  pub fn new(
    learning_rate: impl Into<LearningRate>,
    momentum: f32,
    weight_decay: f32,
    dampening: f32,
    nesterov: bool,
  ) -> Result<Self> {
    if nesterov && (momentum <= 0.0 || dampening != 0.0) {
      return Err(Error::Backend {
        message: "SGD: Nesterov momentum requires momentum > 0 and dampening == 0".into(),
      });
    }
    let lr = learning_rate.into();
    let current_lr = lr.current(0);
    Ok(Self {
      learning_rate: lr,
      momentum,
      weight_decay,
      dampening,
      nesterov,
      step_count: 0,
      current_lr,
      state: HashMap::new(),
    })
  }

  /// Construct a vanilla SGD (no momentum / decay / dampening / Nesterov).
  /// Convenience wrapper over [`SGD::new`].
  pub fn vanilla(learning_rate: impl Into<LearningRate>) -> Result<Self> {
    Self::new(learning_rate, 0.0, 0.0, 0.0, false)
  }
}

impl Optimizer for SGD {
  fn init(&mut self, params: &Weights) -> Result<()> {
    self.state = zeros_like_map(params)?;
    Ok(())
  }

  fn apply_gradients(&mut self, gradients: &Weights, params: &mut Weights) -> Result<()> {
    if self.state.is_empty() {
      self.init(gradients)?;
    }
    self.step_count += 1;
    self.current_lr = self.learning_rate.current(self.step_count);
    let lr = self.current_lr;

    for (key, grad) in gradients {
      let Some(param) = params.get(key) else {
        continue;
      };
      // ── Optional weight decay: g = g + weight_decay * w ──
      let effective_grad = if self.weight_decay != 0.0 {
        let wd = Array::full::<f32>(&[0i32; 0], self.weight_decay)?;
        let decay_term = arithmetic::multiply(&wd, param)?;
        arithmetic::add(grad, &decay_term)?
      } else {
        grad.try_clone()?
      };

      // ── Vanilla SGD branch (no momentum) ──
      if self.momentum <= 0.0 {
        let lr_scalar = Array::full::<f32>(&[0i32; 0], lr)?;
        let step = arithmetic::multiply(&lr_scalar, &effective_grad)?;
        let new_w = arithmetic::subtract(param, &step)?;
        params.insert(key.clone(), new_w);
        continue;
      }

      // ── Momentum / Nesterov branch ──
      let prev_v = match self.state.get(key) {
        Some(v) => v.try_clone()?,
        None => zeros_like(param)?,
      };
      let mu_scalar = Array::full::<f32>(&[0i32; 0], self.momentum)?;
      let v_scaled = arithmetic::multiply(&mu_scalar, &prev_v)?;
      let v_new = if self.dampening > 0.0 {
        let one_minus_damp = Array::full::<f32>(&[0i32; 0], 1.0 - self.dampening)?;
        let g_damped = arithmetic::multiply(&one_minus_damp, &effective_grad)?;
        arithmetic::add(&v_scaled, &g_damped)?
      } else {
        arithmetic::add(&v_scaled, &effective_grad)?
      };

      let update = if self.nesterov {
        // update = g + momentum * v
        let mu_v = arithmetic::multiply(&mu_scalar, &v_new)?;
        arithmetic::add(&effective_grad, &mu_v)?
      } else {
        v_new.try_clone()?
      };

      let lr_scalar = Array::full::<f32>(&[0i32; 0], lr)?;
      let step = arithmetic::multiply(&lr_scalar, &update)?;
      let new_w = arithmetic::subtract(param, &step)?;
      params.insert(key.clone(), new_w);
      self.state.insert(key.clone(), v_new);
    }
    Ok(())
  }

  fn step(&self) -> usize {
    self.step_count
  }

  fn learning_rate(&self) -> f32 {
    self.current_lr
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn scalar(v: f32) -> Result<Array> {
    Array::full::<f32>(&[0i32; 0], v)
  }

  fn read_scalar(a: &Array) -> Result<f32> {
    let mut clone = a.try_clone()?;
    clone.item::<f32>()
  }

  #[test]
  fn vanilla_sgd_single_step_matches_python_ref() -> Result<()> {
    // Python: w_new = w - lr * g; w=1.0, g=0.5, lr=0.1 → w_new = 0.95.
    let mut sgd = SGD::vanilla(0.1)?;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(0.5)?);
    sgd.apply_gradients(&grads, &mut params)?;
    let got = read_scalar(&params["w"])?;
    assert!((got - 0.95).abs() < 1e-6, "expected 0.95, got {got}");
    assert_eq!(sgd.step(), 1);
    assert!((sgd.learning_rate() - 0.1).abs() < 1e-6);
    Ok(())
  }

  #[test]
  fn sgd_with_momentum_single_step_matches_python_ref() -> Result<()> {
    // Python (first step, v=0): v = 0 + g; w_new = w - lr * v.
    // w=1.0, g=0.5, lr=0.1, momentum=0.9 → v=0.5, w_new=0.95.
    let mut sgd = SGD::new(0.1, 0.9, 0.0, 0.0, false)?;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(0.5)?);
    sgd.apply_gradients(&grads, &mut params)?;
    assert!((read_scalar(&params["w"])? - 0.95).abs() < 1e-6);
    let v = read_scalar(&sgd.state["w"])?;
    assert!((v - 0.5).abs() < 1e-6, "expected v=0.5, got {v}");
    Ok(())
  }

  #[test]
  fn sgd_with_weight_decay_matches_python_ref() -> Result<()> {
    // Python: g_eff = g + wd*w; w_new = w - lr * g_eff (vanilla; momentum=0).
    // w=2.0, g=1.0, lr=0.1, wd=0.5 → g_eff=2.0, w_new=1.8.
    let mut sgd = SGD::new(0.1, 0.0, 0.5, 0.0, false)?;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(2.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(1.0)?);
    sgd.apply_gradients(&grads, &mut params)?;
    assert!((read_scalar(&params["w"])? - 1.8).abs() < 1e-6);
    Ok(())
  }

  #[test]
  fn sgd_nesterov_precondition_rejects_zero_momentum() {
    let res = SGD::new(0.1, 0.0, 0.0, 0.0, true);
    assert!(matches!(res, Err(Error::Backend { .. })));
  }

  #[test]
  fn sgd_schedule_advances_lr_each_step() -> Result<()> {
    // Schedule: lr(step) = 0.1 / step (just to verify dispatch). Initial
    // lr is queried at step 0 (current_lr cached), then each
    // apply_gradients bumps to the next step.
    let sched: Box<dyn Fn(usize) -> f32> = Box::new(|step| 0.1 / (step as f32).max(1.0));
    let mut sgd = SGD::vanilla(LearningRate::Schedule(sched))?;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(1.0)?);
    sgd.apply_gradients(&grads, &mut params)?;
    assert!((sgd.learning_rate() - 0.1).abs() < 1e-6);
    sgd.apply_gradients(&grads, &mut params)?;
    assert!((sgd.learning_rate() - 0.05).abs() < 1e-6);
    Ok(())
  }
}
