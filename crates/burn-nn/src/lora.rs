//! LoRA (Low-Rank Adaptation) module for Burn.
//!
//! Provides `LoraConfig` for batch injection of LoRA into existing models,
//! `merge_lora` for baking LoRA weights into the base model for inference,
//! and `freeze_non_lora` for freezing base weights during LoRA fine-tuning.

use burn_core::module::{Module, ModuleMapper, Param};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::Tensor;

/// Configuration for applying LoRA to a model.
///
/// # Example
///
/// ```ignore
/// let config = LoraConfig::new(8, 16.0)
///     .with_target_modules(vec!["q_proj", "v_proj"]);
/// let model = config.apply(model, &device);
/// ```
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// The rank of the LoRA decomposition.
    pub rank: usize,
    /// Alpha scaling factor. The actual scaling is `alpha / rank`.
    pub alpha: f64,
    /// Optional list of module name patterns to target.
    /// If empty, all 2D parameters get LoRA.
    pub target_modules: Vec<String>,
}

impl LoraConfig {
    /// Create a new LoRA configuration.
    pub fn new(rank: usize, alpha: f64) -> Self {
        Self {
            rank,
            alpha,
            target_modules: Vec::new(),
        }
    }

    /// Set the target module name patterns.
    ///
    /// Only parameters whose path contains one of these patterns will get LoRA.
    pub fn with_target_modules(mut self, target_modules: Vec<&str>) -> Self {
        self.target_modules = target_modules.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Apply LoRA to all matching 2D parameters in the given module.
    ///
    /// This injects a `LoraState` into each `Param<Tensor<B, 2>>` whose path
    /// matches one of the target module patterns (or all 2D params if no targets).
    pub fn apply<B: Backend, M: Module<B>>(&self, module: M, device: &B::Device) -> M {
        let mut injector = LoraInjector::new(self.rank, self.alpha, &self.target_modules, device);
        module.map(&mut injector)
    }
}

/// ModuleMapper that injects LoRA into matching 2D parameters.
struct LoraInjector<'a, B: Backend> {
    rank: usize,
    alpha: f64,
    target_modules: &'a [String],
    device: &'a B::Device,
    /// Current module path stack for matching target_modules.
    path_stack: Vec<String>,
}

impl<'a, B: Backend> LoraInjector<'a, B> {
    fn new(
        rank: usize,
        alpha: f64,
        target_modules: &'a [String],
        device: &'a B::Device,
    ) -> Self {
        Self {
            rank,
            alpha,
            target_modules,
            device,
            path_stack: Vec::new(),
        }
    }

    fn path_matches(&self) -> bool {
        if self.target_modules.is_empty() {
            return true;
        }
        let path = self.path_stack.join(".");
        self.target_modules.iter().any(|t| path.contains(t.as_str()))
    }
}

impl<B: Backend> ModuleMapper<B> for LoraInjector<'_, B> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path_stack.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path_stack.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if D == 2 && self.path_matches() {
            // Safety: D == 2 is verified. Param<Tensor<B, D>> and Param<Tensor<B, 2>>
            // have identical memory layout when D == 2.
            let cast: Param<Tensor<B, 2>> = {
                let mut md = core::mem::ManuallyDrop::new(param);
                unsafe { core::ptr::read(&mut *md as *mut _ as *mut Param<Tensor<B, 2>>) }
            };
            let injected = cast.with_lora(self.rank, self.alpha, self.device);
            let result: Param<Tensor<B, D>> = {
                let mut md = core::mem::ManuallyDrop::new(injected);
                unsafe { core::ptr::read(&mut *md as *mut _ as *mut Param<Tensor<B, D>>) }
            };
            result
        } else {
            param
        }
    }
}

/// Merge all LoRA deltas into base weights and remove the LoRA transforms.
///
/// After merging, the model can be used for inference with no LoRA overhead.
///
/// # Example
///
/// ```ignore
/// let merged_model = merge_lora(finetuned_model);
/// ```
pub fn merge_lora<B: Backend, M: Module<B>>(module: M) -> M {
    let mut merger = LoraMerger;
    module.map(&mut merger)
}

/// ModuleMapper that merges LoRA deltas into base weights.
struct LoraMerger;

impl<B: Backend> ModuleMapper<B> for LoraMerger {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        param.merge_lora()
    }
}

/// Freeze all non-LoRA parameters in the module.
///
/// LoRA parameters (`lora_a`, `lora_b`) keep `require_grad = true`, while
/// all other parameters get `require_grad = false`.
///
/// # Example
///
/// ```ignore
/// let model = freeze_non_lora(model);
/// ```
pub fn freeze_non_lora<B: Backend, M: Module<B>>(module: M) -> M {
    let mut freezer = LoraFreezer::new();
    module.map(&mut freezer)
}

/// ModuleMapper that freezes base weights while keeping LoRA params trainable.
struct LoraFreezer {
    /// Tracks the nesting depth within LoRA scopes.
    /// When > 0, we're inside a LoRA sub-module (lora_a/lora_b scope).
    lora_depth: usize,
}

impl LoraFreezer {
    fn new() -> Self {
        Self { lora_depth: 0 }
    }
}

impl<B: Backend> ModuleMapper<B> for LoraFreezer {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        if name == "lora" {
            self.lora_depth += 1;
        }
    }

    fn exit_module(&mut self, name: &str, _container_type: &str) {
        if name == "lora" {
            self.lora_depth = self.lora_depth.saturating_sub(1);
        }
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if self.lora_depth > 0 {
            // Inside lora scope: keep require_grad = true for lora_a/lora_b
            param
        } else {
            // Outside lora scope: freeze base weights
            param.set_require_grad(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Linear, LinearConfig};
    use burn_core::module::{Module, ModuleVisitor, Param, ParamId};
    use burn_core::tensor::backend::Backend;

    type B = burn_flex::Flex;

    fn create_test_module<B_: Backend>(device: &B_::Device) -> Linear<B_> {
        LinearConfig::new(8, 4).init(device)
    }

    /// Returns true if any visited float param has LoRA attached.
    fn any_has_lora<M: Module<B>>(m: &M) -> bool {
        struct C(bool);
        impl ModuleVisitor<B> for C {
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
                if p.has_lora() { self.0 = true; }
            }
        }
        let mut c = C(false);
        m.visit(&mut c);
        c.0
    }

    /// Returns the number of float params visited.
    fn count_float_params<M: Module<B>>(m: &M) -> usize {
        struct C(usize);
        impl ModuleVisitor<B> for C {
            fn visit_float<const D: usize>(&mut self, _: &Param<Tensor<B, D>>) {
                self.0 += 1;
            }
        }
        let mut c = C(0);
        m.visit(&mut c);
        c.0
    }

    #[test]
    fn lora_init_preserves_output() {
        let device = Default::default();
        let module = create_test_module::<B>(&device);
        let x = Tensor::<B, 2>::ones([1, 8], &device);
        let before = module.forward(x.clone());

        let config = LoraConfig::new(2, 2.0);
        let lora_module = config.apply(module, &device);

        let after = lora_module.forward(x);
        // LoRA init: lora_b is zeros, so output should be unchanged
        assert_eq!(before.to_data(), after.to_data());
    }

    #[test]
    fn has_lora_before_and_after_injection() {
        let device = Default::default();
        let module = create_test_module::<B>(&device);
        assert!(!any_has_lora(&module));

        let config = LoraConfig::new(2, 2.0);
        let module = config.apply(module, &device);
        assert!(any_has_lora(&module));
    }

    #[test]
    fn merge_lora_clears_all_transforms() {
        let device = Default::default();
        let module = create_test_module::<B>(&device);
        let config = LoraConfig::new(2, 2.0);
        let module = config.apply(module, &device);
        let module = merge_lora(module);
        assert!(!any_has_lora(&module));
    }

    #[test]
    fn merge_lora_preserves_output() {
        let device = Default::default();
        let module = create_test_module::<B>(&device);
        let x = Tensor::<B, 2>::ones([1, 8], &device);

        let config = LoraConfig::new(2, 2.0);
        let lora_module = config.apply(module, &device);

        // Forward before merge
        let output_before = lora_module.forward(x.clone());

        // Merge and forward again
        let merged_module = merge_lora(lora_module);
        let output_after = merged_module.forward(x.clone());

        // lora_b is zero-initialized → delta=0 → merge is a no-op
        assert_eq!(
            output_before.to_data(),
            output_after.to_data(),
            "Forward output should be preserved after merge (lora_b=0 → delta=0)"
        );
    }

    #[test]
    fn val_original_is_unaffected_by_lora() {
        let device = Default::default();
        let module = create_test_module::<B>(&device);

        // Collect base values before LoRA injection
        let before: Vec<_> = {
            struct C(Vec<burn_core::tensor::TensorData>);
            impl<B: Backend> ModuleVisitor<B> for C {
                fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
                    self.0.push(p.val().to_data());
                }
            }
            let mut c = C(Vec::new());
            module.visit(&mut c);
            c.0
        };

        let config = LoraConfig::new(2, 2.0);
        let module = config.apply(module, &device);

        // val_original() must return the pre-LoRA values for base params.
        // Skip lora_a/lora_b sub-params (inside "lora" scope).
        let after: Vec<_> = {
            struct C(Vec<burn_core::tensor::TensorData>, usize);
            impl<B: Backend> ModuleVisitor<B> for C {
                fn enter_module(&mut self, name: &str, _: &str) {
                    if name == "lora" { self.1 += 1; }
                }
                fn exit_module(&mut self, name: &str, _: &str) {
                    if name == "lora" { self.1 = self.1.saturating_sub(1); }
                }
                fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
                    if self.1 == 0 { self.0.push(p.val_original().to_data()); }
                }
            }
            let mut c = C(Vec::new(), 0);
            module.visit(&mut c);
            c.0
        };
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(after.iter()) {
            assert_eq!(a, b, "val_original() changed after LoRA injection");
        }
    }

    #[test]
    fn lora_config_apply_all_params() {
        let device = Default::default();
        let module = create_test_module::<B>(&device);
        let config = LoraConfig::new(2, 2.0);
        let module = config.apply(module, &device);

        // weight is 2D → should have lora
        assert!(module.weight.has_lora());
    }

    #[test]
    fn lora_config_apply_target_filter() {
        let device = Default::default();
        let module = create_test_module::<B>(&device);
        let config = LoraConfig::new(2, 2.0).with_target_modules(vec!["nonexistent"]);
        let module = config.apply(module, &device);
        assert!(!any_has_lora(&module));
    }

    #[test]
    fn freeze_non_lora_freezes_base_weights() {
        let device = Default::default();
        let module = create_test_module::<B>(&device);
        let config = LoraConfig::new(2, 2.0);
        let module = config.apply(module, &device);
        let module = freeze_non_lora(module);

        // The weight param (outside lora scope) should be frozen
        assert!(!module.weight.is_require_grad());
    }

    #[test]
    fn freeze_non_lora_output_unchanged() {
        let device = Default::default();
        let module = create_test_module::<B>(&device);
        let x = Tensor::<B, 2>::ones([1, 8], &device);
        let before = module.forward(x.clone());

        let config = LoraConfig::new(2, 2.0);
        let module = config.apply(module, &device);
        let module = freeze_non_lora(module);
        let after = module.forward(x);

        // Output should be unchanged (lora_b is zeros)
        assert_eq!(before.to_data(), after.to_data());
    }

    #[test]
    fn visit_exposes_lora_params() {
        let device = Default::default();
        let module = create_test_module::<B>(&device);
        let config = LoraConfig::new(2, 2.0);
        let module = config.apply(module, &device);

        // Collect all param IDs (including lora sub-params)
        let ids: Vec<ParamId> = {
            struct IdCollector(Vec<ParamId>);
            impl<B: Backend> ModuleVisitor<B> for IdCollector {
                fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                    self.0.push(param.id);
                }
            }
            let mut c = IdCollector(Vec::new());
            module.visit(&mut c);
            c.0
        };

        // Should have more params than just weight + bias (lora_a and lora_b are added)
        assert!(ids.len() >= 3, "Expected at least 3 params (weight, bias, lora_a, lora_b), got {}", ids.len());
    }

    #[test]
    fn visit_param_count_restored_after_merge() {
        let device = Default::default();
        let module = create_test_module::<B>(&device);
        let config = LoraConfig::new(2, 2.0);
        let module = config.apply(module, &device);

        let count_before = count_float_params(&module);
        let module = merge_lora(module);
        let count_after = count_float_params(&module);

        // After merge, lora_a/lora_b are removed → fewer params
        assert!(count_after < count_before,
            "After merge expected fewer params, got before={count_before} after={count_after}");
    }
}
