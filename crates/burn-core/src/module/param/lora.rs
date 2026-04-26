use super::{Param, Parameter};
use burn_tensor::{Tensor, backend::Backend};
use burn_tensor::Distribution;

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

/// Trait for applying a transformation to a parameter (e.g., LoRA).
///
/// `LoraTransform<T>` is stored as a trait object in `Param<T>::lora_transform`
/// and is invoked by `Param::val()` to compute the effective parameter value.
pub trait LoraTransform<T: Parameter>: core::fmt::Debug + Send + Sync {
    /// Apply the transformation to the base value and return the result.
    fn apply(&self, base: T) -> T;

    /// The rank of the LoRA decomposition.
    fn rank(&self) -> usize;

    /// Returns a reference to the lora_a parameter, if this transform exposes it.
    fn lora_a_param(&self) -> Option<&Param<T>> {
        None
    }

    /// Returns a reference to the lora_b parameter, if this transform exposes it.
    fn lora_b_param(&self) -> Option<&Param<T>> {
        None
    }

    /// Returns a new transform with updated lora_a and lora_b parameters.
    ///
    /// This is used by the optimizer to update LoRA weights after a gradient step.
    fn with_updated_params(
        &self,
        new_a: Param<T>,
        new_b: Param<T>,
    ) -> Arc<dyn LoraTransform<T> + Send + Sync>;
}

/// LoRA state for a single 2D weight matrix.
///
/// Contains the low-rank decomposition matrices `lora_a` (in_features × rank)
/// and `lora_b` (rank × out_features), plus a scaling factor.
pub struct LoraState<B: Backend> {
    /// LoRA A matrix: [in_features, rank], initialized with Kaiming uniform.
    pub lora_a: Param<Tensor<B, 2>>,
    /// LoRA B matrix: [rank, out_features], initialized to zeros.
    pub lora_b: Param<Tensor<B, 2>>,
    /// Scaling factor: alpha / rank.
    pub scaling: f64,
}

impl<B: Backend> core::fmt::Debug for LoraState<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LoraState")
            .field("rank", &self.rank())
            .field("scaling", &self.scaling)
            .finish()
    }
}

impl<B: Backend> LoraState<B> {
    /// Create a new LoRA state with the given rank, alpha, and device.
    ///
    /// `lora_a` is initialized with Kaiming uniform, `lora_b` with zeros.
    pub fn new(
        rank: usize,
        alpha: f64,
        in_features: usize,
        out_features: usize,
        device: &B::Device,
    ) -> Self {
        let a_shape = [in_features, rank];
        let b_shape = [rank, out_features];

        let lora_a = {
            let tensor = Tensor::random(a_shape, Distribution::Uniform(-1.0, 1.0), device)
                .mul_scalar((1.0 / (in_features as f64)).sqrt());
            Param::from_tensor(tensor)
        };

        let lora_b = {
            let tensor = Tensor::zeros(b_shape, device);
            Param::from_tensor(tensor)
        };

        Self {
            lora_a,
            lora_b,
            scaling: alpha / rank as f64,
        }
    }

    /// Returns the rank of the LoRA decomposition.
    pub fn rank(&self) -> usize {
        self.lora_a.val_original().dims()[1]
    }
}

/// Computes the LoRA weight delta: `scaling * (lora_a @ lora_b)`.
///
/// lora_a: `[in_features, rank]`, lora_b: `[rank, out_features]`
/// Result: `[in_features, out_features]` — same shape as the base weight.
fn compute_delta<B: Backend>(
    base: Tensor<B, 2>,
    lora_a: &Tensor<B, 2>,
    lora_b: &Tensor<B, 2>,
    scaling: f64,
) -> Tensor<B, 2> {
    let delta = lora_a
        .clone()
        .matmul(lora_b.clone())
        .mul_scalar(scaling);

    base + delta
}

impl<B: Backend> LoraTransform<Tensor<B, 2>> for LoraState<B> {
    fn apply(&self, base: Tensor<B, 2>) -> Tensor<B, 2> {
        let lora_a = self.lora_a.val();
        let lora_b = self.lora_b.val();
        compute_delta(base, &lora_a, &lora_b, self.scaling)
    }

    fn rank(&self) -> usize {
        self.rank()
    }

    fn lora_a_param(&self) -> Option<&Param<Tensor<B, 2>>> {
        Some(&self.lora_a)
    }

    fn lora_b_param(&self) -> Option<&Param<Tensor<B, 2>>> {
        Some(&self.lora_b)
    }

    fn with_updated_params(
        &self,
        new_a: Param<Tensor<B, 2>>,
        new_b: Param<Tensor<B, 2>>,
    ) -> Arc<dyn LoraTransform<Tensor<B, 2>> + Send + Sync> {
        Arc::new(LoraState {
            lora_a: new_a,
            lora_b: new_b,
            scaling: self.scaling,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    type B = TestBackend;

    #[test]
    fn val_no_lora_returns_original() {
        let device = Default::default();
        let base = Tensor::<B, 2>::ones([4, 8], &device);
        let param: Param<Tensor<B, 2>> = Param::from_tensor(base.clone());
        assert_eq!(param.val().to_data(), base.to_data());
    }

    #[test]
    fn val_original_unaffected_by_lora() {
        let device = Default::default();
        let base = Tensor::<B, 2>::ones([4, 8], &device);
        let param: Param<Tensor<B, 2>> = Param::from_tensor(base.clone());
        let with_lora = param.clone().with_lora(2, 2.0, &device);
        // val_original should still return the base
        assert_eq!(with_lora.val_original().to_data(), base.to_data());
    }

    #[test]
    fn val_with_lora_equals_original_at_init() {
        let device = Default::default();
        let base = Tensor::<B, 2>::ones([4, 8], &device);
        let param: Param<Tensor<B, 2>> = Param::from_tensor(base.clone());
        let with_lora = param.with_lora(2, 2.0, &device);
        // At initialization, lora_b is zeros → delta is zero → val() == base
        assert_eq!(with_lora.val().to_data(), base.to_data());
    }

    #[test]
    fn lora_delta_nonzero_when_lora_b_is_ones() {
        let device = Default::default();
        let base = Tensor::<B, 2>::ones([4, 8], &device);
        let param: Param<Tensor<B, 2>> = Param::from_tensor(base.clone());

        // Build LoraState with ones → delta = scaling * (ones @ ones) ≠ 0
        // base=[4, 8] → in_features=4, out_features=8, rank=2
        let lora_a = Param::from_tensor(Tensor::<B, 2>::ones([4, 2], &device));
        let lora_b = Param::from_tensor(Tensor::<B, 2>::ones([2, 8], &device));
        let lora = LoraState { lora_a, lora_b, scaling: 1.0 };
        let with_lora = param.with_lora_transform(Arc::new(lora));

        assert_ne!(with_lora.val().to_data(), base.to_data());
    }

    #[test]
    fn has_lora_reflects_state() {
        let device = Default::default();
        let base = Tensor::<B, 2>::ones([4, 8], &device);
        let param: Param<Tensor<B, 2>> = Param::from_tensor(base);
        assert!(!param.has_lora());
        let with_lora = param.with_lora(2, 2.0, &device);
        assert!(with_lora.has_lora());
    }

    #[test]
    fn merge_lora_removes_transform() {
        let device = Default::default();
        let base = Tensor::<B, 2>::ones([4, 8], &device);
        let param: Param<Tensor<B, 2>> = Param::from_tensor(base);
        let with_lora = param.with_lora(2, 2.0, &device);
        assert!(with_lora.has_lora());
        let merged = with_lora.merge_lora();
        assert!(!merged.has_lora());
    }

    #[test]
    fn merge_lora_on_no_lora_is_noop() {
        let device = Default::default();
        let base = Tensor::<B, 2>::ones([4, 8], &device);
        let param: Param<Tensor<B, 2>> = Param::from_tensor(base.clone());
        let merged = param.merge_lora();
        assert_eq!(merged.val().to_data(), base.to_data());
    }

    #[test]
    fn merge_lora_bakes_delta_into_base() {
        let device = Default::default();
        let base = Tensor::<B, 2>::ones([4, 8], &device);
        let param: Param<Tensor<B, 2>> = Param::from_tensor(base);
        let with_lora = param.with_lora(2, 2.0, &device);
        let lora_val = with_lora.val().to_data();
        let merged = with_lora.merge_lora();
        // After merge, val() should return the same as before merge
        assert_eq!(merged.val().to_data(), lora_val);
    }

    #[test]
    fn custom_lora_transform_is_applied() {
        // A test transform that doubles the base value
        #[derive(Debug)]
        struct DoublingTransform;

        impl LoraTransform<Tensor<B, 2>> for DoublingTransform {
            fn apply(&self, base: Tensor<B, 2>) -> Tensor<B, 2> {
                base.mul_scalar(2.0)
            }

            fn rank(&self) -> usize {
                0
            }

            fn with_updated_params(
                &self,
                _new_a: Param<Tensor<B, 2>>,
                _new_b: Param<Tensor<B, 2>>,
            ) -> Arc<dyn LoraTransform<Tensor<B, 2>> + Send + Sync> {
                Arc::new(DoublingTransform)
            }
        }

        let device = Default::default();
        let base = Tensor::<B, 2>::ones([3, 3], &device);
        let param: Param<Tensor<B, 2>> = Param::from_tensor(base);
        let with_transform = param.with_lora_transform(Arc::new(DoublingTransform));
        let result = with_transform.val();
        let expected = Tensor::<B, 2>::ones([3, 3], &device).mul_scalar(2.0);
        assert_eq!(result.to_data(), expected.to_data());
    }
}
