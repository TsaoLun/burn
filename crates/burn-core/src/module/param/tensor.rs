use super::{LoraState, Param, ParamId, Parameter};
use crate::module::{
    AutodiffModule, Content, HasAutodiffModule, Module, ModuleDisplay, ModuleDisplayDefault,
    ModuleMapper, ModuleVisitor,
};
use crate::tensor::{
    Tensor,
    backend::{AutodiffBackend, Backend},
};
use alloc::{format, string::ToString, vec::Vec};
use burn_tensor::{Bool, Float, Int, TensorData, ops::Device};

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

impl<B: Backend, const D: usize> Parameter for Tensor<B, D, Float> {
    type Device = B::Device;

    fn device(&self) -> Self::Device {
        Tensor::device(self)
    }

    fn is_require_grad(&self) -> bool {
        Tensor::is_require_grad(self)
    }

    fn set_require_grad(self, require_grad: bool) -> Self {
        Tensor::set_require_grad(self, require_grad)
    }
}

impl<B: Backend, const D: usize> Parameter for Tensor<B, D, Int> {
    type Device = B::Device;

    fn device(&self) -> Self::Device {
        Tensor::device(self)
    }

    fn is_require_grad(&self) -> bool {
        false
    }

    fn set_require_grad(self, _require_grad: bool) -> Self {
        self
    }
}

impl<B: Backend, const D: usize> Parameter for Tensor<B, D, Bool> {
    type Device = B::Device;

    fn device(&self) -> Self::Device {
        Tensor::device(self)
    }

    fn is_require_grad(&self) -> bool {
        false
    }

    fn set_require_grad(self, _require_grad: bool) -> Self {
        self
    }
}

impl<B: Backend, const D: usize> Param<Tensor<B, D>> {
    /// Create a new parameter from a float tensor.
    ///
    /// # Warnings
    ///
    /// We strongly recommend using [Param::uninitialized] if you are using this method to
    /// initialize parameters inside a module, since the tensor initialization will be lazy,
    /// making the loading of weights more performant.
    pub fn from_tensor(value: Tensor<B, D>) -> Self {
        // When creating a parameter from a float tensor, we automatically mark it as requiring
        // gradients, so that it can be updated by an optimizer.
        Param::initialized(ParamId::new(), value.require_grad())
    }

    /// The shape of the parameter, **without triggering initialization**.
    ///
    /// This is critical for shape validation during loading: when applying tensors to an
    /// uninitialized parameter, we need to validate the shape without triggering the
    /// initialization function (which would allocate an unnecessary tensor).
    ///
    /// Use this instead of [crate::tensor::Tensor::shape] when you need the shape but want to
    /// preserve lazy initialization.
    pub fn lazy_shape(&self) -> burn_tensor::Shape {
        let initialization = match &self.initialization {
            Some(init) => init,
            None => return self.shape(),
        };

        let init = initialization.read().unwrap();

        match init.as_ref() {
            Some(value) => value.shape.clone(),
            None => self.shape(),
        }
    }

    /// Create a new parameter from data.
    pub fn from_data<T>(data: T, device: &B::Device) -> Self
    where
        T: Into<TensorData>,
    {
        let data: TensorData = data.into();
        // When creating a parameter from a float tensor, we automatically mark it as requiring
        // gradients, so that it can be updated by an optimizer.
        B::memory_persistent_allocations(device, data, |data| {
            let value = Tensor::from_data(data, device);
            Param::initialized(ParamId::new(), value.require_grad())
        })
    }

    /// Transform a parameter for loading by applying load transformations.
    ///
    /// This method is used to restore a parameter from a tensor (typically during deserialization).
    /// It ensures the tensor is moved to the expected device, applies the param mapper's
    /// `on_load` transformation, and preserves the autodiff settings (require_grad).
    pub fn transform_for_load(self, tensor: Tensor<B, D>, param_id: ParamId) -> Self {
        let mut new_tensor = tensor;

        let mapper = self.param_mapper.clone();

        let expected_device = self.lazy_device();
        let expected_require_grad = self.lazy_is_require_grad();

        // Make sure we load the tensor into the same module device.
        if new_tensor.device() != expected_device {
            new_tensor = new_tensor.to_device(&expected_device).detach();
        }

        new_tensor = mapper.on_load(new_tensor);

        // Make sure we load the tensor with the same autodiff setting.
        new_tensor = new_tensor.set_require_grad(expected_require_grad);

        let mut loaded = Self::initialized(param_id, new_tensor);
        loaded.param_mapper = mapper;
        loaded
    }

    /// Transform a parameter for saving by applying save transformations.
    ///
    /// This method is used to prepare a parameter for saving (typically during serialization).
    /// It applies the param mapper's `on_save` transformation, which can be used
    /// to modify the tensor before serialization (e.g., quantization, precision conversion).
    ///
    /// Uses `val_original()` to discard LoRA state, which is not serialized.
    pub fn transform_for_save(&self) -> Self {
        let mut tensor = self.val_original();
        let mapper = self.param_mapper.clone();

        tensor = mapper.on_save(tensor);

        Self::initialized(self.id, tensor)
    }
}

impl<B: Backend, const D: usize> Param<Tensor<B, D, Int>> {
    /// The shape of the parameter, **without triggering initialization**.
    ///
    /// This is critical for shape validation during loading: when applying tensors to an
    /// uninitialized parameter, we need to validate the shape without triggering the
    /// initialization function (which would allocate an unnecessary tensor).
    ///
    /// Use this instead of [crate::tensor::Tensor::shape] when you need the shape but want to
    /// preserve lazy initialization.
    pub fn lazy_shape(&self) -> burn_tensor::Shape {
        let initialization = match &self.initialization {
            Some(init) => init,
            None => return self.shape(),
        };

        let init = initialization.read().unwrap();

        match init.as_ref() {
            Some(value) => value.shape.clone(),
            None => self.shape(),
        }
    }

    /// Transform a parameter for loading by applying load transformations.
    ///
    /// This method is used to restore a parameter from a tensor (typically during deserialization).
    /// It ensures the tensor is moved to the expected device and applies the param mapper's
    /// `on_load` transformation.
    pub fn transform_for_load(self, tensor: Tensor<B, D, Int>, param_id: ParamId) -> Self {
        let mut new_tensor = tensor;

        let mapper = self.param_mapper.clone();

        let expected_device = self.lazy_device();

        // Make sure we load the tensor into the same module device.
        if new_tensor.device() != expected_device {
            new_tensor = new_tensor.to_device(&expected_device);
        }

        new_tensor = mapper.on_load(new_tensor);

        let mut loaded = Self::initialized(param_id, new_tensor);
        loaded.param_mapper = mapper;
        loaded
    }

    /// Transform a parameter for saving by applying save transformations.
    ///
    /// This method is used to prepare a parameter for saving (typically during serialization).
    /// It applies the param mapper's `on_save` transformation, which can be used
    /// to modify the tensor before serialization (e.g., quantization, precision conversion).
    pub fn transform_for_save(&self) -> Self {
        let mut tensor = self.val();
        let mapper = self.param_mapper.clone();

        tensor = mapper.on_save(tensor);

        Self::initialized(self.id, tensor)
    }
}

impl<B: Backend> Param<Tensor<B, 2>> {
    /// Attach a LoRA decomposition to this 2D parameter.
    ///
    /// Creates `LoraState` with `lora_a` ([in_features, rank], Kaiming uniform)
    /// and `lora_b` ([rank, out_features], zeros), scaled by `alpha / rank`.
    ///
    /// After calling this, `val()` returns `base + scaling * (lora_a @ lora_b)`.
    pub fn with_lora(mut self, rank: usize, alpha: f64, device: &B::Device) -> Self {
        let shape = self.lazy_shape();
        let in_features = shape.dims::<2>()[0]; // weight shape is [in_features, out_features]
        let out_features = shape.dims::<2>()[1];

        let lora_state = LoraState::new(rank, alpha, in_features, out_features, device);
        self.lora_transform = Some(Arc::new(lora_state));
        self
    }

    /// Lora-aware map for optimizer mappers.
    ///
    /// When a LoRA transform is attached, this splits the transform into
    /// `lora_a`/`lora_b` sub-params, maps each separately through the mapper,
    /// and reassembles the transform with updated params.
    ///
    /// Called from the generic `Param<Tensor<B, D>>::map` when D == 2 and
    /// `mapper.updates_lora_params()` is true.
    pub(crate) fn map_float_lora<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let (id, base_tensor, param_mapper, lora_transform) = self.into_parts();

        // Extract lora_a and lora_b if present; otherwise fall through to normal map
        let (lora_a, lora_b) = match lora_transform
            .as_ref()
            .and_then(|lt| {
                let a = lt.lora_a_param().cloned();
                let b = lt.lora_b_param().cloned();
                a.zip(b)
            }) {
            Some(pair) => pair,
            None => {
                let require_grad = base_tensor.is_require_grad();
                return Param {
                    id,
                    state: super::sync_once_cell::SyncOnceCell::initialized(base_tensor),
                    initialization: None,
                    param_mapper,
                    require_grad,
                    lora_transform,
                };
            }
        };

        // Map base (optimizer will skip it since require_grad is false)
        let base_param: Param<Tensor<B, 2>> = Param {
            id,
            state: super::sync_once_cell::SyncOnceCell::initialized(base_tensor),
            initialization: None,
            param_mapper: param_mapper.clone(),
            require_grad: false,
            lora_transform: None,
        };
        let base_param = mapper.map_float(base_param);
        let (base_id, base_tensor, _) = base_param.consume();

        // Map lora_a and lora_b through the optimizer mapper
        let lora_a = mapper.map_float(lora_a);
        let lora_b = mapper.map_float(lora_b);

        // Reassemble lora_transform with updated params
        let new_lora = lora_transform
            .expect("lora_transform must be Some here")
            .with_updated_params(lora_a, lora_b);

        let require_grad = base_tensor.is_require_grad();
        Param {
            id: base_id,
            state: super::sync_once_cell::SyncOnceCell::initialized(base_tensor),
            initialization: None,
            param_mapper,
            require_grad,
            lora_transform: Some(new_lora),
        }
    }
}

impl<B: Backend, const D: usize> Param<Tensor<B, D, Bool>> {
    /// The shape of the parameter, **without triggering initialization**.
    ///
    /// This is critical for shape validation during loading: when applying tensors to an
    /// uninitialized parameter, we need to validate the shape without triggering the
    /// initialization function (which would allocate an unnecessary tensor).
    ///
    /// **Returns:**
    /// - For uninitialized params: the shape from the `Uninitialized` struct
    /// - For initialized params: the actual shape from the tensor
    ///
    /// Use this instead of [crate::tensor::Tensor::shape] when you need the shape but want to
    /// preserve lazy initialization.
    pub fn lazy_shape(&self) -> burn_tensor::Shape {
        let initialization = match &self.initialization {
            Some(init) => init,
            None => return self.shape(),
        };

        let init = initialization.read().unwrap();

        match init.as_ref() {
            Some(value) => value.shape.clone(),
            None => self.shape(),
        }
    }

    /// Transform a parameter for loading by applying load transformations.
    ///
    /// This method is used to restore a parameter from a tensor (typically during deserialization).
    /// It ensures the tensor is moved to the expected device and applies the param mapper's
    /// `on_load` transformation.
    pub fn transform_for_load(self, tensor: Tensor<B, D, Bool>, param_id: ParamId) -> Self {
        let mut new_tensor = tensor;

        let mapper = self.param_mapper.clone();

        let expected_device = self.lazy_device();

        // Make sure we load the tensor into the same module device.
        if new_tensor.device() != expected_device {
            new_tensor = new_tensor.to_device(&expected_device);
        }

        new_tensor = mapper.on_load(new_tensor);

        let mut loaded = Self::initialized(param_id, new_tensor);
        loaded.param_mapper = mapper;
        loaded
    }

    /// Transform a parameter for saving by applying save transformations.
    ///
    /// This method is used to prepare a parameter for saving (typically during serialization).
    /// It applies the param mapper's `on_save` transformation, which can be used
    /// to modify the tensor before serialization (e.g., quantization, precision conversion).
    pub fn transform_for_save(&self) -> Self {
        let mut tensor = self.val();
        let mapper = self.param_mapper.clone();

        tensor = mapper.on_save(tensor);

        Self::initialized(self.id, tensor)
    }
}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D>> {
    type Record = Param<Tensor<B, D>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit_float(self);

        // When a LoRA transform is attached, expose lora_a and lora_b
        // as sub-parameters so visitors (e.g., gradient collectors) can discover them.
        if let Some(ref lora) = self.lora_transform {
            if let (Some(a), Some(b)) = (lora.lora_a_param(), lora.lora_b_param()) {
                visitor.enter_module("lora", "Struct:LoraState");
                visitor.visit_float(a);
                visitor.visit_float(b);
                visitor.exit_module("lora", "Struct:LoraState");
            }
        }
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        if mapper.updates_lora_params() && D == 2 && self.lora_transform.is_some() {
            // Safety: D == 2 is verified at runtime. Param<Tensor<B, D>> and
            // Param<Tensor<B, 2>> have the same memory layout when D == 2,
            // since const generics don't affect layout.
            let cast: Param<Tensor<B, 2>> = {
                let mut md = core::mem::ManuallyDrop::new(self);
                unsafe { core::ptr::read(&mut *md as *mut _ as *mut Param<Tensor<B, 2>>) }
            };
            let mapped = cast.map_float_lora(mapper);
            let result: Param<Tensor<B, D>> = {
                let mut md = core::mem::ManuallyDrop::new(mapped);
                unsafe { core::ptr::read(&mut *md as *mut _ as *mut Param<Tensor<B, D>>) }
            };
            result
        } else {
            mapper.map_float(self)
        }
    }

    fn into_record(self) -> Self::Record {
        self.transform_for_save()
    }

    fn load_record(self, record: Self::Record) -> Self {
        let (record_param_id, record_tensor, _) = record.consume();
        self.transform_for_load(record_tensor, record_param_id)
    }

    fn to_device(self, device: &Device<B>) -> Self {
        self.map(|tensor| tensor.to_device(device))
    }

    fn fork(self, device: &Device<B>) -> Self {
        self.map(|tensor| {
            let is_require_grad = tensor.is_require_grad();
            let mut tensor = tensor.to_device(device).detach();

            if is_require_grad {
                tensor = tensor.require_grad();
            }

            tensor
        })
    }

    fn collect_devices(&self, mut devices: Vec<Device<B>>) -> Vec<Device<B>> {
        let device = self.val().device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }
}

impl<const D: usize, B: Backend> ModuleDisplayDefault for Param<Tensor<B, D>> {
    fn content(&self, content: Content) -> Option<Content> {
        let id = if content.display_settings.show_param_id() {
            format!(", id: {}", self.id)
        } else {
            "".to_string()
        };
        let string = format!(
            "ParamTensor {{rank: {D}, shape: {:?}, kind: float{id}}}",
            self.shape().as_slice()
        );
        content.add_formatted(&string).optional()
    }
}
impl<const D: usize, B: Backend> ModuleDisplay for Param<Tensor<B, D>> {}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D, Int>> {
    type Record = Param<Tensor<B, D, Int>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit_int(self)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        mapper.map_int(self)
    }

    fn into_record(self) -> Self::Record {
        self.transform_for_save()
    }

    fn load_record(self, record: Self::Record) -> Self {
        let (record_param_id, record_tensor, _) = record.consume();
        self.transform_for_load(record_tensor, record_param_id)
    }

    fn to_device(self, device: &Device<B>) -> Self {
        self.map(|tensor| tensor.to_device(device))
    }

    fn fork(self, device: &Device<B>) -> Self {
        self.to_device(device) // Don't support autodiff.
    }

    fn collect_devices(&self, mut devices: Vec<Device<B>>) -> Vec<Device<B>> {
        let device = self.val().device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }
}

impl<const D: usize, B: Backend> ModuleDisplayDefault for Param<Tensor<B, D, Int>> {
    fn content(&self, content: Content) -> Option<Content> {
        let id = if content.display_settings.show_param_id() {
            format!(", id: {}", self.id)
        } else {
            "".to_string()
        };
        let string = format!(
            "ParamTensor {{rank: {D}, shape: {:?}, kind: int{id}}}",
            self.shape().as_slice()
        );
        content.add_formatted(&string).optional()
    }
}
impl<const D: usize, B: Backend> ModuleDisplay for Param<Tensor<B, D, Int>> {}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D, Bool>> {
    type Record = Param<Tensor<B, D, Bool>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit_bool(self)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        mapper.map_bool(self)
    }

    fn into_record(self) -> Self::Record {
        self.transform_for_save()
    }

    fn load_record(self, record: Self::Record) -> Self {
        let (record_param_id, record_tensor, _) = record.consume();
        self.transform_for_load(record_tensor, record_param_id)
    }

    fn to_device(self, device: &Device<B>) -> Self {
        self.map(|tensor| tensor.to_device(device))
    }

    fn fork(self, device: &Device<B>) -> Self {
        self.to_device(device) // Don't support autodiff.
    }

    fn collect_devices(&self, mut devices: Vec<Device<B>>) -> Vec<Device<B>> {
        let device = self.val().device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }
}

impl<const D: usize, B: Backend> ModuleDisplayDefault for Param<Tensor<B, D, Bool>> {
    fn content(&self, content: Content) -> Option<Content> {
        let id = if content.display_settings.show_param_id() {
            format!(", id: {}", self.id)
        } else {
            "".to_string()
        };

        let string = format!(
            "ParamTensor {{rank: {D}, shape: {:?}, kind: bool{id}}}",
            self.shape().as_slice()
        );
        content.add_formatted(&string).optional()
    }
}

impl<const D: usize, B: Backend> ModuleDisplay for Param<Tensor<B, D, Bool>> {}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for Param<Tensor<B, D>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D>>;

    fn valid(&self) -> Self::InnerModule {
        // Preserve initialized param `require_grad` state, but reset the inner value's
        let require_grad = self.require_grad;
        let mut param = Param::initialized(self.id, self.val().inner().set_require_grad(false));
        param.require_grad = require_grad;
        param
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        // Reinstate the param's `require_grad` state
        let tensor = Tensor::from_inner(module.val()).set_require_grad(module.require_grad);
        Param::initialized(module.id, tensor)
    }
}

impl<const D: usize, B: AutodiffBackend> HasAutodiffModule<B>
    for Param<Tensor<B::InnerBackend, D>>
{
    type TrainModule = Param<Tensor<B, D>>;
}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for Param<Tensor<B, D, Int>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D, Int>>;

    fn valid(&self) -> Self::InnerModule {
        Param::initialized(self.id, self.val().inner())
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        Param::initialized(module.id, Tensor::from_inner(module.val()))
    }
}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for Param<Tensor<B, D, Bool>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D, Bool>>;

    fn valid(&self) -> Self::InnerModule {
        Param::initialized(self.id, self.val().inner())
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        Param::initialized(module.id, Tensor::from_inner(module.val()))
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::{
        TestAutodiffBackend,
        module::Module,
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    };

    #[test]
    fn test_load_record_setting() {
        let device = Default::default();
        let tensor = Tensor::<TestAutodiffBackend, 2>::ones([3, 3], &device).require_grad();

        let byte_recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = byte_recorder
            .record(
                Param::initialized(ParamId::new(), tensor.clone()).into_record(),
                (),
            )
            .unwrap();

        let no_grad_is_require_grad = Param::initialized(ParamId::new(), tensor.clone())
            .no_grad()
            .load_record(byte_recorder.load(bytes.clone(), &device).unwrap())
            .is_require_grad();

        let with_default_is_require_grad = Param::initialized(ParamId::new(), tensor)
            .load_record(byte_recorder.load(bytes, &device).unwrap())
            .is_require_grad();

        assert!(!no_grad_is_require_grad);
        assert!(with_default_is_require_grad);
    }

    #[test]
    fn test_param_require_grad_stateful() {
        let device = Default::default();
        let tensor = Tensor::<TestAutodiffBackend, 2>::ones([3, 3], &device).require_grad();

        let param = Param::initialized(ParamId::new(), tensor);
        assert!(param.is_require_grad());
        assert!(param.require_grad);

        let param = param.valid();
        assert!(!param.is_require_grad());
        assert!(param.require_grad); // stateful

        // Without `HasAutodiffModule`, we would need to specify the param type as well, which would be annoying:
        // let param: Param<Tensor<TestAutodiffBackend, _>> = param.train();
        let param = param.train::<TestAutodiffBackend>();
        assert!(param.is_require_grad());
        assert!(param.require_grad); // stateful

        let param = param.no_grad();
        assert!(!param.is_require_grad());
        assert!(!param.require_grad); // stateful

        let param = param.valid();
        assert!(!param.is_require_grad()); // always
        assert!(!param.require_grad); // stateful

        let param = param.train::<TestAutodiffBackend>();
        assert!(!param.is_require_grad());
        assert!(!param.require_grad); // stateful
    }
}
