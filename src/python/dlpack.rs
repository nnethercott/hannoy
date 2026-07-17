use dlpark::{SafeManagedTensorVersioned, ffi::{DataType, Flags}, traits::TensorView};
use pyo3::{exceptions::PyValueError, PyErr, PyResult};

/// A Rust type corresponding to one DLPack dtype, usable with [`MutTensorExt`].
pub(super) trait DlpackDtype {
    /// Whether a tensor's DLPack dtype is `Self`.
    fn matches(dtype: &DataType) -> bool;
    /// The name used in error messages, e.g. `"float32"`.
    fn name() -> &'static str;
}

impl DlpackDtype for f32 {
    fn matches(dtype: &DataType) -> bool {
        dtype == &DataType::F32
    }
    fn name() -> &'static str {
        "32 bit float"
    }
}

impl DlpackDtype for u32 {
    fn matches(dtype: &DataType) -> bool {
        dtype == &DataType::U32
    }
    fn name() -> &'static str {
        "32 bit unsigned integer"
    }
}

pub(super) trait MutTensorExt<T> {
    /// Returns a mutable slice of the tensor data.
    ///
    /// Safety
    /// ======
    /// the caller is required to not to pass a tensor aliasing another live reference.
    unsafe fn as_slice_contiguous_mut(&mut self) -> PyResult<&mut [T]>;
}

impl<T: DlpackDtype> MutTensorExt<T> for SafeManagedTensorVersioned {
    unsafe fn as_slice_contiguous_mut(&mut self) -> PyResult<&mut [T]> {
        // Check data type first because of: https://github.com/SunDoge/dlpark/issues/55
        if !T::matches(self.data_type()) {
            return Err(PyValueError::new_err(format!(
                "tensor must have data type {}, got DLPack data type code {:?}",
                T::name(),
                self.data_type().code
            )));
        } 
        self.as_slice_contiguous::<T>().map_err(dl2py_err)?;
        if self.flags().contains(Flags::READ_ONLY) {
            return Err(pyo3::exceptions::PyValueError::new_err("tensor is read-only"));
        }
        Ok(unsafe {
            std::slice::from_raw_parts_mut(
                self.data_ptr().add(self.byte_offset()).cast(),
                self.num_elements(),
            )
        })
    }
}

pub(super) fn dl2py_err(err: dlpark::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}
