use pyo3::{exceptions::PyValueError, PyResult};
use pyo3_dlpack::{DLDataType, PyTensor};

/// A Rust type corresponding to one DLPack dtype, usable with [`AsTypedSlice`].
pub(super) trait DlpackDtype {
    /// Whether a tensor's DLPack dtype is `Self`.
    fn matches(dtype: DLDataType) -> bool;
    /// The name used in error messages, e.g. `"float32"`.
    fn name() -> &'static str;
}

impl DlpackDtype for f32 {
    fn matches(dtype: DLDataType) -> bool {
        dtype.is_f32()
    }
    fn name() -> &'static str {
        "float32"
    }
}

impl DlpackDtype for u32 {
    fn matches(dtype: DLDataType) -> bool {
        dtype.is_u32()
    }
    fn name() -> &'static str {
        "uint32"
    }
}

fn check_cpu_contiguous(tensor: &PyTensor) -> PyResult<()> {
    if !tensor.device().is_cpu() {
        return Err(PyValueError::new_err("only CPU tensors are supported"));
    }
    if !tensor.is_contiguous() {
        return Err(PyValueError::new_err("only contiguous tensors are supported"));
    }
    Ok(())
}

/// Extension trait for borrowing a `PyTensor`'s data as a typed slice, checking that the
/// tensor's DLPack dtype is `T`. Implemented once for every `T: DlpackDtype`, so each dtype only
/// has to supply [`DlpackDtype::matches`]/[`DlpackDtype::name`].
pub(super) trait AsTypedSlice<T> {
    fn as_slice(&self) -> PyResult<&[T]>;
    /// As [`AsTypedSlice::as_slice`], but the tensor must also be writable.
    ///
    /// SAFETY: the caller is required to not to pass a tensor aliasing another live reference.
    unsafe fn as_mut_slice(&mut self) -> PyResult<&mut [T]>;
}

impl<T: DlpackDtype> AsTypedSlice<T> for PyTensor {
    fn as_slice(&self) -> PyResult<&[T]> {
        check_cpu_contiguous(self)?;
        if !T::matches(self.dtype()) {
            return Err(PyValueError::new_err(format!("only {} tensors are supported", T::name())));
        }
        assert!(self.nbytes() < isize::MAX as usize);
        // SAFETY:
        // 1: `data` is non-null and contains `self.nbytes()` properly aligned values (`is_cpu()`)
        // 2: `data` contains consecutive initialized `T` values (`T::matches()` & `is_contiguous()`)
        // 3: `data` is not being mutated for `self`’s lifetime (`PyTensor` guarantee)
        // 4: we assert the `isize::MAX` invariant
        Ok(unsafe { std::slice::from_raw_parts(self.data_ptr() as *const T, self.numel()) })
    }

    unsafe fn as_mut_slice(&mut self) -> PyResult<&mut [T]> {
        check_cpu_contiguous(self)?;
        if !T::matches(self.dtype()) {
            return Err(PyValueError::new_err(format!("only {} tensors are supported", T::name())));
        }
        if self.is_read_only() {
            return Err(PyValueError::new_err("out tensors must be writable"));
        }
        assert!(self.nbytes() < isize::MAX as usize);
        // SAFETY: as `as_slice`, with the replaced condition:
        // 3: the caller is required to not to pass a tensor aliasing another live reference.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.data_ptr() as *mut T, self.numel()) })
    }
}
