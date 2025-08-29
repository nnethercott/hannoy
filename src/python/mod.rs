use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use pyo3::{types::PyModule, Bound, PyResult};

// LMDB environment.
static ENV: OnceCell<heed::Env> = OnceCell::new();

mod reader;
mod writer;

use writer::{PyDatabase, PyDistance, PyWriter};

#[pyo3::pymodule]
#[pyo3(name = "hannoy")]
fn hannoy_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistance>()?;
    m.add_class::<PyDatabase>()?;
    m.add_class::<PyWriter>()?;
    Ok(())
}
