use heed::{RoTxn, RwTxn, WithoutTls};
use once_cell::sync::OnceCell;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use pyo3::{
    exceptions::{PyIOError, PyRuntimeError},
    prelude::*,
    types::PyType,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use std::{path::PathBuf, sync::LazyLock};

use crate::{distance, Database, ItemId, Reader, Writer};
static DEFAULT_ENV_SIZE: usize = 1024 * 1024 * 1024 * 1; // 1GiB

// LMDB environment.
static ENV: OnceCell<heed::Env<WithoutTls>> = OnceCell::new();
static RW_TXN: LazyLock<Mutex<Option<heed::RwTxn<'static>>>> = LazyLock::new(|| Mutex::new(None));
// FIXME: find better way to do this; since the variable is static we need it to be Sync ? but the
// heed::RoTxn is !Sync (it's Send) so we need to wrap it in something. 
// But i guess we'd also like to invalidate the rtxn if a new wtxn is created (check heed docs) so
// we need a notion of interior mutability
static RO_TXN: LazyLock<Mutex<Option<heed::RoTxn<'static, WithoutTls>>>> =
    LazyLock::new(|| Mutex::new(None));

#[gen_stub_pyclass_enum]
#[pyclass(name = "Metric")]
#[derive(Clone)]
pub(super) enum PyDistance {
    Cosine,
    Euclidean,
}

enum DynDatabase {
    Cosine(Database<distance::Cosine>),
    Euclidean(Database<distance::Euclidean>),
}
impl DynDatabase {
    pub fn new(
        env: &heed::Env<WithoutTls>,
        wtxn: &mut RwTxn,
        name: Option<&str>,
        distance: PyDistance,
    ) -> heed::Result<Self> {
        match distance {
            PyDistance::Cosine => Ok(DynDatabase::Cosine(env.create_database(wtxn, name)?)),
            PyDistance::Euclidean => Ok(DynDatabase::Euclidean(env.create_database(wtxn, name)?)),
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "Database")]
pub(super) struct PyDatabase(DynDatabase);

#[gen_stub_pymethods]
#[pymethods]
impl PyDatabase {
    #[new]
    #[pyo3(signature = (path, name=None, env_size=None, distance=PyDistance::Euclidean))]
    fn new(
        path: PathBuf,
        name: Option<&str>,
        env_size: Option<usize>,
        distance: PyDistance,
    ) -> PyResult<PyDatabase> {
        let size = env_size.unwrap_or(DEFAULT_ENV_SIZE);
        let env = ENV
            .get_or_try_init(|| unsafe {
                heed::EnvOpenOptions::new().read_txn_without_tls().map_size(size).open(path)
            })
            .map_err(h2py_err)?;
        let mut wtxn = get_rw_txn()?;
        let db = DynDatabase::new(env, &mut wtxn, name, distance).map_err(h2py_err)?;
        Ok(PyDatabase(db))
    }

    /// Get a writer for a specific index and dimensions.
    fn writer(&self, index: u16, dimensions: usize) -> PyWriter {
        let opts = BuildOptions::default();

        match self.0 {
            DynDatabase::Cosine(db) => {
                PyWriter(DynWriter::Cosine(Writer::new(db, index, dimensions)), opts)
            }
            DynDatabase::Euclidean(db) => {
                PyWriter(DynWriter::Euclidean(Writer::new(db, index, dimensions)), opts)
            }
        }
    }

    /// Get a reader for a specific index and dimensions
    fn reader(&self, index: u16) -> PyResult<PyReader> {
        let opts = SearchOpts::default();
        let rtxn = get_ro_txn()?;

        let reader = match self.0 {
            DynDatabase::Cosine(database) => {
                let reader = Reader::open(&rtxn, index, database).map_err(h2py_err)?;
                PyReader(DynReader::Cosine(reader), opts)
            }
            DynDatabase::Euclidean(database) => {
                let reader = Reader::open(&rtxn, index, database).map_err(h2py_err)?;
                PyReader(DynReader::Euclidean(reader), opts)
            }
        };
        Ok(reader)
    }

    #[staticmethod]
    fn commit_rw_txn() -> PyResult<bool> {
        if let Some(wtxn) = RW_TXN.lock().take() {
            wtxn.commit().map_err(h2py_err)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    #[staticmethod]
    fn abort_rw_txn() -> bool {
        if let Some(wtxn) = RW_TXN.lock().take() {
            wtxn.abort();
            true
        } else {
            false
        }
    }
}

enum DynWriter {
    Cosine(Writer<distance::Cosine>),
    Euclidean(Writer<distance::Euclidean>),
}

#[derive(Clone)]
struct BuildOptions {
    pub ef: usize,
    pub m: usize,
    pub m0: usize,
}

impl Default for BuildOptions {
    fn default() -> Self {
        BuildOptions { ef: 100, m: 16, m0: 32 }
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "Writer")]
pub(super) struct PyWriter(DynWriter, BuildOptions);

impl PyWriter {
    fn build(&self) -> PyResult<()> {
        use rand::{rngs::StdRng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let mut wtxn = get_rw_txn()?;

        // NOTE: maybe proc macro this to inject all combos up to a point...
        let BuildOptions { ef, m, m0 } = self.1;
        macro_rules! hnsw_build {
            ($w:expr) => {
                match (m, m0) {
                    (4, 8) => $w.builder(&mut rng).ef_construction(ef).build::<4, 8>(&mut wtxn),
                    (8, 16) => $w.builder(&mut rng).ef_construction(ef).build::<8, 16>(&mut wtxn),
                    (16, 32) => $w.builder(&mut rng).ef_construction(ef).build::<16, 32>(&mut wtxn),
                    (24, 48) => $w.builder(&mut rng).ef_construction(ef).build::<32, 64>(&mut wtxn),
                    _ => panic!("not supported"),
                }
                .map_err(h2py_err)
            };
        }
        match &self.0 {
            DynWriter::Cosine(writer) => hnsw_build!(writer)?,
            DynWriter::Euclidean(writer) => hnsw_build!(writer)?,
        };
        Ok(())
    }
}

#[pymethods]
#[gen_stub_pymethods]
impl PyWriter {
    #[setter]
    fn ef_construction(&mut self, ef: usize) -> PyResult<()> {
        self.1.ef = ef;
        Ok(())
    }

    #[pyo3(signature = ())] // make pyo3_stub_gen ignore “slf”
    fn __enter__<'py>(slf: Bound<'py, Self>) -> Bound<'py, Self> {
        slf
    }

    fn __exit__<'py>(
        &self,
        _exc_type: Option<Bound<'py, PyType>>,
        _exc_value: Option<Bound<'py, PyAny /*PyBaseException*/>>,
        _traceback: Option<Bound<'py, PyAny /*PyTraceback*/>>,
    ) -> PyResult<()> {
        self.build()?;

        // Commit the txn; abort and rethrow on failure.
        PyDatabase::commit_rw_txn().map_err(|e| {
            PyDatabase::abort_rw_txn();
            e
        })?;

        Ok(())
    }

    /// Store a vector associated with an item ID in the database.
    fn add_item(&self, item: ItemId, vector: Vec<f32>) -> PyResult<()> {
        let mut wtxn = get_rw_txn()?;
        match &self.0 {
            DynWriter::Cosine(writer) => {
                writer.add_item(&mut wtxn, item, &vector).map_err(h2py_err)?
            }
            DynWriter::Euclidean(writer) => {
                writer.add_item(&mut wtxn, item, &vector).map_err(h2py_err)?
            }
        }
        Ok(())
    }
}

enum DynReader {
    Cosine(Reader<distance::Cosine>),
    Euclidean(Reader<distance::Euclidean>),
}

#[derive(Clone, Default)]
struct SearchOpts {
    ef_search: usize,
    count: usize,
}

/// A thread-local Database reader.
#[gen_stub_pyclass]
#[pyclass(name = "Reader", unsendable)]
struct PyReader(DynReader, SearchOpts);

#[pymethods]
impl PyReader {
    fn get(&self, query: Vec<f32>) -> PyResult<Vec<(ItemId, f32)>> {
        let rtxn = get_ro_txn()?;
        let SearchOpts { ef_search, count } = self.1;

        macro_rules! hnsw_search {
            ($read:expr, $q:expr) => {
                $read.nns(count).ef_search(ef_search).by_vector(&rtxn, $q).map_err(h2py_err)
            };
        }

        let neighbours = match &self.0 {
            DynReader::Cosine(reader) => hnsw_search!(reader, &query)?,
            DynReader::Euclidean(reader) => hnsw_search!(reader, &query)?,
        };
        Ok(neighbours)
    }
}

fn h2py_err<E: Into<crate::error::Error>>(e: E) -> PyErr {
    match e.into() {
        crate::Error::Heed(heed::Error::Io(e)) | crate::Error::Io(e) => {
            PyIOError::new_err(e.to_string())
        }
        e => PyRuntimeError::new_err(e.to_string()),
    }
}

fn get_rw_txn<'a>() -> PyResult<MappedMutexGuard<'a, RwTxn<'static>>> {
    let mut maybe_txn = RW_TXN.lock();
    if maybe_txn.is_none() {
        let env = ENV.get().ok_or_else(|| PyRuntimeError::new_err("No environment"))?;
        let wtxn = env.write_txn().map_err(h2py_err)?;
        *maybe_txn = Some(wtxn);
    }
    Ok(MutexGuard::map(maybe_txn, |txn| txn.as_mut().unwrap()))
}

fn get_ro_txn<'a>() -> PyResult<MappedMutexGuard<'a, RoTxn<'static, WithoutTls>>> {
    let mut maybe_txn = RO_TXN.lock();
    if maybe_txn.is_none() {
        let env = ENV.get().ok_or_else(|| PyRuntimeError::new_err("No environment"))?;
        let wtxn = env.read_txn().map_err(h2py_err)?;
        *maybe_txn = Some(wtxn);
    }
    Ok(MutexGuard::map(maybe_txn, |txn| txn.as_mut().unwrap()))
}

#[pyo3::pymodule]
#[pyo3(name = "hannoy")]
fn hannoy_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistance>()?;
    m.add_class::<PyDatabase>()?;
    m.add_class::<PyWriter>()?;
    Ok(())
}
