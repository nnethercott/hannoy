//! Python bindings for hannoy.
use heed::{RoTxn, RwTxn, WithoutTls};
use once_cell::sync::OnceCell;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use pyo3::{
    exceptions::{PyIOError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyType,
};
use pyo3_stub_gen::{
    define_stub_info_gatherer,
    derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods},
};
use std::{path::PathBuf, str::FromStr, sync::LazyLock};

use crate::{distance, Database, ItemId, Reader, Writer};
static DEFAULT_ENV_SIZE: usize = 1024 * 1024 * 1024; // 1GiB

// LMDB environment.
static ENV: OnceCell<heed::Env<WithoutTls>> = OnceCell::new();
static RW_TXN: LazyLock<Mutex<Option<heed::RwTxn<'static>>>> = LazyLock::new(|| Mutex::new(None));

/// Supported distance metrics in hannoy.
#[gen_stub_pyclass_enum]
#[pyclass(name = "Metric")]
#[derive(Clone)]
pub(super) enum PyDistance {
    #[pyo3(name = "COSINE")]
    Cosine,
    #[pyo3(name = "EUCLIDEAN")]
    Euclidean,
    #[pyo3(name = "MANHATTAN")]
    Manhattan,
    #[pyo3(name = "BQ_COSINE")]
    BqCosine,
    #[pyo3(name = "BQ_EUCLIDEAN")]
    BqEuclidean,
    #[pyo3(name = "BQ_MANHATTAN")]
    BqManhattan,
    #[pyo3(name = "HAMMING")]
    Hamming,
}

impl FromStr for PyDistance {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cosine" => Ok(Self::Cosine),
            "euclidean" => Ok(Self::Euclidean),
            "manhattan" => Ok(Self::Manhattan),
            "bq_cosine" => Ok(Self::BqCosine),
            "bq_euclidean" => Ok(Self::BqEuclidean),
            "bq_manhattan" => Ok(Self::BqManhattan),
            "hamming" => Ok(Self::Hamming),
            _ => Err("unknown metric"),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDistance {
    #[new]
    fn new(variant: &str) -> PyResult<Self> {
        Self::from_str(variant).map_err(PyValueError::new_err)
    }

    fn __str__(&self) -> String {
        match self {
            PyDistance::Cosine => "cosine".into(),
            PyDistance::Euclidean => "euclidean".into(),
            PyDistance::Manhattan => "manhattan".into(),
            PyDistance::BqCosine => "bq_cosine".into(),
            PyDistance::BqEuclidean => "bq_euclidean".into(),
            PyDistance::BqManhattan => "bq_manhattan".into(),
            PyDistance::Hamming => "hamming".into(),
        }
    }
}

enum DynDatabase {
    Cosine(Database<distance::Cosine>),
    Euclidean(Database<distance::Euclidean>),
    Manhattan(Database<distance::Manhattan>),
    BqCosine(Database<distance::BinaryQuantizedCosine>),
    BqEuclidean(Database<distance::BinaryQuantizedEuclidean>),
    BqManhattan(Database<distance::BinaryQuantizedManhattan>),
    Hamming(Database<distance::Hamming>),
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
            PyDistance::Manhattan => Ok(DynDatabase::Manhattan(env.create_database(wtxn, name)?)),
            PyDistance::BqCosine => Ok(DynDatabase::BqCosine(env.create_database(wtxn, name)?)),
            PyDistance::BqEuclidean => {
                Ok(DynDatabase::BqEuclidean(env.create_database(wtxn, name)?))
            }
            PyDistance::BqManhattan => {
                Ok(DynDatabase::BqManhattan(env.create_database(wtxn, name)?))
            }
            PyDistance::Hamming => Ok(DynDatabase::Hamming(env.create_database(wtxn, name)?)),
        }
    }
}

/// An LMDB-backed vector database for vector search.
#[gen_stub_pyclass]
#[pyclass(name = "Database")]
pub(super) struct PyDatabase(DynDatabase);

#[gen_stub_pymethods]
#[pymethods]
impl PyDatabase {
    #[new]
    #[pyo3(signature = (path, distance=PyDistance::Euclidean, name=None, env_size=None))]
    fn new(
        path: PathBuf,
        distance: PyDistance,
        name: Option<&str>,
        env_size: Option<usize>,
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
    #[pyo3(signature = (dimensions, index=0, m=16, ef=96))]
    fn writer(&self, dimensions: usize, index: u16, m: usize, ef: usize) -> PyWriter {
        let opts = BuildOptions { ef, m, m0: 2 * m };

        match self.0 {
            DynDatabase::Cosine(db) => {
                PyWriter { dyn_writer: DynWriter::Cosine(Writer::new(db, index, dimensions)), opts }
            }
            DynDatabase::Euclidean(db) => PyWriter {
                dyn_writer: DynWriter::Euclidean(Writer::new(db, index, dimensions)),
                opts,
            },
            DynDatabase::Manhattan(db) => PyWriter {
                dyn_writer: DynWriter::Manhattan(Writer::new(db, index, dimensions)),
                opts,
            },
            DynDatabase::BqCosine(db) => PyWriter {
                dyn_writer: DynWriter::BqCosine(Writer::new(db, index, dimensions)),
                opts,
            },
            DynDatabase::BqEuclidean(db) => PyWriter {
                dyn_writer: DynWriter::BqEuclidean(Writer::new(db, index, dimensions)),
                opts,
            },
            DynDatabase::BqManhattan(db) => PyWriter {
                dyn_writer: DynWriter::BqManhattan(Writer::new(db, index, dimensions)),
                opts,
            },
            DynDatabase::Hamming(db) => PyWriter {
                dyn_writer: DynWriter::Hamming(Writer::new(db, index, dimensions)),
                opts,
            },
        }
    }

    /// Get a reader for a specific index and dimensions
    #[pyo3(signature = (index = 0))]
    fn reader(&self, index: u16) -> PyResult<PyReader> {
        let rtxn = get_ro_txn()?;

        let reader = match self.0 {
            DynDatabase::Cosine(database) => {
                let reader = Reader::open(&rtxn, index, database).map_err(h2py_err)?;
                let dyn_reader = DynReader::Cosine(reader);
                PyReader { dyn_reader, rtxn }
            }
            DynDatabase::Euclidean(database) => {
                let reader = Reader::open(&rtxn, index, database).map_err(h2py_err)?;
                let dyn_reader = DynReader::Euclidean(reader);
                PyReader { dyn_reader, rtxn }
            }
            DynDatabase::Manhattan(database) => {
                let reader = Reader::open(&rtxn, index, database).map_err(h2py_err)?;
                let dyn_reader = DynReader::Manhattan(reader);
                PyReader { dyn_reader, rtxn }
            }
            DynDatabase::BqCosine(database) => {
                let reader = Reader::open(&rtxn, index, database).map_err(h2py_err)?;
                let dyn_reader = DynReader::BqCosine(reader);
                PyReader { dyn_reader, rtxn }
            }
            DynDatabase::BqEuclidean(database) => {
                let reader = Reader::open(&rtxn, index, database).map_err(h2py_err)?;
                let dyn_reader = DynReader::BqEuclidean(reader);
                PyReader { dyn_reader, rtxn }
            }
            DynDatabase::BqManhattan(database) => {
                let reader = Reader::open(&rtxn, index, database).map_err(h2py_err)?;
                let dyn_reader = DynReader::BqManhattan(reader);
                PyReader { dyn_reader, rtxn }
            }
            DynDatabase::Hamming(database) => {
                let reader = Reader::open(&rtxn, index, database).map_err(h2py_err)?;
                let dyn_reader = DynReader::Hamming(reader);
                PyReader { dyn_reader, rtxn }
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
    Manhattan(Writer<distance::Manhattan>),
    BqCosine(Writer<distance::BinaryQuantizedCosine>),
    BqEuclidean(Writer<distance::BinaryQuantizedEuclidean>),
    BqManhattan(Writer<distance::BinaryQuantizedManhattan>),
    Hamming(Writer<distance::Hamming>),
}

#[derive(Clone)]
struct BuildOptions {
    pub ef: usize,
    pub m: usize,
    pub m0: usize,
}

/// A struct for configuring the HNSW build and performing transactional insertions/deletions from
/// LMDB.
///
/// Example:
/// ```python
/// from hannoy import Database, Metric
///
/// db = Database("./", Metric.Cosine)
///
/// with db.writer(2, m=4, ef=10) as writer:
///     writer.add_item(0, [1.0, 0.0])
///     writer.add_item(1, [0.0, 1.0])
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "Writer")]
pub(super) struct PyWriter {
    dyn_writer: DynWriter,
    opts: BuildOptions,
}

impl PyWriter {
    fn build(&self) -> PyResult<()> {
        use rand::{rngs::StdRng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let mut wtxn = get_rw_txn()?;

        let BuildOptions { ef, m, m0 } = self.opts;

        // a helper macro to auto generating some matches
        macro_rules! match_table {
            ($w:expr => $(($M:literal, $M0:literal)),* $(,)?) => {
                match (m, m0) {
                    $(
                        ($M, $M0) => $w.builder(&mut rng).ef_construction(ef).build::<$M, $M0>(&mut wtxn),
                    )*
                    _ => panic!("not supported: m = {}, m0 = {}", m, m0),
                }.map_err(h2py_err)?
            };
        }
        // the real macro
        macro_rules! hnsw_build {
            ($w:expr) => {{
                match_table! {$w => (4, 8), (8, 16), (12, 24), (16, 32), (24, 48), (32, 64)}
            }};
        }

        match &self.dyn_writer {
            DynWriter::Cosine(writer) => hnsw_build!(writer),
            DynWriter::Euclidean(writer) => hnsw_build!(writer),
            DynWriter::Manhattan(writer) => hnsw_build!(writer),
            DynWriter::BqCosine(writer) => hnsw_build!(writer),
            DynWriter::BqEuclidean(writer) => hnsw_build!(writer),
            DynWriter::BqManhattan(writer) => hnsw_build!(writer),
            DynWriter::Hamming(writer) => hnsw_build!(writer),
        };
        Ok(())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWriter {
    #[pyo3(signature = ())] // make pyo3_stub_gen ignore “slf”
    fn __enter__(slf: Bound<Self>) -> Bound<Self> {
        slf
    }

    fn __exit__<'py>(
        &self,
        _exc_type: Option<Bound<'py, PyType>>,
        _exc_value: Option<Bound<'py, PyAny /*PyBaseException*/>>,
        _traceback: Option<Bound<'py, PyAny /*PyTraceback*/>>,
    ) -> PyResult<()> {
        self.build()?;
        PyDatabase::commit_rw_txn()?;
        Ok(())
    }

    /// Store a vector associated with an item ID in the database.
    fn add_item(&self, item: ItemId, vector: Vec<f32>) -> PyResult<()> {
        let mut wtxn = get_rw_txn()?;
        match &self.dyn_writer {
            DynWriter::Cosine(writer) => {
                writer.add_item(&mut wtxn, item, &vector).map_err(h2py_err)?
            }
            DynWriter::Euclidean(writer) => {
                writer.add_item(&mut wtxn, item, &vector).map_err(h2py_err)?
            }
            DynWriter::Manhattan(writer) => {
                writer.add_item(&mut wtxn, item, &vector).map_err(h2py_err)?
            }
            DynWriter::BqCosine(writer) => {
                writer.add_item(&mut wtxn, item, &vector).map_err(h2py_err)?
            }
            DynWriter::BqEuclidean(writer) => {
                writer.add_item(&mut wtxn, item, &vector).map_err(h2py_err)?
            }
            DynWriter::BqManhattan(writer) => {
                writer.add_item(&mut wtxn, item, &vector).map_err(h2py_err)?
            }
            DynWriter::Hamming(writer) => {
                writer.add_item(&mut wtxn, item, &vector).map_err(h2py_err)?
            }
        }
        Ok(())
    }
}

enum DynReader {
    Cosine(Reader<distance::Cosine>),
    Euclidean(Reader<distance::Euclidean>),
    Manhattan(Reader<distance::Manhattan>),
    BqCosine(Reader<distance::BinaryQuantizedCosine>),
    BqEuclidean(Reader<distance::BinaryQuantizedEuclidean>),
    BqManhattan(Reader<distance::BinaryQuantizedManhattan>),
    Hamming(Reader<distance::Hamming>),
}

/// A thread-local Database reader holding its own `RoTxn`. It is safe to spawn multiple readers in
/// different threads.
///
/// Example:
/// ```python
/// db = hannoy.Database("./")
///
/// reader = db.reader()
/// reader.by_vec([1.0, 0.0], n = 1)
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "Reader", unsendable)]
struct PyReader {
    dyn_reader: DynReader,
    rtxn: RoTxn<'static, WithoutTls>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyReader {
    /// Retrieve similar items from the db given a query.
    #[pyo3(signature = (query, n=10, ef_search=200))]
    fn by_vec(&self, query: Vec<f32>, n: usize, ef_search: usize) -> PyResult<Vec<(ItemId, f32)>> {
        let rtxn = &self.rtxn;

        macro_rules! hnsw_search {
            ($read:expr, $q:expr) => {
                $read.nns(n).ef_search(ef_search).by_vector(&rtxn, $q).map_err(h2py_err)
            };
        }

        let neighbours = match &self.dyn_reader {
            DynReader::Cosine(reader) => hnsw_search!(reader, &query)?,
            DynReader::Euclidean(reader) => hnsw_search!(reader, &query)?,
            DynReader::Manhattan(reader) => hnsw_search!(reader, &query)?,
            DynReader::BqCosine(reader) => hnsw_search!(reader, &query)?,
            DynReader::BqEuclidean(reader) => hnsw_search!(reader, &query)?,
            DynReader::BqManhattan(reader) => hnsw_search!(reader, &query)?,
            DynReader::Hamming(reader) => hnsw_search!(reader, &query)?,
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

fn get_ro_txn() -> PyResult<RoTxn<'static, WithoutTls>> {
    let env = ENV.get().ok_or_else(|| PyRuntimeError::new_err("No environment"))?;
    let rtxn = env.read_txn().map_err(h2py_err)?;
    Ok(rtxn)
}

#[pyo3::pymodule]
#[pyo3(name = "hannoy")]
fn hannoy_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistance>()?;
    m.add_class::<PyDatabase>()?;
    m.add_class::<PyWriter>()?;
    Ok(())
}

// Define a function to gather stub information.
define_stub_info_gatherer!(stub_info);
