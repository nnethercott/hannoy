use heed::{RoTxn, RwTxn, WithoutTls};
use once_cell::sync::OnceCell;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use pyo3::{
    exceptions::{PyIOError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyType,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use std::{path::PathBuf, str::FromStr, sync::LazyLock};

use crate::{distance, Database, ItemId, Reader, Writer};
static DEFAULT_ENV_SIZE: usize = 1024 * 1024 * 1024 * 1; // 1GiB

// LMDB environment.
static ENV: OnceCell<heed::Env<WithoutTls>> = OnceCell::new();
static RW_TXN: LazyLock<Mutex<Option<heed::RwTxn<'static>>>> = LazyLock::new(|| Mutex::new(None));

#[gen_stub_pyclass_enum]
#[pyclass(name = "Metric")]
#[derive(Clone)]
pub(super) enum PyDistance {
    #[pyo3(name = "COSINE")]
    Cosine,
    #[pyo3(name = "EUCLIDEAN")]
    Euclidean,
}

impl FromStr for PyDistance {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cosine" => Ok(Self::Cosine),
            "euclidean" => Ok(Self::Euclidean),
            _ => Err("unknown metric"),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDistance {
    #[new]
    fn new(variant: &str) -> PyResult<Self> {
        Self::from_str(variant).map_err(|e| PyValueError::new_err(e))
    }

    fn __str__(&self) -> String {
        match self {
            PyDistance::Cosine => "cosine".into(),
            PyDistance::Euclidean => "euclidean".into(),
        }
    }
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
                match_table! {$w =>(3, 6), (4, 8), (5, 10), (6, 12), (7, 14), (8, 16), (9, 18), (10, 20),
                (11, 22), (12, 24), (13, 26), (14, 28), (15, 30), (16, 32), (17, 34), (18, 36), (19, 38),
                (20, 40), (21, 42), (22, 44), (23, 46), (24, 48), (25, 50), (26, 52), (27, 54), (28, 56),
                (29, 58), (30, 60), (31, 62), (32, 64), (33, 66), (34, 68), (35, 70), (36, 72), (37, 74),
                (38, 76), (39, 78), (40, 80), (41, 82), (42, 84), (43, 86), (44, 88), (45, 90), (46, 92),
                (47, 94), (48, 96), (49, 98), (50, 100), (51, 102), (52, 104), (53, 106), (54, 108), (55, 110),
                (56, 112), (57, 114), (58, 116), (59, 118), (60, 120), (61, 122), (62, 124), (63, 126),
                (64, 128), (65, 130), (66, 132), (67, 134), (68, 136), (69, 138), (70, 140), (71, 142),
                (72, 144), (73, 146), (74, 148), (75, 150), (76, 152), (77, 154), (78, 156), (79, 158),
                (80, 160), (81, 162), (82, 164), (83, 166), (84, 168), (85, 170), (86, 172), (87, 174),
                (88, 176), (89, 178), (90, 180), (91, 182), (92, 184), (93, 186), (94, 188), (95, 190),
                (96, 192), (97, 194), (98, 196), (99, 198), (100, 200)}
            }};
        }

        match &self.dyn_writer {
            DynWriter::Cosine(writer) => hnsw_build!(writer),
            DynWriter::Euclidean(writer) => hnsw_build!(writer),
        };
        Ok(())
    }
}

#[pymethods]
#[gen_stub_pymethods]
impl PyWriter {
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
        }
        Ok(())
    }
}

enum DynReader {
    Cosine(Reader<distance::Cosine>),
    Euclidean(Reader<distance::Euclidean>),
}

/// A thread-local Database reader holding its own `RoTxn`.
#[gen_stub_pyclass]
#[pyclass(name = "Reader", unsendable)]
struct PyReader {
    dyn_reader: DynReader,
    rtxn: RoTxn<'static, WithoutTls>,
}

#[pymethods]
impl PyReader {
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

fn get_ro_txn<'a>() -> PyResult<RoTxn<'static, WithoutTls>> {
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
