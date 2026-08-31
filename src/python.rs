//! Python bindings for hannoy.
use std::path::PathBuf;
use std::sync::LazyLock;

use crate::{distance, Database, ItemId, Reader, Writer};
use either::Either;
use heed::{RoTxn, RwTxn, WithoutTls};
use numpy::ndarray::{ArrayView1, ArrayViewMut, ArrayViewMut1, Dim, Dimension};
use numpy::{
    prelude::*, Element, PyReadonlyArray, PyReadonlyArray2, PyReadonlyArrayDyn, PyReadwriteArray,
    PyReadwriteArrayDyn,
};
use once_cell::sync::OnceCell;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
static DEFAULT_ENV_SIZE: usize = 1024 * 1024 * 1024; // 1GiB

// LMDB environment.
static ENV: OnceCell<heed::Env<WithoutTls>> = OnceCell::new();
static RW_TXN: LazyLock<Mutex<Option<heed::RwTxn<'static>>>> = LazyLock::new(|| Mutex::new(None));

/// The supported distance metrics in hannoy.
#[gen_stub_pyclass_enum]
#[pyclass(name = "Metric", from_py_object)]
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

#[gen_stub_pymethods]
#[pymethods]
impl PyDistance {
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

/// An LMDB-backed database for vector search.
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

    /// Open a reader for a specific index.
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
        use rand::rngs::StdRng;
        use rand::SeedableRng;

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

    /// Store vectors associated with item IDs in the database, given a 2D numpy array of dtype
    /// `float32`, one row per item.
    fn add_items<'py>(
        &self,
        items: Vec<ItemId>,
        vectors: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<()> {
        let vectors = vectors.as_array();
        let rows = vectors.nrows();
        if items.len() != rows {
            return Err(PyValueError::new_err(format!(
                "add_items requires as many items as the array has rows: got {} items for {rows} rows",
                items.len()
            )));
        }
        items
            .into_iter()
            .zip(vectors.rows())
            .try_for_each(|(item, vector)| self.add_item(item, vector.to_vec()))
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

macro_rules! hnsw_search {
    ($reader:expr, |r| r . $($q:tt)*) => {
        match $reader {
            DynReader::Cosine(reader) => reader . $($q)*,
            DynReader::Euclidean(reader) => reader . $($q)*,
            DynReader::Manhattan(reader) => reader . $($q)*,
            DynReader::BqCosine(reader) => reader . $($q)*,
            DynReader::BqEuclidean(reader) => reader . $($q)*,
            DynReader::BqManhattan(reader) => reader . $($q)*,
            DynReader::Hamming(reader) => reader . $($q)*,
        }
    };
}

type ByArrayResult = Either<Vec<(ItemId, f32)>, Vec<Vec<(ItemId, f32)>>>;

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
        let found = hnsw_search!(&self.dyn_reader, |r| r
            .nns(n)
            .ef_search(ef_search)
            .by_vector(&rtxn, &query))
        .map_err(h2py_err)?;
        Ok(found.into_nns())
    }

    /// Retrieve similar items from the db, given a 1D or 2D numpy array of dtype `float32`.
    ///
    /// A 1D array is treated as a single query vector; a 2D array is treated as one query vector per row.
    ///
    /// If `out` is given, `None` is returned instead of a list of results.
    /// Results are written into `out`, which must be an `(ids, distances)` tuple of writable,
    /// contiguous numpy arrays, `ids` holding `u32`s and `distances` holding `f32`s.
    /// Both are to be shaped `(n,)` for a 1D query or `(rows, n)` for a 2D query.
    /// Rows with fewer than `n` hits are padded with `id` `u32::MAX` and `distance` `f32::INFINITY`.
    #[pyo3(signature = (array, n=10, ef_search=200, out=None))]
    fn by_array<'py>(
        &self,
        array: PyReadonlyArrayDyn<'py, f32>,
        n: usize,
        ef_search: usize,
        out: Option<(PyReadonlyArrayDyn<'py, u32>, PyReadonlyArrayDyn<'py, f32>)>,
    ) -> PyResult<Option<ByArrayResult>> {
        let view = array.as_array();
        let rtxn = &self.rtxn;

        let search = |row: ArrayView1<'_, f32>| {
            let row = row
                .to_slice()
                .ok_or_else(|| PyValueError::new_err("by_array requires a C-contiguous array"))?;
            hnsw_search!(&self.dyn_reader, |r| r.nns(n).ef_search(ef_search).by_vector(rtxn, row))
                .map(|found| found.into_nns())
                .map_err(h2py_err)
        };

        match view.shape() {
            &[_] => {
                let row: ArrayView1<f32> =
                    view.into_dimensionality().expect("We just validated it’s 1D");
                let Some((ids, distances)) = out else {
                    return Ok(Some(Either::Left(search(row)?)));
                };
                let (mut ids, mut distances) =
                    (try_into_readwrite(ids)?, try_into_readwrite(distances)?);
                let ids = validate_out(&mut ids, "out[0] (ids)", &[n])?;
                let distances = validate_out(&mut distances, "out[1] (distances)", &[n])?;
                let found = search(row)?;
                write_row(ids, distances, found, n);
                Ok(None)
            }
            &[rows, _] => {
                let Some((ids, distances)) = out else {
                    return Ok(Some(Either::Right(
                        view.rows().into_iter().map(search).collect::<PyResult<_>>()?,
                    )));
                };
                if n == 0 {
                    return Err(PyValueError::new_err("n must be at least 1 when out is given"));
                }
                let (mut ids, mut distances) =
                    (try_into_readwrite(ids)?, try_into_readwrite(distances)?);
                let mut ids = validate_out(&mut ids, "out[0] (ids)", &[rows, n])?;
                let mut distances = validate_out(&mut distances, "out[1] (distances)", &[rows, n])?;

                for ((query, ids_row), distances_row) in
                    view.rows().into_iter().zip(ids.rows_mut()).zip(distances.rows_mut())
                {
                    write_row(ids_row, distances_row, search(query)?, n);
                }
                Ok(None)
            }
            shape => Err(PyValueError::new_err(format!(
                "by_array requires a 1D or 2D array, got {}D",
                shape.len()
            ))),
        }
    }

    /// Retrieve similar items from the db given an item ID.
    /// Returns `None` if the item is not in the database.
    #[pyo3(signature = (item, n=10, ef_search=200))]
    fn by_item(
        &self,
        item: ItemId,
        n: usize,
        ef_search: usize,
    ) -> PyResult<Option<Vec<(ItemId, f32)>>> {
        let rtxn = &self.rtxn;
        let found =
            hnsw_search!(&self.dyn_reader, |r| r.nns(n).ef_search(ef_search).by_item(&rtxn, item))
                .map_err(h2py_err)?;
        Ok(found.map(|s| s.into_nns()))
    }

    /// Retrieve similar items for each of the batched item IDs.
    // Paralle version of by_items
    // Returns None if the item(s) is not in the database
    #[pyo3(signature = (items, n=10, ef_search=200))]
    fn by_items(
        &self,
        items: Vec<ItemId>,
        n: usize,
        ef_search: usize,
    ) -> PyResult<Vec<Option<Vec<(ItemId, f32)>>>>{
        let env = ENV.get().ok_or_else( || PyRuntimeError::new_err("No environment"))?;
        let found = hnsw_search!(&self.dyn_reader, |r| r.par_by_items(env, &items, n, ef_search)).map_err(h2py_err)?;
        let results: Vec<Option<Vec<(ItemId, f32)>>> = found.into_iter().map(|opt| opt.map(|s| s.into_nns())).collect();
        Ok(results)
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

/// Upgrade a read-only array borrow into a mutable one.
///
/// `PyReadwriteArrayDyn` can't be used as a `#[pymethods]` parameter directly:
/// https://github.com/PyO3/rust-numpy/pull/560
fn try_into_readwrite<'py, T: Element, D: Dimension>(
    array: PyReadonlyArray<'py, T, D>,
) -> PyResult<PyReadwriteArray<'py, T, D>> {
    let bound = (*array).clone();
    drop(array); // drop the read borrow so we can create a rw one
    Ok(bound.try_readwrite()?)
}

/// Validate that `array` is shaped `expected_shape`.
fn validate_out<'a, 'py, T: Element, const N: usize>(
    array: &'a mut PyReadwriteArrayDyn<'py, T>,
    label: &str,
    expected_shape: &[usize; N],
) -> PyResult<ArrayViewMut<'a, T, Dim<[usize; N]>>>
where
    Dim<[usize; N]>: Dimension,
{
    if array.shape() != expected_shape {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape {expected_shape:?}, got {:?}",
            array.shape()
        )));
    }
    Ok(array.as_array_mut().into_dimensionality().expect("shape was just validated"))
}

/// Write one query's `(item, distance)` results into `n`-length `ids`/`distances` rows,
/// padding with `(u32::MAX, f32::INFINITY)` past the number of hits found.
fn write_row(
    mut ids: ArrayViewMut1<'_, u32>,
    mut distances: ArrayViewMut1<'_, f32>,
    found: Vec<(ItemId, f32)>,
    n: usize,
) {
    for i in 0..n {
        let (item, distance) = found.get(i).copied().unwrap_or((ItemId::MAX, f32::INFINITY));
        ids[i] = item;
        distances[i] = distance;
    }
}

/// Python bindings for Hannoy <https://github.com/nnethercott/hannoy>; a KV-backed HNSW
/// implementation in Rust using LMDB <https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database>.
#[pyo3::pymodule]
#[pyo3(name = "hannoy")]
fn hannoy_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistance>()?;
    m.add_class::<PyDatabase>()?;
    m.add_class::<PyWriter>()?;
    m.add_class::<PyReader>()?;
    Ok(())
}

// Define a function to gather stub information.
define_stub_info_gatherer!(stub_info);
#[cfg(test)]
mod test {
    use super::*;
    use numpy::{PyArray2, PyArrayMethods};

    #[test]
    fn write_vectors_py() {
        Python::attach(|py| {
            let dir = tempfile::tempdir().unwrap();
            let distance = PyDistance::Cosine;
            let database = PyDatabase::new(dir.path().to_path_buf(), distance, None, None).unwrap();
            let writer = database.writer(3, 0, 4, 10);
            let input = PyArray2::<f32>::from_vec2(py, &[vec![0.0, 1.0, 2.0], vec![1.0, 0.0, 2.0]])
                .unwrap();
            writer.add_items(vec![0, 1], input.readonly()).unwrap();
            writer.build().unwrap();
            PyDatabase::commit_rw_txn().unwrap();
            let reader = database.reader(0).unwrap();
            assert_eq!(vec![(0, 0.0)], reader.by_vec(vec![0.0, 1.0, 2.0], 1, 10).unwrap());
            assert_eq!(vec![(1, 0.0)], reader.by_vec(vec![1.0, 0.0, 2.0], 1, 10).unwrap());
        });
    }
}
