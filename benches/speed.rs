use std::env;
use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::Write;

use hannoy::Reader;
use hannoy::{distances::Cosine, Database, Writer};
use heed::{Env, EnvOpenOptions, RwTxn};
use hnsw_rs;
use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::prelude::DistCosine;
use rand::thread_rng;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tempfile::tempdir;

static M: usize = 16;
static M0: usize = 32;

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn gen_vecs<const DIM: usize>(size: usize) -> Vec<[f32; DIM]> {
    let mut rng = rng();

    (0..size)
        .into_iter()
        .map(|_| {
            let mut arr = [0.0; DIM];
            rng.fill(&mut arr);
            arr
        })
        .collect()
}

fn setup_lmdb() -> Env {
    let temp_dir = tempdir().unwrap();
    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(1024 * 1024 * 1024 * 2) // 2GiB
            .open(temp_dir)
    }
    .unwrap();
    env
}

fn create_lmdb_and_fill_with_vecs<'a, const DIM: usize>(
    env: &'a Env,
    vecs: &'a Vec<[f32; DIM]>,
) -> hannoy::Result<(Writer<Cosine>, RwTxn<'a>, Database<Cosine>)> {
    let mut wtxn = env.write_txn().unwrap();

    let db: Database<Cosine> = env.create_database(&mut wtxn, None)?;
    let writer: Writer<Cosine> = Writer::new(db, 0, DIM);

    // insert random vectors
    for (id, vec) in vecs.iter().enumerate() {
        writer.add_item(&mut wtxn, id as u32, vec)?;
    }

    Ok((writer, wtxn, db))
}

#[divan::bench(
        consts = [512, 768, 1536],
    )]
fn search_hannoy<const DIM: usize>(bencher: divan::Bencher) {
    // first build a vector db
    let env = setup_lmdb();
    let vecs = gen_vecs::<DIM>(100_000);
    let (writer, mut wtxn, db) = create_lmdb_and_fill_with_vecs::<DIM>(&env, &vecs).unwrap();
    let mut rng = rng();
    let mut builder = writer.builder(&mut rng);
    builder.ef_construction(32).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    let rtxn = env.read_txn().unwrap();

    // https://www.kernel.org/doc/Documentation/sysctl/vm.txt
    fn drop_caches() -> std::io::Result<()> {
        let mut f = OpenOptions::new().write(true).open("/proc/sys/vm/drop_caches")?;
        f.write_all(b"3")?;
        Ok(())
    }

    bencher
        .with_inputs(|| {
            drop_caches().unwrap();
            let mut query = [f32::default(); DIM];
            thread_rng().fill(&mut query);
            let reader = Reader::<Cosine>::open(&rtxn, 0, db).unwrap();
            (reader, query)
        })
        .bench_local_values(|(reader, query)| {
            black_box(reader.nns(10).ef_search(10).by_vector(&rtxn, &query).unwrap());
        });
}

#[divan::bench(
        consts = [512, 768, 1536],
    )]
fn search_hannoy_in_cache<const DIM: usize>(bencher: divan::Bencher) {
    // first build a vector db
    let env = setup_lmdb();
    let vecs = gen_vecs::<DIM>(100_000);
    let (writer, mut wtxn, db) = create_lmdb_and_fill_with_vecs::<DIM>(&env, &vecs).unwrap();
    let mut rng = rng();
    let mut builder = writer.builder(&mut rng);
    builder.ef_construction(32).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    let rtxn = env.read_txn().unwrap();

    // prefetch whole graph
    env::set_var("HANNOY_READER_PREFETCH_MEMORY", format!("{}", 1024 * 1024 * 1024));
    bencher
        .with_inputs(|| {
            let mut query = [f32::default(); DIM];
            thread_rng().fill(&mut query);
            let reader = Reader::<Cosine>::open(&rtxn, 0, db).unwrap();
            (reader, query)
        })
        .bench_local_values(|(reader, query)| {
            black_box(reader.nns(10).ef_search(10).by_vector(&rtxn, &query).unwrap());
        });
}

// hnsw-rs
#[divan::bench(
    consts = [512, 768, 1536],
)]
fn search_hnsw_rs<const DIM: usize>(bencher: divan::Bencher) {
    let nb_elem = 100_000;
    let vecs = gen_vecs::<DIM>(nb_elem);
    let max_nb_connection = M0;
    let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
    let ef_c = 32;

    // allocating network
    let mut hnsw =
        Hnsw::<f32, DistCosine>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistCosine {});
    hnsw.set_extend_candidates(false);

    // parallel insertion of train data
    let vecs: Vec<_> = vecs.into_iter().map(|x| x.to_vec()).collect();
    let data_for_par_insertion: Vec<_> = vecs.iter().enumerate().map(|(i, x)| (x, i)).collect();
    hnsw.parallel_insert(&data_for_par_insertion[..]);

    // set to searching
    hnsw.set_searching_mode(true);

    bencher
        .with_inputs(|| {
            let mut query = [f32::default(); DIM];
            thread_rng().fill(&mut query);
            query
        })
        .bench_local_values(|query| {
            let knbn = 10;
            let ef_c = 10;
            black_box(hnsw.search(&query, knbn, ef_c));
        });
}

fn main() {
    divan::main();
}
