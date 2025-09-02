use hannoy::{distances::Cosine, Database, Writer};
use heed::{Env, EnvOpenOptions, RwTxn};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tempfile::tempdir;

static M: usize = 16;
static M0: usize = 32;

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

// hnsw build and search benchmarks
mod hnsw {
    use hannoy::Reader;
    use rand::thread_rng;

    use super::*;

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

    fn create_db_and_fill_with_vecs<const DIM: usize>(
        env: &Env,
        size: usize,
    ) -> hannoy::Result<(Writer<Cosine>, RwTxn, Database<Cosine>)> {
        let mut wtxn = env.write_txn().unwrap();

        let db: Database<Cosine> = env.create_database(&mut wtxn, None)?;
        let writer: Writer<Cosine> = Writer::new(db, 0, DIM);
        let mut rng = rng();

        // insert random vectors
        for vec_id in 0..size {
            let mut vec = [0.0; DIM];
            rng.fill(&mut vec);
            writer.add_item(&mut wtxn, vec_id as u32, &vec)?;
        }

        Ok((writer, wtxn, db))
    }

    // time hnsw build
    #[divan::bench(
        consts = [512, 768, 1536],
        max_time = 60.0,
    )]
    fn build_hnsw<const DIM: usize>(bencher: divan::Bencher) {
        let env = setup_lmdb();

        bencher
            .with_inputs(|| create_db_and_fill_with_vecs::<DIM>(&env, 5000).unwrap())
            .bench_local_values(|(writer, mut wtxn, _)| {
                let mut rng = rng();
                let mut builder = writer.builder(&mut rng);
                builder.ef_construction(32).build::<M, M0>(&mut wtxn).unwrap();
            });
    }

    // time hnsw search
    #[divan::bench(
        consts = [512, 768, 1536],
        sample_count = 100,
    )]
    fn search_hnsw<const DIM: usize>(bencher: divan::Bencher) {
        // first build a vector db
        let env = setup_lmdb();
        let (writer, mut wtxn, db) = create_db_and_fill_with_vecs::<DIM>(&env, 50000).unwrap();
        let mut rng = rng();
        let mut builder = writer.builder(&mut rng);
        builder.ef_construction(32).build::<M, M0>(&mut wtxn).unwrap();
        wtxn.commit().unwrap();

        // reader should have a lifetime to this
        let rtxn = env.read_txn().unwrap();

        bencher
            .with_inputs(|| {
                let mut query = [f32::default(); DIM];
                thread_rng().fill(&mut query);
                // Reader::open can incur some system calls that mess with profiling
                let reader = Reader::<Cosine>::open(&rtxn, 0, db).unwrap();
                (reader, query)
            })
            .bench_local_values(|(reader, query)| {
                reader.nns(10).by_vector(&rtxn, &query).unwrap();
            });
    }
}

fn main() {
    divan::main();
}
