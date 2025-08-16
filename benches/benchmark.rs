use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use hannoy::{distances::Cosine, Database, Writer};
use heed::{Env, EnvOpenOptions, RwTxn};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tempfile::tempdir;

static M: usize = 16;
static M0: usize = 32;

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
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

fn index_and_search_10k(c: &mut Criterion) {
    const DIM: usize = 512;
    let env = setup_lmdb();

    fn create_db_and_fill_with_vecs(env: &Env) -> hannoy::Result<(Writer<Cosine>, RwTxn)> {
        let mut wtxn = env.write_txn().unwrap();

        let db: Database<Cosine> = env.create_database(&mut wtxn, None)?;
        let writer: Writer<Cosine> = Writer::new(db, 0, DIM);
        let mut rng = rng();

        // insert 1k random vectors
        for vec_id in 0..10000 {
            let mut vec = [f32::default(); DIM];
            rng.fill(&mut vec);
            writer.add_item(&mut wtxn, vec_id, &vec)?;
        }

        Ok((writer, wtxn))
    }

    // bench writer
    let mut group = c.benchmark_group("writer");
    group
        .sample_size(100)
        .warm_up_time(Duration::from_secs(10))
        .measurement_time(Duration::from_secs(100));

    group.bench_function("hnsw build 10k", move |b| {
        b.iter(|| {
            let (writer, mut wtxn) = create_db_and_fill_with_vecs(&env).unwrap();
            let mut rng = rng();
            let mut builder = writer.builder(&mut rng);
            builder.ef_construction(32).build::<M, M0>(&mut wtxn).unwrap();
        });
    });
}

criterion_group!(benches, index_and_search_10k);
criterion_main!(benches);
