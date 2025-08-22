use hannoy::{distances::Cosine, Database, Reader, Result, Writer};
use heed::EnvOpenOptions;
use rand::{rngs::StdRng, SeedableRng};

fn main() -> Result<()> {
    const DIM: usize = 3;
    let vecs: Vec<[f32; DIM]> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(1024 * 1024 * 1024 * 1) // 1GiB
            .open("./")
    }
    .unwrap();

    let mut wtxn = env.write_txn().unwrap();
    let db: Database<Cosine> = env.create_database(&mut wtxn, None)?;
    let writer: Writer<Cosine> = Writer::new(db, 0, DIM);

    // insert into lmdb
    writer.add_item(&mut wtxn, 0, &vecs[0])?;
    writer.add_item(&mut wtxn, 1, &vecs[1])?;
    writer.add_item(&mut wtxn, 2, &vecs[2])?;

    // ...and build hnsw
    let mut rng = StdRng::seed_from_u64(42);

    let mut builder = writer.builder(&mut rng);
    builder.ef_construction(100).build::<16,32>(&mut wtxn)?;
    wtxn.commit()?;

    // search hnsw using a new lmdb read transaction
    let rtxn = env.read_txn()?;
    let reader = Reader::<Cosine>::open(&rtxn, 0, db)?;

    let query = vec![0.0, 1.0, 0.0];
    let nns = reader.nns(1).ef_search(10).by_vector(&rtxn, &query)?;

    dbg!("{:?}", &nns);
    Ok(())
}
