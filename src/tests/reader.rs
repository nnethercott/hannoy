use rand::{distributions::Uniform, rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

use crate::{
    distance::{BinaryQuantizedCosine, Cosine},
    tests::{create_database, rng, DatabaseHandle},
    Reader, Writer,
};

const M: usize = 16;
const M0: usize = 32;

// Minimal reproducer for issue #78
// <https://github.com/nnethercott/hannoy/issues/78>
#[test]
fn quantized_iter_has_right_dimensions() {
    let DatabaseHandle { env, database, tempdir: _ } = create_database::<BinaryQuantizedCosine>();
    let mut wtxn = env.write_txn().unwrap();
    // use a prime number of dims
    const DIM: usize = 1063;
    let writer = Writer::new(database, 0, DIM);

    let mut rng = StdRng::seed_from_u64(42);

    let mut vec = [0f32; DIM];
    rng.fill(&mut vec);
    writer.add_item(&mut wtxn, 0, &vec).unwrap();
    writer.builder(&mut rng).build::<16, 32>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    let rtxn = env.read_txn().unwrap();
    let reader = Reader::open(&rtxn, 0, database).unwrap();
    let mut cursor = reader.iter(&rtxn).unwrap();
    let (_, new_vec) = cursor.next().unwrap().unwrap();

    assert!(new_vec.len() == DIM);
}

#[test]
fn unreachable_items() {
    const DIM: usize = 1025;

    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    let dir = tempfile::tempdir().unwrap();
    let env = unsafe { heed::EnvOpenOptions::new().map_size(200 * 1024 * 1024).open(dir.path()) }
        .unwrap();
    let mut wtxn = env.write_txn().unwrap();

    let database: crate::Database<Cosine> = env.create_database(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();

    let mut rng = rng();
    let mut wtxn = env.write_txn().unwrap();

    let mut db_indexes: Vec<u16> = (1..2).collect();
    db_indexes.shuffle(&mut rng);

    const HOW_MANY: usize = 1000;

    for index in db_indexes.iter().copied() {
        let writer = Writer::new(database, index, DIM);

        // We're going to write 10k vectors per index
        let unif = Uniform::new(-1.0, 1.0);
        for i in 0..HOW_MANY {
            let vector: [f32; DIM] = std::array::from_fn(|_| rng.sample(unif));
            writer.add_item(&mut wtxn, i as u32, &vector).unwrap();
        }

        // build with smallest number of links possible
        writer.builder(&mut rng).build::<3, 3>(&mut wtxn).unwrap();

        // Check that all items were written correctly
        let reader = crate::Reader::<Cosine>::open(&wtxn, index, database).unwrap();
        assert_eq!(reader.item_ids().len(), HOW_MANY as u64);
        assert!((0..HOW_MANY as u32).all(|i| reader.contains_item(&wtxn, i).unwrap()));
        let found = reader.nns(HOW_MANY).ef_search(HOW_MANY).by_vector(&wtxn, &[0.0; DIM]).unwrap();
        assert_eq!(found.len(), HOW_MANY);
    }
    wtxn.commit().unwrap();
}
