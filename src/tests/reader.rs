use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    distance::BinaryQuantizedCosine,
    tests::{create_database, DatabaseHandle},
    Reader, Writer,
};

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
