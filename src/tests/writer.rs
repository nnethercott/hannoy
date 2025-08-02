use super::{create_database, rng};
use crate::distance::{Cosine, Euclidean};
use crate::tests::DatabaseHandle;
use crate::{Reader, Writer};
use insta::assert_snapshot;

const M: usize = 16;
const M0: usize = 32;

// do i add edge-cases for the build, e.g. M = e.ciel() as usize ?

#[test]
fn clear_small_database() {
    let DatabaseHandle { env, database, tempdir: _ } = create_database::<Cosine>();
    let mut wtxn = env.write_txn().unwrap();

    let zero_writer = Writer::new(database, 0, 3);
    zero_writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();
    zero_writer.clear(&mut wtxn).unwrap();
    zero_writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();

    let one_writer = Writer::new(database, 1, 3);
    one_writer.add_item(&mut wtxn, 0, &[1.0, 2.0, 3.0]).unwrap();
    one_writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    let mut wtxn = env.write_txn().unwrap();
    let zero_writer = Writer::new(database, 0, 3);
    zero_writer.clear(&mut wtxn).unwrap();

    let one_reader = Reader::open(&wtxn, 1, database).unwrap();
    assert_eq!(one_reader.item_vector(&wtxn, 0).unwrap().unwrap(), vec![1.0, 2.0, 3.0]);
    wtxn.commit().unwrap();
}

#[test]
fn use_u32_max_minus_one_for_a_vec() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3);
    writer.add_item(&mut wtxn, u32::MAX - 1, &[0.0, 1.0, 2.0]).unwrap();

    writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[4294967294]>, distance: "euclidean", entry_points: [4294967294], max_level: 0 }
    Version: Version { major: 0, minor: 0, patch: 2 }
    Links 4294967294: Links(Links { links: RoaringBitmap<[]> })
    Item 4294967294: Item(Leaf { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 1.0000, 2.0000] })
    "#);
}

#[test]
fn use_u32_max_for_a_vec() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3);
    writer.add_item(&mut wtxn, u32::MAX, &[0.0, 1.0, 2.0]).unwrap();

    writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[4294967295]>, distance: "euclidean", entry_points: [4294967295], max_level: 0 }
    Version: Version { major: 0, minor: 0, patch: 2 }
    Links 4294967295: Links(Links { links: RoaringBitmap<[]> })
    Item 4294967295: Item(Leaf { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 1.0000, 2.0000] })
    "#);
}

#[test]
fn write_one_vector() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3);
    writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

    writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, distance: "euclidean", entry_points: [0], max_level: 0 }
    Version: Version { major: 0, minor: 0, patch: 2 }
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Item 0: Item(Leaf { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 1.0000, 2.0000] })
    "#);
}
