use heed::types::DecodeIgnore;
use proptest::proptest;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng, SeedableRng};
use roaring::RoaringBitmap;

use super::{create_database, rng};
use crate::distance::{Cosine, Euclidean};
use crate::key::{KeyCodec, Prefix, PrefixCodec};
use crate::reader::get_item;
use crate::tests::DatabaseHandle;
use crate::{Reader, Writer};

const M: usize = 3;
const M0: usize = 3;

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
fn delete_all_entry_points_and_build() {
    let handle = create_database::<Cosine>();
    let mut wtxn = handle.env.write_txn().unwrap();

    let writer = Writer::new(handle.database, 0, 3);
    const ITEM_VEC: &'static [[f32; 3]] =
        &[[0.0, 1.0, 2.0], [1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [1.0, 0.0, 2.0]];

    writer.del_item(&mut wtxn, 0).unwrap();

    for (item, vec) in ITEM_VEC.iter().enumerate() {
        writer.add_item(&mut wtxn, item as u32, vec).unwrap();
    }
    writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();
    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0, 1, 2, 3]>, distance: "cosine", entry_points: [0, 2, 3], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[1, 3]> })
    Links 0: Links(Links { links: RoaringBitmap<[3]> })
    Links 1: Links(Links { links: RoaringBitmap<[0, 2]> })
    Links 2: Links(Links { links: RoaringBitmap<[1, 3]> })
    Links 2: Links(Links { links: RoaringBitmap<[3]> })
    Links 3: Links(Links { links: RoaringBitmap<[0, 2]> })
    Links 3: Links(Links { links: RoaringBitmap<[0, 2]> })
    Item 0: Item(Item { header: NodeHeaderCosine { norm: "2.2361" }, vector: [0.0000, 1.0000, 2.0000] })
    Item 1: Item(Item { header: NodeHeaderCosine { norm: "2.2361" }, vector: [1.0000, 2.0000, 0.0000] })
    Item 2: Item(Item { header: NodeHeaderCosine { norm: "2.2361" }, vector: [2.0000, 1.0000, 0.0000] })
    Item 3: Item(Item { header: NodeHeaderCosine { norm: "2.2361" }, vector: [1.0000, 0.0000, 2.0000] })
    "#);

    // NOW delete all the entry points !
    let mut wtxn = handle.env.write_txn().unwrap();
    writer.del_item(&mut wtxn, 0).unwrap();
    writer.del_item(&mut wtxn, 2).unwrap();
    writer.del_item(&mut wtxn, 3).unwrap();
    writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();
}

// Minimal reproducer for issue #52
// <https://github.com/nnethercott/hannoy/issues/52>
#[test]
fn delete_every_items_and_build() {
    const DIMENSIONS: usize = 3;

    fn vector_store_for_embedder(embedder_id: u8, store_id: u8) -> u16 {
        let embedder_id = (embedder_id as u16) << 8;
        embedder_id | (store_id as u16)
    }

    const ITEM_VEC: &'static [[f32; 3]] =
        &[[0.0, 1.0, 2.0], [1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [1.0, 0.0, 2.0]];

    let DatabaseHandle { env, database, tempdir: _ } = create_database::<Cosine>();
    let emb0_store0 = vector_store_for_embedder(0, 0);
    let emb0_store1 = vector_store_for_embedder(0, 1);
    let emb1_store0 = vector_store_for_embedder(1, 0);
    let emb1_store1 = vector_store_for_embedder(1, 1);
    let emb2_store0 = vector_store_for_embedder(2, 0);
    let emb2_store1 = vector_store_for_embedder(2, 1);

    let mut wtxn = env.write_txn().unwrap();

    // Deleting item_id=1 in all stores in embedder_index=0
    for index in [emb0_store0, emb0_store1] {
        let writer = Writer::new(database, index, DIMENSIONS);
        writer.del_item(&mut wtxn, 1).unwrap();
    }

    // Adding item_id=1 to all stores in embedder_index=0
    for index in [emb0_store0, emb0_store1] {
        let writer = Writer::new(database, index, DIMENSIONS);
        writer.add_item(&mut wtxn, 1, &ITEM_VEC[1]).unwrap();
    }

    // Adding item_id=0 in store_id=0 and embedder_index=0
    // Adding item_id=2 in store_id=0 and embedder_index=0
    {
        let writer = Writer::new(database, emb0_store0, DIMENSIONS);
        writer.add_item(&mut wtxn, 0, &ITEM_VEC[0]).unwrap();
        writer.add_item(&mut wtxn, 2, &ITEM_VEC[2]).unwrap();
        writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();
    }

    // Adding item_id=2 in store_id=1 and embedder_index=0
    {
        let writer = Writer::new(database, emb0_store1, DIMENSIONS);
        writer.add_item(&mut wtxn, 2, &ITEM_VEC[2]).unwrap();
        writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();
    }

    // Adding item_id=3 in store_id=0 and embedder_index=0
    {
        let writer = Writer::new(database, emb0_store0, DIMENSIONS);
        writer.add_item(&mut wtxn, 3, &ITEM_VEC[3]).unwrap();
        writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();
    }

    // Adding item_id=3 in store_id=1 and embedder_index=0
    {
        let writer = Writer::new(database, emb0_store1, DIMENSIONS);
        writer.add_item(&mut wtxn, 3, &ITEM_VEC[3]).unwrap();
        writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();
    }

    // Build and quantize embedder_index=0
    for index in [emb0_store0, emb0_store1] {
        let writer = Writer::new(database, index, DIMENSIONS);
        writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();
    }

    // Deleting item_id=0 in all stores in embedder_index=1
    // Deleting item_id=0 in all stores in embedder_index=2
    // Deleting item_id=1 in all stores in embedder_index=1
    // Deleting item_id=1 in all stores in embedder_index=2
    // Deleting item_id=2 in all stores in embedder_index=1
    // Deleting item_id=2 in all stores in embedder_index=2
    // Deleting item_id=3 in all stores in embedder_index=1
    // Deleting item_id=3 in all stores in embedder_index=2
    for item in [0, 1, 2, 3] {
        for index in [emb1_store0, emb1_store1, emb2_store0, emb2_store1] {
            let writer = Writer::new(database, index, DIMENSIONS);
            writer.del_item(&mut wtxn, item).unwrap();
        }
    }

    // Adding item_id=0 in store_id=0 and embedder_index=1
    // Adding item_id=1 in store_id=0 and embedder_index=1
    // Adding item_id=2 in store_id=0 and embedder_index=1
    for item in [0, 1, 2] {
        let writer = Writer::new(database, emb1_store0, DIMENSIONS);
        writer.add_item(&mut wtxn, item, &ITEM_VEC[item as usize]).unwrap();
    }

    // Adding item_id=2 in store_id=1 and embedder_index=1
    {
        let writer = Writer::new(database, emb1_store1, DIMENSIONS);
        writer.add_item(&mut wtxn, 2, &ITEM_VEC[2]).unwrap();
    }

    // Adding item_id=3 in store_id=0 and embedder_index=1
    {
        let writer = Writer::new(database, emb1_store0, DIMENSIONS);
        writer.add_item(&mut wtxn, 3, &ITEM_VEC[3]).unwrap();
    }

    // Adding item_id=3 in store_id=1 and embedder_index=1
    {
        let writer = Writer::new(database, emb1_store1, DIMENSIONS);
        writer.add_item(&mut wtxn, 3, &ITEM_VEC[3]).unwrap();
    }

    // Adding item_id=0 in store_id=0 and embedder_index=2
    // Adding item_id=1 in store_id=0 and embedder_index=2
    // Adding item_id=2 in store_id=0 and embedder_index=2
    // Adding item_id=3 in store_id=0 and embedder_index=2
    for item in [0, 1, 2, 3] {
        let writer = Writer::new(database, emb2_store0, DIMENSIONS);
        writer.add_item(&mut wtxn, item, &ITEM_VEC[item as usize]).unwrap();
    }

    // Build and quantize embedder_index=1
    // Build and quantize embedder_index=2
    for index in [emb1_store0, emb1_store1, emb2_store0, emb2_store1] {
        let writer = Writer::new(database, index, DIMENSIONS);
        writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();
    }

    // Clearing all items from embedder_index=1
    for index in [emb1_store0, emb1_store1] {
        let writer = Writer::new(database, index, DIMENSIONS);
        writer.clear(&mut wtxn).unwrap();
    }

    // Deleting item_id=0 in store_id=0 and embedder_index=0
    // Deleting item_id=2 in store_id=0 and embedder_index=0
    // Deleting item_id=3 in store_id=0 and embedder_index=0
    // Build and quantize embedder_index=0
    for index in [emb0_store0, emb0_store1] {
        let writer = Writer::new(database, index, DIMENSIONS);
        writer.del_item(&mut wtxn, 0).unwrap();
        writer.del_item(&mut wtxn, 2).unwrap();
        writer.del_item(&mut wtxn, 3).unwrap();
        writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();
    }
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
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[4294967294]>, distance: "euclidean", entry_points: [4294967294], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 4294967294: Links(Links { links: RoaringBitmap<[]> })
    Links 4294967294: Links(Links { links: RoaringBitmap<[]> })
    Item 4294967294: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 1.0000, 2.0000] })
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
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[4294967295]>, distance: "euclidean", entry_points: [4294967295], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 4294967295: Links(Links { links: RoaringBitmap<[]> })
    Links 4294967295: Links(Links { links: RoaringBitmap<[]> })
    Item 4294967295: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 1.0000, 2.0000] })
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
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, distance: "euclidean", entry_points: [0], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Item 0: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 1.0000, 2.0000] })
    "#);
}

#[test]
fn write_and_update_lot_of_random_points_with_snapshot() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 30);
    let mut rng = rng();
    for id in 0..100 {
        let vector: [f32; 30] = std::array::from_fn(|_| rng.gen());
        writer.add_item(&mut wtxn, id, &vector).unwrap();
    }

    writer.builder(&mut rng).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();
    insta::assert_snapshot!(handle);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 30);
    for id in (0..100).step_by(2) {
        let vector: [f32; 30] = std::array::from_fn(|_| rng.gen());
        writer.add_item(&mut wtxn, id, &vector).unwrap();
    }
    writer.builder(&mut rng).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle);
}

#[test]
fn write_multiple_indexes() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();

    for i in 0..5 {
        let writer = Writer::new(handle.database, i, 3);
        writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();
        writer.builder(&mut rng()).build::<M, M0>(&mut wtxn).unwrap();
    }
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, distance: "euclidean", entry_points: [0], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Item 0: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 1.0000, 2.0000] })
    ==================
    Dumping index 1
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, distance: "euclidean", entry_points: [0], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Item 0: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 1.0000, 2.0000] })
    ==================
    Dumping index 2
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, distance: "euclidean", entry_points: [0], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Item 0: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 1.0000, 2.0000] })
    ==================
    Dumping index 3
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, distance: "euclidean", entry_points: [0], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Item 0: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 1.0000, 2.0000] })
    ==================
    Dumping index 4
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, distance: "euclidean", entry_points: [0], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Item 0: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 1.0000, 2.0000] })
    "#);
}

#[test]
fn write_random_vectors_to_random_indexes() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();

    let mut indexes: Vec<u16> = (0..10).collect();
    indexes.shuffle(&mut rng);

    for index in indexes {
        let writer = Writer::new(handle.database, index, 10);

        // We're going to write 10 vectors per index
        for i in 0..10 {
            let vector: [f32; 10] = std::array::from_fn(|_| rng.gen());
            writer.add_item(&mut wtxn, i, &vector).unwrap();
        }
        writer.builder(&mut rng).build::<M, M0>(&mut wtxn).unwrap();
    }
    wtxn.commit().unwrap();
}

#[test]
fn convert_from_arroy_to_hannoy() {
    // let handle = create_database::<Euclidean>();
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    let dir = tempfile::tempdir().unwrap();
    let env = unsafe { heed::EnvOpenOptions::new().map_size(200 * 1024 * 1024).open(dir.path()) }
        .unwrap();
    let mut wtxn = env.write_txn().unwrap();
    let database: arroy::Database<arroy::distances::Cosine> =
        env.create_database(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();

    let mut rng = rng();
    let mut wtxn = env.write_txn().unwrap();

    let mut db_indexes: Vec<u16> = (0..10).collect();
    db_indexes.shuffle(&mut rng);

    for index in db_indexes.iter().copied() {
        let writer = arroy::Writer::new(database, index, 1024);

        // We're going to write 100 vectors per index
        for i in 0..100 {
            let vector: [f32; 1024] = std::array::from_fn(|_| rng.gen());
            writer.add_item(&mut wtxn, i, &vector).unwrap();
        }
        writer.builder(&mut rng).build(&mut wtxn).unwrap();
    }
    wtxn.commit().unwrap();

    // Now it's time to convert the indexes

    let mut wtxn = env.write_txn().unwrap();
    let rtxn = env.read_txn().unwrap();
    let database: crate::Database<Cosine> = env.open_database(&mut wtxn, None).unwrap().unwrap();

    db_indexes.shuffle(&mut rng);

    for index in db_indexes {
        let pre_commit_arroy_reader =
            arroy::Reader::<arroy::distances::Cosine>::open(&rtxn, index, database.remap_types())
                .unwrap();

        let writer = Writer::new(database, index, pre_commit_arroy_reader.dimensions());
        writer.builder(&mut rng).prepare_arroy_conversion(&mut wtxn).unwrap();
        assert!(writer.need_build(&mut wtxn).unwrap());
        writer.builder(&mut rng).build::<16, 32>(&mut wtxn).unwrap();

        for result in pre_commit_arroy_reader.iter(&rtxn).unwrap() {
            let (item_id, vector) = result.unwrap();
            let reader = Reader::open(&wtxn, index, database).unwrap();
            assert_eq!(reader.item_vector(&wtxn, item_id).unwrap().as_deref(), Some(&vector[..]));
            let mut found = reader.nns(1).by_vector(&wtxn, &vector).unwrap();
            dbg!(&found);
            let (found_item_id, found_distance) = found.pop().unwrap();
            assert_eq!(found_item_id, item_id);
            approx::assert_abs_diff_eq!(found_distance, 0.0);
        }
    }
}

#[test]
fn overwrite_one_item_incremental() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    for i in 0..6 {
        writer.add_item(&mut wtxn, i, &[i as f32, 0.]).unwrap();
    }
    writer.builder(&mut rng).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, distance: "euclidean", entry_points: [0, 2, 3], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[1, 2]> })
    Links 0: Links(Links { links: RoaringBitmap<[2]> })
    Links 1: Links(Links { links: RoaringBitmap<[0, 2]> })
    Links 2: Links(Links { links: RoaringBitmap<[0, 1, 3]> })
    Links 2: Links(Links { links: RoaringBitmap<[0, 3]> })
    Links 3: Links(Links { links: RoaringBitmap<[2, 4]> })
    Links 3: Links(Links { links: RoaringBitmap<[2]> })
    Links 4: Links(Links { links: RoaringBitmap<[3, 5]> })
    Links 5: Links(Links { links: RoaringBitmap<[4]> })
    Item 0: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 0.0000] })
    Item 1: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [1.0000, 0.0000] })
    Item 2: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [2.0000, 0.0000] })
    Item 3: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [3.0000, 0.0000] })
    Item 4: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [4.0000, 0.0000] })
    Item 5: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [5.0000, 0.0000] })
    "#);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);
    writer.add_item(&mut wtxn, 3, &[6., 0.]).unwrap();

    writer.builder(&mut rng).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, distance: "euclidean", entry_points: [0, 2, 3], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[1]> })
    Links 0: Links(Links { links: RoaringBitmap<[2]> })
    Links 1: Links(Links { links: RoaringBitmap<[0, 2]> })
    Links 2: Links(Links { links: RoaringBitmap<[1, 4]> })
    Links 2: Links(Links { links: RoaringBitmap<[0, 3]> })
    Links 3: Links(Links { links: RoaringBitmap<[5]> })
    Links 3: Links(Links { links: RoaringBitmap<[2]> })
    Links 4: Links(Links { links: RoaringBitmap<[2, 5]> })
    Links 5: Links(Links { links: RoaringBitmap<[3, 4]> })
    Item 0: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 0.0000] })
    Item 1: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [1.0000, 0.0000] })
    Item 2: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [2.0000, 0.0000] })
    Item 3: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [6.0000, 0.0000] })
    Item 4: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [4.0000, 0.0000] })
    Item 5: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [5.0000, 0.0000] })
    "#);
}

// NOTE: this will fail while our deletions aren't properly handled
#[test]
fn delete_one_item_in_a_one_item_db() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.builder(&mut rng).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0]>, distance: "euclidean", entry_points: [0], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Item 0: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 0.0000] })
    "#);

    // new transaction for the delete
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.del_item(&mut wtxn, 0).unwrap();
    writer.builder(&mut rng).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[]>, distance: "euclidean", entry_points: [], max_level: 0 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    "#);

    let rtxn = handle.env.read_txn().unwrap();
    let one_reader = Reader::open(&rtxn, 0, handle.database).unwrap();
    assert!(one_reader.item_vector(&rtxn, 0).unwrap().is_none());
}

#[test]
fn delete_document_in_an_empty_index_74() {
    // See https://github.com/meilisearch/arroy/issues/74
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();

    let writer = Writer::new(handle.database, 0, 2);
    writer.del_item(&mut wtxn, 0).unwrap();
    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.builder(&mut rng).build::<M, M0>(&mut wtxn).unwrap();

    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0]>, distance: "euclidean", entry_points: [0], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Item 0: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 0.0000] })
    "#);

    let mut wtxn = handle.env.write_txn().unwrap();

    let writer1 = Writer::new(handle.database, 0, 2);
    writer1.del_item(&mut wtxn, 0).unwrap();

    let writer2 = Writer::new(handle.database, 1, 2);
    writer2.del_item(&mut wtxn, 0).unwrap();

    writer1.builder(&mut rng).build::<M, M0>(&mut wtxn).unwrap();
    writer2.builder(&mut rng).build::<M, M0>(&mut wtxn).unwrap();

    let reader = Reader::open(&wtxn, 1, handle.database).unwrap();
    let ret = reader.nns(10).by_vector(&wtxn, &[0., 0.]).unwrap();
    insta::assert_debug_snapshot!(ret, @"[]");

    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[]>, distance: "euclidean", entry_points: [], max_level: 0 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    ==================
    Dumping index 1
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[]>, distance: "euclidean", entry_points: [], max_level: 0 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    "#);

    let rtxn = handle.env.read_txn().unwrap();
    let reader = Reader::open(&rtxn, 1, handle.database).unwrap();
    let ret = reader.nns(10).by_vector(&rtxn, &[0., 0.]).unwrap();
    insta::assert_debug_snapshot!(ret, @"[]");
}

#[test]
fn delete_one_item_in_a_single_document_database() {
    let handle = create_database::<Cosine>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    // first, insert a bunch of elements
    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.builder(&mut rng).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0]>, distance: "cosine", entry_points: [0], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Links 0: Links(Links { links: RoaringBitmap<[]> })
    Item 0: Item(Item { header: NodeHeaderCosine { norm: "0.0000" }, vector: [0.0000, 0.0000] })
    "#);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.del_item(&mut wtxn, 0).unwrap();

    writer.builder(&mut rng).build::<M, M0>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[]>, distance: "cosine", entry_points: [], max_level: 0 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    "#);
}

#[test]
fn delete_one_item() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    // first, insert a bunch of elements
    for i in 0..6 {
        writer.add_item(&mut wtxn, i, &[i as f32, 0.]).unwrap();
    }
    writer.builder(&mut rng).build::<3, 3>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, distance: "euclidean", entry_points: [0, 2, 3], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[1, 2]> })
    Links 0: Links(Links { links: RoaringBitmap<[2]> })
    Links 1: Links(Links { links: RoaringBitmap<[0, 2]> })
    Links 2: Links(Links { links: RoaringBitmap<[0, 1, 3]> })
    Links 2: Links(Links { links: RoaringBitmap<[0, 3]> })
    Links 3: Links(Links { links: RoaringBitmap<[2, 4]> })
    Links 3: Links(Links { links: RoaringBitmap<[2]> })
    Links 4: Links(Links { links: RoaringBitmap<[3, 5]> })
    Links 5: Links(Links { links: RoaringBitmap<[4]> })
    Item 0: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 0.0000] })
    Item 1: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [1.0000, 0.0000] })
    Item 2: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [2.0000, 0.0000] })
    Item 3: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [3.0000, 0.0000] })
    Item 4: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [4.0000, 0.0000] })
    Item 5: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [5.0000, 0.0000] })
    "#);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.del_item(&mut wtxn, 3).unwrap();

    writer.builder(&mut rng).build::<3, 3>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 4, 5]>, distance: "euclidean", entry_points: [0, 1, 2], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[1]> })
    Links 0: Links(Links { links: RoaringBitmap<[1]> })
    Links 1: Links(Links { links: RoaringBitmap<[0, 2]> })
    Links 1: Links(Links { links: RoaringBitmap<[0, 2]> })
    Links 2: Links(Links { links: RoaringBitmap<[1, 2, 4]> })
    Links 2: Links(Links { links: RoaringBitmap<[1, 2]> })
    Links 4: Links(Links { links: RoaringBitmap<[2, 4, 5]> })
    Links 5: Links(Links { links: RoaringBitmap<[4]> })
    Item 0: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 0.0000] })
    Item 1: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [1.0000, 0.0000] })
    Item 2: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [2.0000, 0.0000] })
    Item 4: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [4.0000, 0.0000] })
    Item 5: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [5.0000, 0.0000] })
    "#);

    // delete another one
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.del_item(&mut wtxn, 1).unwrap();

    writer.builder(&mut rng).build::<3, 3>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 2, 4, 5]>, distance: "euclidean", entry_points: [0, 2, 4], max_level: 1 }
    Version: Version { major: 0, minor: 0, patch: 3 }
    Links 0: Links(Links { links: RoaringBitmap<[0, 2]> })
    Links 0: Links(Links { links: RoaringBitmap<[0, 2]> })
    Links 2: Links(Links { links: RoaringBitmap<[0, 2, 4]> })
    Links 2: Links(Links { links: RoaringBitmap<[0, 2, 4]> })
    Links 4: Links(Links { links: RoaringBitmap<[2, 4, 5]> })
    Links 4: Links(Links { links: RoaringBitmap<[2]> })
    Links 5: Links(Links { links: RoaringBitmap<[4]> })
    Item 0: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [0.0000, 0.0000] })
    Item 2: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [2.0000, 0.0000] })
    Item 4: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [4.0000, 0.0000] })
    Item 5: Item(Item { header: NodeHeaderEuclidean { bias: "0.0000" }, vector: [5.0000, 0.0000] })
    "#);
}

#[test]
fn delete_one_item_no_snapshots() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    // first, insert a bunch of elements
    for i in 0..6 {
        writer.add_item(&mut wtxn, i, &[i as f32, 0.]).unwrap();
    }
    writer.builder(&mut rng).build::<3, 3>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.del_item(&mut wtxn, 3).unwrap();

    writer.builder(&mut rng).build::<3, 3>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    // delete another one
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.del_item(&mut wtxn, 1).unwrap();

    writer.builder(&mut rng).build::<3, 3>(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    // verify neither of items are in db nor their links
    let rtxn = handle.env.read_txn().unwrap();
    assert!(get_item(handle.database, 0, &rtxn, 3).unwrap().is_none());
    assert!(get_item(handle.database, 0, &rtxn, 1).unwrap().is_none());

    let links_iter = handle
        .database
        .remap_key_type::<PrefixCodec>()
        .prefix_iter(&rtxn, &Prefix::links(0))
        .unwrap()
        .remap_types::<KeyCodec, DecodeIgnore>();

    let mut keys_of_links = RoaringBitmap::new();
    for res in links_iter {
        let (k, _) = res.unwrap();
        keys_of_links.insert(k.node.item);
    }
    assert!(!keys_of_links.contains(3));
    assert!(!keys_of_links.contains(1));
}

proptest! {
    #[test]
    fn fuzz_writer(n in 1..=1000u32, dim in 1..=100usize) {
        let handle = create_database::<Euclidean>();
        let mut rng = StdRng::from_seed(thread_rng().gen());
        let mut wtxn = handle.env.write_txn().unwrap();

        let writer = Writer::new(handle.database, 0, dim);

        for i in 1..=n {
            let vector: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
            writer.add_item(&mut wtxn, i, &vector).unwrap();
        }
        writer.builder(&mut rng).build::<M, M0>(&mut wtxn).unwrap();
    }
}
