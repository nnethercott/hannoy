use crate::{
    distance::Cosine,
    key::{KeyCodec, Prefix, PrefixCodec},
    node::{Links, Node, NodeCodec},
    node_id::NodeMode,
    tests::{create_database_indices_with_items, DatabaseHandle},
    Database, Reader, Writer,
};
use arbitrary::{Arbitrary, Unstructured};
use heed::RoTxn;
use rand::{self, distributions::Uniform, rngs::StdRng, thread_rng, Rng, SeedableRng};
use roaring::RoaringBitmap;
use tracing::info;

#[derive(Arbitrary, Debug)]
enum WriteOp<const M: usize> {
    Add(u32),
    Del(u32),
}

fn assert_all_readable<const DIM: usize>(rtxn: &RoTxn, database: Database<Cosine>) {
    info!("READING");
    let reader = Reader::<Cosine>::open(&rtxn, 0, database).unwrap();
    let n = reader.item_ids().len() as usize;
    let found = reader.nns(n).ef_search(n).by_vector(&rtxn, &[0.0; DIM]).unwrap().into_nns();
    assert_eq!(&RoaringBitmap::from_iter(found.into_iter().map(|(id, _)| id)), reader.item_ids())
}

fn assert_deleted_items_are_gone(
    rtxn: &RoTxn,
    database: Database<Cosine>,
    deleted: &RoaringBitmap,
) {
    // assert the reader cannot find any deleted vectors
    let reader = Reader::<Cosine>::open(&rtxn, 0, database).unwrap();
    let item_intersection = deleted & reader.item_ids();
    assert!(item_intersection.is_empty(), "{:?} should be deleted!", item_intersection);

    // now iter over ALL links and assert no connection exists to a deleted item
    let mut cursor = database
        .remap_types::<PrefixCodec, NodeCodec<Cosine>>()
        .prefix_iter(rtxn, &Prefix::links(0))
        .unwrap()
        .remap_key_type::<KeyCodec>();

    while let Some((key, node)) = cursor.next().transpose().unwrap() {
        assert!(
            !deleted.contains(key.node.item),
            "the item and its data should be deleted!\n{:?}",
            &key
        );

        match key.node.mode {
            NodeMode::Links => {
                if let Node::Links(Links { links: links_bitmap }) = node {
                    let link_intersection = deleted & links_bitmap.as_ref();
                    assert!(
                        link_intersection.is_empty(),
                        "LINKS VIOLATION: {:?} should be empty",
                        link_intersection
                    );
                }
            }
            _ => continue,
        }
    }
}

#[test]
fn random_read_writes() {
    let seed: u64 = rand::random();
    let mut rng = StdRng::seed_from_u64(seed);

    const DIM: usize = 32;
    const NUMEL: usize = 1000;
    const M: usize = 16;
    const M0: usize = 768;

    // util for generating new vectors on the fly
    fn gen_vec() -> [f32; DIM] {
        let unif = Uniform::new(-1.0, 1.0);
        std::array::from_fn(|_| thread_rng().sample(unif))
    }

    let DatabaseHandle { env, database, tempdir: _ } =
        create_database_indices_with_items::<Cosine, DIM, M, M0, _>(0..1, NUMEL, &mut rng);

    let mut deleted = RoaringBitmap::new();

    for _ in 0..100 {
        let rtxn = env.read_txn().unwrap();
        assert_all_readable::<DIM>(&rtxn, database);
        assert_deleted_items_are_gone(&rtxn, database, &deleted);
        deleted.clear();

        // get batch of write operations and apply them
        info!("WRITING");
        let mut data = [0_u8; 1024 * 1024 * 1];
        rng.fill(&mut data);
        let mut u = Unstructured::new(&data);
        let ops: Vec<WriteOp<DIM>> = (0..100).map(|_| u.arbitrary().unwrap()).collect();

        let mut wtxn = env.write_txn().unwrap();
        let writer = Writer::new(database, 0, DIM);

        for op in ops {
            match op {
                WriteOp::Add(id) => {
                    let id = id % (NUMEL as u32);
                    let vector = gen_vec();
                    assert!(vector != [0.0f32; DIM]);
                    writer.add_item(&mut wtxn, id, &vector).unwrap();

                    // ensure added random id isn't registered in deleted
                    let _ = deleted.remove(id);
                }
                WriteOp::Del(id) => {
                    let id = id % (NUMEL as u32);
                    let _ = writer.del_item(&mut wtxn, id).unwrap();
                    deleted.insert(id);
                }
            }
        }

        writer.builder(&mut rng).ef_construction(32).build::<M, M0>(&mut wtxn).unwrap();
        wtxn.commit().unwrap();
    }
}
