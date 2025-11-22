use crate::{
    distance::Cosine,
    tests::{create_database_indices_with_items, DatabaseHandle},
    Database, Reader, Writer,
};
use arbitrary::{Arbitrary, Unstructured};
use heed::RoTxn;
use rand::{self, rngs::StdRng, Rng, SeedableRng};
use roaring::RoaringBitmap;

#[derive(Debug)]
struct Item<const M: usize> {
    id: u32,
    data: [f32; M],
}

impl<'a, const M: usize> Arbitrary<'a> for Item<M> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let data: [f32; M] = u.arbitrary()?;
        let id: u32 = u.arbitrary()?;

        Ok(Item { data, id })
    }
}

#[derive(Arbitrary, Debug)]
enum WriteOp<const M: usize> {
    Add(Item<M>),
    Del(u32),
}

fn assert_all_readable<const DIM: usize>(rtxn: &RoTxn, database: Database<Cosine>) {
    let reader = Reader::<Cosine>::open(&rtxn, 0, database).unwrap();
    let n = reader.item_ids().len() as usize;
    let found = reader.nns(n).ef_search(n).by_vector(&rtxn, &[0.0; DIM]).unwrap().into_nns();
    assert_eq!(&RoaringBitmap::from_iter(found.into_iter().map(|(id, _)| id)), reader.item_ids())
}

#[test]
fn random_read_writes() {
    let seed: u64 = rand::random();
    let mut rng = StdRng::seed_from_u64(seed);

    const DIM: usize = 768;
    const NUMEL: usize = 1000;
    const M: usize = 3;
    const M0: usize = 3;

    let DatabaseHandle { env, database, tempdir: _ } =
        create_database_indices_with_items::<Cosine, DIM, M, M0, _>(0..1, NUMEL, &mut rng);

    for _ in 0.. {
        let rtxn = env.read_txn().unwrap();
        assert_all_readable::<DIM>(&rtxn, database);

        // get batch of write operations and apply them
        let mut data = [0_u8; 1024 * 1024 * 1];
        rng.fill(&mut data);
        let mut u = Unstructured::new(&data);
        let ops: Vec<WriteOp<DIM>> = (0..1000).map(|_| u.arbitrary().unwrap()).collect();

        let mut wtxn = env.write_txn().unwrap();
        let writer = Writer::new(database, 0, DIM);

        for op in ops {
            match op {
                WriteOp::Add(item) => {
                    let Item { data, id } = item;
                    let id = id % (NUMEL as u32);
                    writer.add_item(&mut wtxn, id, &data).unwrap();
                }
                WriteOp::Del(id) => {
                    let id = id % (NUMEL as u32);
                    let _ = writer.del_item(&mut wtxn, id).unwrap();
                }
            }
        }

        writer.builder(&mut rng).ef_construction(32).build::<M, M0>(&mut wtxn).unwrap();
        wtxn.commit().unwrap();
    }
}
