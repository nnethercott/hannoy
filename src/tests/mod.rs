use std::fmt;

use heed::types::LazyDecode;
use heed::{Env, EnvOpenOptions, WithTls};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tempfile::TempDir;

use crate::version::VersionCodec;
use crate::{Database, Distance, MetadataCodec, NodeCodec, NodeMode, Reader};

mod writer;

pub struct DatabaseHandle<D> {
    pub env: Env<WithTls>,
    pub database: Database<D>,
    #[allow(unused)]
    pub tempdir: TempDir,
}

impl<D: Distance> fmt::Display for DatabaseHandle<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rtxn = self.env.read_txn().unwrap();

        let mut old_index;
        let mut current_index = None;
        let mut last_mode = NodeMode::Item;

        for result in
            self.database.remap_data_type::<LazyDecode<NodeCodec<D>>>().iter(&rtxn).unwrap()
        {
            let (key, lazy_node) = result.unwrap();

            old_index = current_index;
            current_index = Some(key.index);

            if old_index != current_index {
                let reader =
                    Reader::<D>::open(&rtxn, current_index.unwrap(), self.database).unwrap();

                // ensure everything OK with graph
                reader.assert_validity(&rtxn).unwrap();

                writeln!(f, "==================")?;
                writeln!(f, "Dumping index {}", current_index.unwrap())?;
            }

            if last_mode != key.node.mode && key.node.mode == NodeMode::Item {
                writeln!(f)?;
                last_mode = key.node.mode;
            }

            match key.node.mode {
                NodeMode::Item => {
                    let item = lazy_node.decode().unwrap();
                    writeln!(f, "Item {}: {item:?}", key.node.item)?;
                }
                NodeMode::Links => {
                    let links = lazy_node.decode().unwrap();
                    writeln!(f, "Links {}: {links:?}", key.node.item)?;
                }
                NodeMode::Metadata if key.node.item == 0 => {
                    let metadata = self
                        .database
                        .remap_data_type::<MetadataCodec>()
                        .get(&rtxn, &key)
                        .unwrap()
                        .unwrap();
                    writeln!(f, "Root: {metadata:?}")?;
                }
                NodeMode::Metadata if key.node.item == 1 => {
                    let version = self
                        .database
                        .remap_data_type::<VersionCodec>()
                        .get(&rtxn, &key)
                        .unwrap()
                        .unwrap();
                    writeln!(f, "Version: {version:?}")?;
                }
                NodeMode::Updated | NodeMode::Metadata => panic!(),
            }
        }

        Ok(())
    }
}

fn create_database<D: Distance>() -> DatabaseHandle<D> {
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();

    let dir = tempfile::tempdir().unwrap();
    let env =
        unsafe { EnvOpenOptions::new().map_size(200 * 1024 * 1024).open(dir.path()) }.unwrap();
    let mut wtxn = env.write_txn().unwrap();
    let database: Database<D> = env.create_database(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();
    DatabaseHandle { env, database, tempdir: dir }
}

fn rng() -> StdRng {
    StdRng::from_seed(std::array::from_fn(|_| 42))
}
