use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

use hashbrown::HashMap;

use crate::Distance;

// TODO: ignore the phantom
#[derive(Debug)]
pub(crate) struct BuildStats<D> {
    /// a counter to see how many times `HnswBuilder.add_link` is invoked
    pub n_links_added: AtomicUsize,
    /// a counter tracking how many times we hit lmdb
    pub lmdb_hits: AtomicUsize,
    /// number of elements per layer
    pub layer_dist: HashMap<usize, usize>,

    _phantom: PhantomData<D>,
}

impl<D: Distance> BuildStats<D> {
    pub fn new() -> BuildStats<D> {
        BuildStats {
            n_links_added: AtomicUsize::new(0),
            lmdb_hits: AtomicUsize::new(0),
            layer_dist: HashMap::default(),
            _phantom: PhantomData,
        }
    }

    pub fn incr_link_count(&self, val: usize) {
        self.n_links_added.fetch_add(val, Ordering::Relaxed);
    }

    pub fn incr_lmdb_hits(&self) {
        self.lmdb_hits.fetch_add(1, Ordering::Relaxed);
    }
}
