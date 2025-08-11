use std::sync::atomic::Ordering;

use steppe::{make_atomic_progress, make_enum_progress, AtomicSubStep, NamedStep, Progress, Step};

// RemoveItemsFromExistingTrees,
// RetrievingTheUsedTreeNodes,
// RetrievingTheItems,
// RetrievingTheTreeNodes,
// InsertItemsInCurrentTrees,
// RetrieveTheLargeDescendants,
// CreateTreesForItems,
// WriteTheMetadata,

make_enum_progress! {
    pub enum HannoyBuild {
        RetrievingTheItemsIds,
        RetrieveTheUpdatedItems,
    }
}

make_atomic_progress!(UpdatingItems alias UpdatingItemsStep => "updating items");
