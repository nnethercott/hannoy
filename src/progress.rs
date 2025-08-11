use steppe::{make_atomic_progress, make_enum_progress};

make_enum_progress! {
    pub enum HannoyBuild {
        RetrievingTheItemsIds,
        RetrieveTheUpdatedItems,
        FetchItemPointers,
        FetchLinksPointers,
        ResolveGraphEntryPoints,
        BuildingTheGraph,
        PatchOldNewDeletedLinks,
        WritingTheItems,
        WriteTheMetadata,
    }
}

make_atomic_progress!(InsertItems alias AtomicInsertItemsStep => "inserting items");
