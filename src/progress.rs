use steppe::{make_atomic_progress, make_enum_progress};

make_enum_progress! {
    pub enum HannoyBuild {
        RetrievingTheItemsIds,
        DeletingTheLinks,
        RetrieveTheUpdatedItems,
        ResolveGraphEntryPoints,
        BuildingTheGraph,
        PatchOldNewDeletedLinks,
        WritingTheItems,
        WriteTheMetadata,
        ConvertingArroyToHannoy,
    }
}

make_atomic_progress!(InsertItems alias AtomicInsertItemsStep => "inserting items");
