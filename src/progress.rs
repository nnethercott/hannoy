use std::sync::atomic::Ordering;

use steppe::{make_atomic_progress, make_enum_progress, AtomicSubStep, NamedStep, Progress, Step};

make_enum_progress! {
    pub enum HannoyBuild {
        UpdatingItems,
        WalkTheDog,
        TypeALotOnTheKeyboard,
        WalkTheDogAgain,
    }
}

make_atomic_progress!(UpdatingItems alias UpdatingItemsStep => "updating items");
make_atomic_progress!(KeyStrokes alias AtomicKeyStrokesStep => "key strokes");

// let mut progress = steppe::default::DefaultProgress::default();
// progress.update(TamosDay::PetTheDog); // We're at 0/4 and 0% of completion
// progress.update(TamosDay::WalkTheDog); // We're at 1/4 and 25% of completion
// progress.update(TamosDay::TypeALotOnTheKeyboard); // We're at 2/4 and 50% of completion
// let (atomic, key_strokes) = AtomicKeyStrokesStep::new(1000);
// progress.update(key_strokes);
// // Here we enqueued a new step that have 1000 total states. Since we don't want to take a lock everytime
// // we type on the keyboard we're instead going to increase an atomic without taking the mutex.
// atomic.fetch_add(500, Ordering::Relaxed);
// // If we fetch the progress at this point it should be exactly between 50% and 75%.
// progress.update(TamosDay::WalkTheDogAgain); // We're at 3/4 and 75% of completion
// // By enqueuing this new step the progress is going to drop everything that was pushed after the `TamosDay` type was pushed.
