use std::fmt;

pub use binary_quantized_cosine::{BinaryQuantizedCosine, NodeHeaderBinaryQuantizedCosine};
pub use binary_quantized_euclidean::{
    BinaryQuantizedEuclidean, NodeHeaderBinaryQuantizedEuclidean,
};
pub use binary_quantized_manhattan::{
    BinaryQuantizedManhattan, NodeHeaderBinaryQuantizedManhattan,
};
use bytemuck::{Pod, Zeroable};
pub use cosine::{Cosine, NodeHeaderCosine};
pub use euclidean::{Euclidean, NodeHeaderEuclidean};
pub use hamming::{Hamming, NodeHeaderHamming};
pub use manhattan::{Manhattan, NodeHeaderManhattan};

use crate::node::Item;
use crate::unaligned_vector::{UnalignedVector, UnalignedVectorCodec};

mod binary_quantized_cosine;
mod binary_quantized_euclidean;
mod binary_quantized_manhattan;
mod cosine;
mod euclidean;
mod hamming;
mod manhattan;

// FIXME: move elsewhere, also currently unused
fn new_leaf<D: Distance>(vec: Vec<f32>) -> Item<'static, D> {
    let vector = UnalignedVector::from_vec(vec);
    Item { header: D::new_header(&vector), vector }
}

/// A trait used by arroy to compute the distances,
/// compute the split planes, and normalize user vectors.
#[allow(missing_docs)]
pub trait Distance: Send + Sync + Sized + Clone + fmt::Debug + 'static {
    const DEFAULT_OVERSAMPLING: usize = 1;

    /// A header structure with informations related to the
    type Header: Pod + Zeroable + fmt::Debug;
    type VectorCodec: UnalignedVectorCodec;

    fn name() -> &'static str;

    fn new_header(vector: &UnalignedVector<Self::VectorCodec>) -> Self::Header;

    /// Returns a non-normalized distance.
    fn distance(p: &Item<Self>, q: &Item<Self>) -> f32;

    fn norm(item: &Item<Self>) -> f32 {
        Self::norm_no_header(&item.vector)
    }

    fn norm_no_header(v: &UnalignedVector<Self::VectorCodec>) -> f32;
}
