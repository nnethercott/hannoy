use std::fmt;

use bytemuck::{Pod, Zeroable};
use simsimd::SpatialSimilarity;

use crate::distance::Distance;
use crate::node::Item;
use crate::spaces::simple::dot_product;
use crate::unaligned_vector::UnalignedVector;

/// The Cosine similarity is a measure of similarity between two
/// non-zero vectors defined in an inner product space. Cosine similarity
/// is the cosine of the angle between the vectors.
#[derive(Debug, Clone)]
pub enum Cosine {}

/// The header of Cosine leaf nodes.
#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy)]
pub struct NodeHeaderCosine {
    norm: f32,
}
impl fmt::Debug for NodeHeaderCosine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeHeaderCosine").field("norm", &format!("{:.4}", self.norm)).finish()
    }
}

impl Distance for Cosine {
    type Header = NodeHeaderCosine;
    type VectorCodec = f32;

    fn name() -> &'static str {
        "cosine"
    }

    fn new_header(vector: &UnalignedVector<Self::VectorCodec>) -> Self::Header {
        NodeHeaderCosine { norm: Self::norm_no_header(vector) }
    }

    fn distance(p: &Item<Self>, q: &Item<Self>) -> f32 {
        let p = unsafe{std::slice::from_raw_parts(p.vector.as_ptr() as *const f32, p.vector.len())};
        let q = unsafe{std::slice::from_raw_parts(q.vector.as_ptr() as *const f32, q.vector.len())};

        // dbg!(p.to_vec());
        f32::cosine(&p, &q).unwrap() as f32
    }

    fn norm_no_header(v: &UnalignedVector<Self::VectorCodec>) -> f32 {
        dot_product(v, v).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use crate::{distances::Cosine, internals::UnalignedVector, Distance};
    use crate::node::Item;

    #[test]
    fn alignment() {
        let p = vec![1.0f32, 0.0, 2.0];
        let q = vec![1.0f32, 0.0, 3.0];
        let p = UnalignedVector::from_slice(&p);
        let q = UnalignedVector::from_slice(&q);
        let p: Item<Cosine> = Item{header: Cosine::new_header(&p), vector: p};
        let q: Item<Cosine> = Item{header: Cosine::new_header(&q), vector: q};

        dbg!(Cosine::distance(&p, &q));
    }
}
