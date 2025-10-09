use std::borrow::Cow;

use heed::BoxedError;
use roaring::RoaringBitmap;

/// A `heed` codec for `roaring::RoaringBitmap`.
///
/// Encodes via [`RoaringBitmap::serialize_into`] and decodes via
/// [`RoaringBitmap::deserialize_unchecked_from`].
///
/// # Safety
/// Decoding trusts the bytes. Only use with data written by this codec or
/// switch to [`RoaringBitmap::deserialize_from`] if you need validation.

pub struct RoaringBitmapCodec;

impl heed::BytesDecode<'_> for RoaringBitmapCodec {
    type DItem = RoaringBitmap;

    fn bytes_decode(bytes: &[u8]) -> Result<Self::DItem, BoxedError> {
        RoaringBitmap::deserialize_unchecked_from(bytes).map_err(Into::into)
    }
}

impl heed::BytesEncode<'_> for RoaringBitmapCodec {
    type EItem = RoaringBitmap;

    fn bytes_encode(item: &Self::EItem) -> Result<Cow<'_, [u8]>, BoxedError> {
        let mut bytes = Vec::with_capacity(item.serialized_size());
        item.serialize_into(&mut bytes)?;
        Ok(Cow::Owned(bytes))
    }
}
