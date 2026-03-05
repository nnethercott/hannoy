use std::borrow::Cow;
use std::fmt;

use heed::BoxedError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum UpdateStatus {
    Updated = 0,
    Removed = 1,
}

pub enum UpdateStatusCodec {}

impl heed::BytesEncode<'_> for UpdateStatusCodec {
    type EItem = UpdateStatus;

    fn bytes_encode(item: &'_ Self::EItem) -> Result<Cow<'_, [u8]>, BoxedError> {
        Ok(Cow::Owned(vec![*item as u8]))
    }
}

impl heed::BytesDecode<'_> for UpdateStatusCodec {
    type DItem = UpdateStatus;

    fn bytes_decode(bytes: &'_ [u8]) -> Result<Self::DItem, BoxedError> {
        match bytes {
            [0] => Ok(UpdateStatus::Updated),
            [1] => Ok(UpdateStatus::Removed),
            _ => Err(Box::new(InvalidUpdateStatusDecoding { unknown_tag: bytes.to_vec() })),
        }
    }
}

#[derive(Debug, thiserror::Error)]
struct InvalidUpdateStatusDecoding {
    unknown_tag: Vec<u8>,
}

impl fmt::Display for InvalidUpdateStatusDecoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.unknown_tag[..] {
            bytes @ [_, ..] => write!(f, "Invalid update status decoding: unknown tag {bytes:?}"),
            [] => write!(f, "Invalid update status decoding: empty array of bytes"),
        }
    }
}
