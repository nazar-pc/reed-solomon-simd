use core::marker::PhantomData;

use crate::{
    engine::{self, Engine, ShardStorage, GF_MODULUS, GF_ORDER},
    rate::{
        decode_in_place_begin, in_place_shard_stride, undo_in_place_last_chunk_encoding,
        DecoderWork, EncoderWork, InPlaceWork, Rate, RateDecoder, RateEncoder, ReceivedShardFlags,
        ReceivedShards,
    },
    DecoderResult, EncoderResult, Error,
};

// ======================================================================
// HighRate - PUBLIC

/// Reed-Solomon encoder/decoder generator using only high rate.
pub struct HighRate<E: Engine>(PhantomData<E>);

impl<E: Engine> Rate<E> for HighRate<E> {
    type RateEncoder = HighRateEncoder<E>;
    type RateDecoder = HighRateDecoder<E>;

    fn supports(original_count: usize, recovery_count: usize) -> bool {
        original_count > 0
            && recovery_count > 0
            && original_count < GF_ORDER
            && recovery_count < GF_ORDER
            && recovery_count.next_power_of_two() + original_count <= GF_ORDER
    }
}

// ======================================================================
// HighRateEncoder - IN-PLACE - PRIVATE

/// High rate encoding of an already prepared working buffer.
///
/// Shared by [`HighRateEncoder::encode`] and [`HighRateEncoder::encode_in_place`].
fn encode_in_place<E: Engine, S: ShardStorage>(
    engine: &E,
    work: &mut S,
    original_count: usize,
    recovery_count: usize,
) {
    let chunk_size = recovery_count.next_power_of_two();

    // FIRST CHUNK

    let first_count = core::cmp::min(original_count, chunk_size);

    work.zero(first_count..chunk_size);
    engine::ifft_skew_end(engine, work, 0, chunk_size, first_count);

    if original_count > chunk_size {
        // FULL CHUNKS

        let mut chunk_start = chunk_size;
        while chunk_start + chunk_size <= original_count {
            engine::ifft_skew_end(engine, work, chunk_start, chunk_size, chunk_size);
            // SAFETY: Shard-ranges `0 .. chunk_size` and
            // `chunk_start .. chunk_start + chunk_size` are within bounds and don't overlap
            unsafe { engine::xor_within(work, 0, chunk_start, chunk_size) };
            chunk_start += chunk_size;
        }

        // FINAL PARTIAL CHUNK

        let last_count = original_count % chunk_size;
        if last_count > 0 {
            work.zero(chunk_start + last_count..);
            engine::ifft_skew_end(engine, work, chunk_start, chunk_size, last_count);
            // SAFETY: Shard-ranges `0 .. chunk_size` and
            // `chunk_start .. chunk_start + chunk_size` are within bounds and don't overlap
            unsafe { engine::xor_within(work, 0, chunk_start, chunk_size) };
        }
    }

    // FFT

    engine.fft(work, 0, chunk_size, recovery_count, 0);
}

// ======================================================================
// HighRateEncoder - PUBLIC

/// Reed-Solomon encoder using only high rate.
pub struct HighRateEncoder<E: Engine> {
    engine: E,
    work: EncoderWork,
}

impl<E: Engine> RateEncoder<E> for HighRateEncoder<E> {
    type Rate = HighRate<E>;

    fn add_original_shard<T: AsRef<[u8]>>(&mut self, original_shard: T) -> Result<(), Error> {
        self.work.add_original_shard(original_shard)
    }

    fn encode(&mut self) -> Result<EncoderResult<'_>, Error> {
        let (mut work, original_count, recovery_count) = self.work.encode_begin()?;

        encode_in_place(&self.engine, &mut work, original_count, recovery_count);

        // UNDO LAST CHUNK ENCODING

        self.work.undo_last_chunk_encoding();

        // DONE

        Ok(EncoderResult::new(&mut self.work))
    }

    fn into_parts(self) -> (E, EncoderWork) {
        (self.engine, self.work)
    }

    fn new(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        engine: E,
        work: Option<EncoderWork>,
    ) -> Result<Self, Error> {
        let mut work = work.unwrap_or_default();
        Self::reset_work(original_count, recovery_count, shard_bytes, &mut work)?;
        Ok(Self { engine, work })
    }

    fn reset(
        &mut self,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error> {
        Self::reset_work(original_count, recovery_count, shard_bytes, &mut self.work)
    }
}

// ======================================================================
// HighRateEncoder - IN-PLACE - PUBLIC

impl<E: Engine> HighRateEncoder<E> {
    /// Size in bytes of the working buffer required by [`HighRateEncoder::encode_in_place`].
    ///
    /// This is a `const fn` so that the working buffer can be a compile-time sized array when
    /// shard counts are known statically.
    ///
    /// The result is only meaningful for supported `original_count` / `recovery_count`
    /// combinations, which [`HighRateEncoder::encode_in_place`] validates.
    #[must_use]
    pub const fn in_place_work_bytes(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> usize {
        Self::work_count(original_count, recovery_count) * in_place_shard_stride(shard_bytes)
    }

    /// Encodes in a caller-provided working buffer, without allocating anything.
    ///
    /// This is a lower level alternative to the [`RateEncoder`] API for callers that want to
    /// control memory usage: the library allocates no working space of its own, original shards
    /// are read from the working buffer and recovery shards are written back into it.
    ///
    /// The working buffer is either a byte buffer with the flat layout described below, or a
    /// [`ShardStorage`] of the caller's own, see [`InPlaceWork`].
    ///
    /// [`ShardStorage`]: crate::engine::ShardStorage
    /// [`InPlaceWork`]: crate::rate::InPlaceWork
    ///
    /// The working buffer is not simply the original shards: encoding needs
    /// [`HighRateEncoder::in_place_work_bytes`] of space, which is at least as much as the
    /// original shards occupy and often more (shards are padded to [`in_place_shard_stride`] and
    /// the shard count is padded up to a multiple of the chunk size). Shards are stored in it
    /// consecutively, [`in_place_shard_stride`] bytes apart:
    ///
    /// - **before** the call the buffer must contain original shards `0..original_count`, which
    ///   can be written with [`write_in_place_shard`] (or produced in place, see its
    ///   documentation);
    /// - **after** the call the buffer contains recovery shards `0..recovery_count`, which can be
    ///   read with [`read_in_place_shard`].
    ///
    /// Original shards are consumed in the process: the working buffer no longer contains them
    /// once encoding is done. Contents of slots beyond `recovery_count` are unspecified.
    ///
    /// [`in_place_shard_stride`]: crate::rate::in_place_shard_stride
    /// [`write_in_place_shard`]: crate::rate::write_in_place_shard
    /// [`read_in_place_shard`]: crate::rate::read_in_place_shard
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidWorkBufferSize`] if `work` is smaller than
    /// [`HighRateEncoder::in_place_work_bytes`], otherwise same as [`RateEncoder::validate`].
    pub fn encode_in_place<W: InPlaceWork + ?Sized>(
        engine: &E,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        work: &mut W,
    ) -> Result<(), Error> {
        Self::validate(original_count, recovery_count, shard_bytes)?;

        let work_count = Self::work_count(original_count, recovery_count);
        let mut work = work.in_place_storage(work_count, shard_bytes)?;

        encode_in_place(engine, &mut work, original_count, recovery_count);

        undo_in_place_last_chunk_encoding(&mut work, shard_bytes, 0..recovery_count);

        Ok(())
    }
}

// ======================================================================
// HighRateEncoder - PRIVATE

impl<E: Engine> HighRateEncoder<E> {
    fn reset_work(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        work: &mut EncoderWork,
    ) -> Result<(), Error> {
        Self::validate(original_count, recovery_count, shard_bytes)?;
        work.reset(
            original_count,
            recovery_count,
            shard_bytes,
            Self::work_count(original_count, recovery_count),
        );
        Ok(())
    }

    /// Number of shards of working space needed.
    ///
    /// Only meaningful for supported `original_count` / `recovery_count` combinations, which
    /// callers check with [`RateEncoder::validate`] first.
    const fn work_count(original_count: usize, recovery_count: usize) -> usize {
        original_count.next_multiple_of(recovery_count.next_power_of_two())
    }
}

// ======================================================================
// HighRateDecoder - IN-PLACE - PRIVATE

/// High rate decoding of an already prepared working buffer.
///
/// Shared by [`HighRateDecoder::decode`] and [`HighRateDecoder::decode_in_place`].
fn decode_in_place<E: Engine, R: ReceivedShards, S: ShardStorage>(
    engine: &E,
    work: &mut S,
    original_count: usize,
    recovery_count: usize,
    received: &R,
) {
    let chunk_size = recovery_count.next_power_of_two();
    let original_end = chunk_size + original_count;
    let work_count = work.len();

    // ERASURE LOCATIONS

    let mut erasures = [0; GF_ORDER];

    for index in received.missing_in(0..recovery_count) {
        erasures[index] = 1;
    }

    erasures[recovery_count..chunk_size].fill(1);

    for index in received.missing_in(chunk_size..original_end) {
        erasures[index] = 1;
    }

    // EVALUATE POLYNOMIAL

    E::eval_poly(&mut erasures, original_end);

    // MULTIPLY SHARDS

    // work[               .. recovery_count] = recovery * erasures
    // work[recovery_count .. chunk_size    ] = 0
    // work[chunk_size     .. original_end  ] = original * erasures
    // work[original_end   ..               ] = 0

    for i in 0..recovery_count {
        if received.received(i) {
            engine.mul(&mut work[i], erasures[i]);
        } else {
            work[i].fill([0; 64]);
        }
    }

    work.zero(recovery_count..chunk_size);

    for i in chunk_size..original_end {
        if received.received(i) {
            engine.mul(&mut work[i], erasures[i]);
        } else {
            work[i].fill([0; 64]);
        }
    }

    work.zero(original_end..);

    // IFFT / FORMAL DERIVATIVE / FFT

    engine.ifft(work, 0, work_count, original_end, 0);
    engine::formal_derivative(work);
    engine.fft(work, 0, work_count, original_end, 0);

    // REVEAL ERASURES

    for i in chunk_size..original_end {
        if !received.received(i) {
            engine.mul(&mut work[i], GF_MODULUS - erasures[i]);
        }
    }
}

// ======================================================================
// HighRateDecoder - PUBLIC

/// Reed-Solomon decoder using only high rate.
pub struct HighRateDecoder<E: Engine> {
    engine: E,
    work: DecoderWork,
}

impl<E: Engine> RateDecoder<E> for HighRateDecoder<E> {
    type Rate = HighRate<E>;

    fn add_original_shard<T: AsRef<[u8]>>(
        &mut self,
        index: usize,
        original_shard: T,
    ) -> Result<(), Error> {
        self.work.add_original_shard(index, original_shard)
    }

    fn add_recovery_shard<T: AsRef<[u8]>>(
        &mut self,
        index: usize,
        recovery_shard: T,
    ) -> Result<(), Error> {
        self.work.add_recovery_shard(index, recovery_shard)
    }

    fn decode(&mut self) -> Result<DecoderResult<'_>, Error> {
        let Some((mut work, original_count, recovery_count, received)) =
            self.work.decode_begin()?
        else {
            // Nothing to do, original data is complete.
            return Ok(DecoderResult::new(&mut self.work));
        };

        decode_in_place(
            &self.engine,
            &mut work,
            original_count,
            recovery_count,
            &received,
        );

        // UNDO LAST CHUNK ENCODING

        self.work.undo_last_chunk_encoding();

        // DONE

        Ok(DecoderResult::new(&mut self.work))
    }

    fn into_parts(self) -> (E, DecoderWork) {
        (self.engine, self.work)
    }

    fn new(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        engine: E,
        work: Option<DecoderWork>,
    ) -> Result<Self, Error> {
        let mut work = work.unwrap_or_default();
        Self::reset_work(original_count, recovery_count, shard_bytes, &mut work)?;
        Ok(Self { engine, work })
    }

    fn reset(
        &mut self,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error> {
        Self::reset_work(original_count, recovery_count, shard_bytes, &mut self.work)
    }
}

// ======================================================================
// HighRateDecoder - IN-PLACE - PUBLIC

impl<E: Engine> HighRateDecoder<E> {
    /// Size in bytes of the working buffer required by [`HighRateDecoder::decode_in_place`].
    ///
    /// Decoding needs more working space than encoding does, because the working buffer holds
    /// both the received original shards and the received recovery shards.
    ///
    /// This is a `const fn` so that the working buffer can be a compile-time sized array when
    /// shard counts are known statically.
    ///
    /// The result is only meaningful for supported `original_count` / `recovery_count`
    /// combinations, which [`HighRateDecoder::decode_in_place`] validates.
    #[must_use]
    pub const fn in_place_work_bytes(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> usize {
        Self::work_count(original_count, recovery_count) * in_place_shard_stride(shard_bytes)
    }

    /// Working buffer slot of original shard `index` for [`HighRateDecoder::decode_in_place`].
    ///
    /// Pass the returned slot to [`write_in_place_shard`] and [`read_in_place_shard`].
    ///
    /// [`write_in_place_shard`]: crate::rate::write_in_place_shard
    /// [`read_in_place_shard`]: crate::rate::read_in_place_shard
    #[must_use]
    pub const fn in_place_original_slot(
        _original_count: usize,
        recovery_count: usize,
        index: usize,
    ) -> usize {
        recovery_count.next_power_of_two() + index
    }

    /// Working buffer slot of recovery shard `index` for [`HighRateDecoder::decode_in_place`].
    ///
    /// Pass the returned slot to [`write_in_place_shard`].
    ///
    /// [`write_in_place_shard`]: crate::rate::write_in_place_shard
    #[must_use]
    pub const fn in_place_recovery_slot(
        _original_count: usize,
        _recovery_count: usize,
        index: usize,
    ) -> usize {
        index
    }

    /// Decodes in a caller-provided working buffer, without allocating anything.
    ///
    /// This is a lower level alternative to the [`RateDecoder`] API for callers that want to
    /// control memory usage: the library allocates no working space of its own, received shards
    /// are read from the working buffer and restored original shards are written back into it.
    ///
    /// The working buffer is either a byte buffer with the flat layout described below, or a
    /// [`ShardStorage`] of the caller's own, see [`InPlaceWork`].
    ///
    /// [`ShardStorage`]: crate::engine::ShardStorage
    /// [`InPlaceWork`]: crate::rate::InPlaceWork
    ///
    /// The working buffer must be [`HighRateDecoder::in_place_work_bytes`] long, which is more
    /// than the shards themselves occupy. Shards are stored in it [`in_place_shard_stride`] bytes
    /// apart, and unlike in-place encoding the slot of a shard is not its index: use
    /// [`HighRateDecoder::in_place_original_slot`] and
    /// [`HighRateDecoder::in_place_recovery_slot`] to find it.
    ///
    /// `original_received` and `recovery_received` tell which of the `original_count` original
    /// and `recovery_count` recovery shards the working buffer contains. Contents of the slots of
    /// shards that were not received are ignored, they do not need to be zeroed.
    ///
    /// [`ReceivedShards`] is implemented for `[bool]` slices/arrays as well as the bit-packed
    /// [`ReceivedShardBits`], which needs no allocation for small shard counts.
    ///
    /// [`ReceivedShards`]: crate::rate::ReceivedShards
    /// [`ReceivedShardBits`]: crate::rate::ReceivedShardBits
    ///
    /// On success the slots of original shards that were not received contain the restored
    /// original shards, which can be read with [`read_in_place_shard`]. Contents of all other
    /// slots are unspecified.
    ///
    /// [`in_place_shard_stride`]: crate::rate::in_place_shard_stride
    /// [`read_in_place_shard`]: crate::rate::read_in_place_shard
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotEnoughShards`] if fewer shards than `original_count` were received and
    /// [`Error::InvalidWorkBufferSize`] if `work` is smaller than
    /// [`HighRateDecoder::in_place_work_bytes`], otherwise same as [`RateDecoder::validate`].
    pub fn decode_in_place<W: InPlaceWork + ?Sized>(
        engine: &E,
        original_count: usize,
        recovery_count: usize,
        original_received: impl ReceivedShards,
        recovery_received: impl ReceivedShards,
        shard_bytes: usize,
        work: &mut W,
    ) -> Result<(), Error> {
        Self::validate(original_count, recovery_count, shard_bytes)?;

        if !decode_in_place_begin(
            original_count,
            recovery_count,
            &original_received,
            &recovery_received,
        )? {
            // Nothing to do, original data is complete.
            return Ok(());
        }

        let work_count = Self::work_count(original_count, recovery_count);
        let mut work = work.in_place_storage(work_count, shard_bytes)?;

        let chunk_size = recovery_count.next_power_of_two();
        let received = ReceivedShardFlags {
            original: original_received,
            original_base_pos: chunk_size,
            original_count,
            recovery: recovery_received,
            recovery_base_pos: 0,
            recovery_count,
        };

        decode_in_place(engine, &mut work, original_count, recovery_count, &received);

        undo_in_place_last_chunk_encoding(
            &mut work,
            shard_bytes,
            chunk_size..chunk_size + original_count,
        );

        Ok(())
    }
}

// ======================================================================
// HighRateDecoder - PRIVATE

impl<E: Engine> HighRateDecoder<E> {
    fn reset_work(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        work: &mut DecoderWork,
    ) -> Result<(), Error> {
        Self::validate(original_count, recovery_count, shard_bytes)?;

        // work[..recovery_count     ]  =  recovery
        // work[recovery_count_pow2..]  =  original
        work.reset(
            original_count,
            recovery_count,
            shard_bytes,
            recovery_count.next_power_of_two(),
            0,
            Self::work_count(original_count, recovery_count),
        );

        Ok(())
    }

    /// Number of shards of working space needed.
    ///
    /// Only meaningful for supported `original_count` / `recovery_count` combinations, which
    /// callers check with [`RateDecoder::validate`] first.
    const fn work_count(original_count: usize, recovery_count: usize) -> usize {
        (recovery_count.next_power_of_two() + original_count).next_power_of_two()
    }
}

// ======================================================================
// TESTS

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;

    // ============================================================
    // ROUNDTRIPS - SINGLE ROUND

    #[test]
    fn roundtrip_all_originals_missing() {
        roundtrip_single!(
            HighRate,
            3,
            3,
            1024,
            test_util::EITHER_3_3,
            &[],
            &[0..3],
            133,
        );
    }

    #[test]
    fn roundtrip_no_originals_missing() {
        roundtrip_single!(HighRate, 3, 2, 1024, test_util::HIGH_3_2, &[0..3], &[], 132);
    }

    #[test]
    fn roundtrips_tiny() {
        for (original_count, recovery_count, seed, recovery_hash) in test_util::HIGH_TINY {
            roundtrip_single!(
                HighRate,
                *original_count,
                *recovery_count,
                1024,
                recovery_hash,
                &[*recovery_count..*original_count],
                &[0..core::cmp::min(*original_count, *recovery_count)],
                *seed,
            );
        }
    }

    #[test]
    #[ignore]
    fn roundtrip_3000_30000() {
        roundtrip_single!(
            HighRate,
            3000,
            30000,
            64,
            test_util::HIGH_3000_30000_14,
            &[],
            &[0..3000],
            14,
        );
    }

    #[test]
    #[ignore]
    fn roundtrip_32768_32768() {
        roundtrip_single!(
            HighRate,
            32768,
            32768,
            64,
            test_util::EITHER_32768_32768_11,
            &[],
            &[0..32768],
            11,
        );
    }

    #[test]
    #[ignore]
    fn roundtrip_60000_3000() {
        roundtrip_single!(
            HighRate,
            60000,
            3000,
            64,
            test_util::HIGH_60000_3000_12,
            &[3000..60000],
            &[0..3000],
            12,
        );
    }

    #[test]
    fn roundtrip_34000_2000_shard_size_8() {
        roundtrip_single!(
            HighRate,
            34000,
            2000,
            8,
            test_util::HIGH_34000_2000_123_8,
            &[0..32000],
            &[0..2000],
            123
        );
    }

    // ============================================================
    // ROUNDTRIPS - TWO ROUNDS

    #[test]
    fn two_rounds_implicit_reset() {
        roundtrip_two_rounds!(
            HighRate,
            false,
            (3, 2, 1024, test_util::HIGH_3_2, &[1], &[0, 1], 132),
            (3, 2, 1024, test_util::HIGH_3_2_232, &[0], &[0, 1], 232),
        );
    }

    #[test]
    fn two_rounds_explicit_reset() {
        roundtrip_two_rounds!(
            HighRate,
            true,
            (3, 2, 1024, test_util::HIGH_3_2, &[1], &[0, 1], 132),
            (5, 2, 1024, test_util::HIGH_5_2, &[0, 2, 4], &[0, 1], 152),
        );
    }

    // ============================================================
    // HighRate

    mod high_rate {
        use crate::{
            engine::NoSimd,
            rate::{HighRate, Rate},
            Error,
        };

        #[test]
        fn decoder() {
            assert_eq!(
                HighRate::<NoSimd>::decoder(4096, 61440, 64, NoSimd::new(), None).err(),
                Some(Error::UnsupportedShardCount {
                    original_count: 4096,
                    recovery_count: 61440,
                })
            );

            assert!(HighRate::<NoSimd>::decoder(61440, 4096, 64, NoSimd::new(), None).is_ok());
        }

        #[test]
        fn encoder() {
            assert_eq!(
                HighRate::<NoSimd>::encoder(4096, 61440, 64, NoSimd::new(), None).err(),
                Some(Error::UnsupportedShardCount {
                    original_count: 4096,
                    recovery_count: 61440,
                })
            );

            assert!(HighRate::<NoSimd>::encoder(61440, 4096, 64, NoSimd::new(), None).is_ok());
        }

        #[test]
        fn supports() {
            assert!(!HighRate::<NoSimd>::supports(0, 1));
            assert!(!HighRate::<NoSimd>::supports(1, 0));

            assert!(!HighRate::<NoSimd>::supports(4096, 61440));

            assert!(HighRate::<NoSimd>::supports(61440, 4096));
            assert!(!HighRate::<NoSimd>::supports(61440, 4097));
            assert!(!HighRate::<NoSimd>::supports(61441, 4096));

            assert!(!HighRate::<NoSimd>::supports(usize::MAX, usize::MAX));
        }

        #[test]
        fn validate() {
            assert_eq!(
                HighRate::<NoSimd>::validate(1, 1, 123).err(),
                Some(Error::InvalidShardSize { shard_bytes: 123 })
            );

            assert_eq!(
                HighRate::<NoSimd>::validate(4096, 61440, 64).err(),
                Some(Error::UnsupportedShardCount {
                    original_count: 4096,
                    recovery_count: 61440,
                })
            );

            assert!(HighRate::<NoSimd>::validate(61440, 4096, 64).is_ok());
        }
    }

    // ============================================================
    // HighRateEncoder

    mod high_rate_encoder {
        use crate::{
            engine::NoSimd,
            rate::{HighRateEncoder, RateEncoder},
            Error,
        };

        // ==================================================
        // ERRORS

        test_rate_encoder_errors! {HighRateEncoder}

        // ==================================================
        // supports

        #[test]
        fn supports() {
            assert!(!HighRateEncoder::<NoSimd>::supports(4096, 61440));
            assert!(HighRateEncoder::<NoSimd>::supports(61440, 4096));
        }

        // ==================================================
        // validate

        #[test]
        fn validate() {
            assert_eq!(
                HighRateEncoder::<NoSimd>::validate(1, 1, 123).err(),
                Some(Error::InvalidShardSize { shard_bytes: 123 })
            );

            assert_eq!(
                HighRateEncoder::<NoSimd>::validate(4096, 61440, 64).err(),
                Some(Error::UnsupportedShardCount {
                    original_count: 4096,
                    recovery_count: 61440,
                })
            );

            assert!(HighRateEncoder::<NoSimd>::validate(61440, 4096, 64).is_ok());
        }

        // ==================================================
        // work_count

        #[test]
        fn work_count() {
            assert_eq!(HighRateEncoder::<NoSimd>::work_count(1, 1), 1);
            assert_eq!(HighRateEncoder::<NoSimd>::work_count(4096, 1024), 4096);
            assert_eq!(HighRateEncoder::<NoSimd>::work_count(4097, 1024), 5120);
            assert_eq!(HighRateEncoder::<NoSimd>::work_count(4097, 1025), 6144);
            assert_eq!(HighRateEncoder::<NoSimd>::work_count(32768, 32768), 32768);
        }
    }

    // ============================================================
    // HighRateDecoder

    mod high_rate_decoder {
        use crate::{
            engine::NoSimd,
            rate::{HighRateDecoder, RateDecoder},
            Error,
        };

        // ==================================================
        // ERRORS

        test_rate_decoder_errors! {HighRateDecoder}

        // ==================================================
        // supports

        #[test]
        fn supports() {
            assert!(!HighRateDecoder::<NoSimd>::supports(4096, 61440));
            assert!(HighRateDecoder::<NoSimd>::supports(61440, 4096));
        }

        // ==================================================
        // validate

        #[test]
        fn validate() {
            assert_eq!(
                HighRateDecoder::<NoSimd>::validate(1, 1, 123).err(),
                Some(Error::InvalidShardSize { shard_bytes: 123 })
            );

            assert_eq!(
                HighRateDecoder::<NoSimd>::validate(4096, 61440, 64).err(),
                Some(Error::UnsupportedShardCount {
                    original_count: 4096,
                    recovery_count: 61440,
                })
            );

            assert!(HighRateDecoder::<NoSimd>::validate(61440, 4096, 64).is_ok());
        }

        // ==================================================
        // work_count

        #[test]
        fn work_count() {
            assert_eq!(HighRateDecoder::<NoSimd>::work_count(1, 1), 2);
            assert_eq!(HighRateDecoder::<NoSimd>::work_count(2048, 1025), 4096);
            assert_eq!(HighRateDecoder::<NoSimd>::work_count(2049, 1025), 8192);
            assert_eq!(HighRateDecoder::<NoSimd>::work_count(3072, 1024), 4096);
            assert_eq!(HighRateDecoder::<NoSimd>::work_count(3073, 1024), 8192);
            assert_eq!(HighRateDecoder::<NoSimd>::work_count(32768, 32768), 65536);
        }
    }
}
