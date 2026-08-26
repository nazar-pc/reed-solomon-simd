//! In-place encoding and decoding must produce exactly the same results as the regular
//! encoders and decoders, both with a flat working buffer and with a custom shard storage.

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use reed_solomon_simd::engine::{DefaultEngine, Naive, NoSimd, ShardStorage};
use reed_solomon_simd::rate::{
    in_place_shard_stride, read_in_place_shard, write_in_place_shard, DefaultRateDecoder,
    DefaultRateEncoder, HighRate, HighRateDecoder, HighRateEncoder, LowRate, LowRateDecoder,
    LowRateEncoder, Rate, RateDecoder, RateEncoder, ReceivedShardBits,
};
use reed_solomon_simd::Error;
use std::ops::{Bound, Index, IndexMut, RangeBounds};

fn original_shards(original_count: usize, shard_bytes: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..original_count)
        .map(|_| {
            let mut shard = vec![0u8; shard_bytes];
            rng.fill_bytes(&mut shard);
            shard
        })
        .collect()
}

fn reference_recovery<R: RateEncoder<NoSimd>>(
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
    original: &[Vec<u8>],
) -> Vec<Vec<u8>> {
    let mut encoder = R::new(
        original_count,
        recovery_count,
        shard_bytes,
        NoSimd::new(),
        None,
    )
    .unwrap();
    for shard in original {
        encoder.add_original_shard(shard).unwrap();
    }
    let recovery = encoder
        .encode()
        .unwrap()
        .recovery_iter()
        .map(<[u8]>::to_vec)
        .collect();
    recovery
}

fn in_place_recovery(
    work_bytes: usize,
    encode: impl FnOnce(&NoSimd, &mut [u8]) -> Result<(), Error>,
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
    original: &[Vec<u8>],
) -> Vec<Vec<u8>> {
    let mut work = vec![0u8; work_bytes];
    for (index, shard) in original.iter().enumerate().take(original_count) {
        write_in_place_shard(&mut work, shard_bytes, index, shard).unwrap();
    }

    encode(&NoSimd::new(), &mut work).unwrap();

    (0..recovery_count)
        .map(|index| {
            read_in_place_shard(&work, shard_bytes, index)
                .unwrap()
                .to_vec()
        })
        .collect()
}

fn check(original_count: usize, recovery_count: usize, shard_bytes: usize, seed: u64) {
    let original = original_shards(original_count, shard_bytes, seed);

    if HighRate::<NoSimd>::supports(original_count, recovery_count) {
        let expected = reference_recovery::<HighRateEncoder<NoSimd>>(
            original_count,
            recovery_count,
            shard_bytes,
            &original,
        );
        let actual = in_place_recovery(
            HighRateEncoder::<NoSimd>::in_place_work_bytes(
                original_count,
                recovery_count,
                shard_bytes,
            ),
            |engine, work| {
                HighRateEncoder::encode_in_place(
                    engine,
                    original_count,
                    recovery_count,
                    shard_bytes,
                    work,
                )
            },
            original_count,
            recovery_count,
            shard_bytes,
            &original,
        );
        assert_eq!(
            actual, expected,
            "high rate {original_count}/{recovery_count} x {shard_bytes}"
        );
    }

    if LowRate::<NoSimd>::supports(original_count, recovery_count) {
        let expected = reference_recovery::<LowRateEncoder<NoSimd>>(
            original_count,
            recovery_count,
            shard_bytes,
            &original,
        );
        let actual = in_place_recovery(
            LowRateEncoder::<NoSimd>::in_place_work_bytes(
                original_count,
                recovery_count,
                shard_bytes,
            ),
            |engine, work| {
                LowRateEncoder::encode_in_place(
                    engine,
                    original_count,
                    recovery_count,
                    shard_bytes,
                    work,
                )
            },
            original_count,
            recovery_count,
            shard_bytes,
            &original,
        );
        assert_eq!(
            actual, expected,
            "low rate {original_count}/{recovery_count} x {shard_bytes}"
        );
    }

    let expected = reference_recovery::<DefaultRateEncoder<NoSimd>>(
        original_count,
        recovery_count,
        shard_bytes,
        &original,
    );
    let actual = in_place_recovery(
        DefaultRateEncoder::<NoSimd>::in_place_work_bytes(
            original_count,
            recovery_count,
            shard_bytes,
        ),
        |engine, work| {
            DefaultRateEncoder::encode_in_place(
                engine,
                original_count,
                recovery_count,
                shard_bytes,
                work,
            )
        },
        original_count,
        recovery_count,
        shard_bytes,
        &original,
    );
    assert_eq!(
        actual, expected,
        "default rate {original_count}/{recovery_count} x {shard_bytes}"
    );
}

#[test]
fn matches_regular_encoders() {
    let shard_sizes = [2, 8, 32, 62, 64, 66, 128, 1024, 1026];

    for (seed, shard_bytes) in shard_sizes.into_iter().enumerate() {
        let seed = seed as u64;
        check(1, 1, shard_bytes, seed);
        check(3, 2, shard_bytes, seed + 100);
        check(2, 3, shard_bytes, seed + 200);
        check(17, 5, shard_bytes, seed + 300);
        check(5, 17, shard_bytes, seed + 400);
        check(64, 64, shard_bytes, seed + 500);
        check(100, 33, shard_bytes, seed + 600);
    }
}

#[test]
fn in_place_encoding_is_engine_independent() {
    let original_count = 37;
    let recovery_count = 19;
    let shard_bytes = 130;
    let original = original_shards(original_count, shard_bytes, 42);

    let expected = reference_recovery::<HighRateEncoder<NoSimd>>(
        original_count,
        recovery_count,
        shard_bytes,
        &original,
    );

    let work_bytes = HighRateEncoder::<DefaultEngine>::in_place_work_bytes(
        original_count,
        recovery_count,
        shard_bytes,
    );
    let mut work = vec![0u8; work_bytes];
    for (index, shard) in original.iter().enumerate() {
        write_in_place_shard(&mut work, shard_bytes, index, shard).unwrap();
    }
    HighRateEncoder::encode_in_place(
        &DefaultEngine::new(),
        original_count,
        recovery_count,
        shard_bytes,
        &mut work,
    )
    .unwrap();

    for (index, expected) in expected.iter().enumerate() {
        assert_eq!(
            read_in_place_shard(&work, shard_bytes, index).unwrap(),
            expected.as_slice(),
            "recovery shard {index}"
        );
    }
}

/// When `shard_bytes` is a multiple of 64 the caller can lay original shards out itself and read
/// recovery shards straight out of the working buffer.
#[test]
fn multiple_of_64_needs_no_helpers() {
    let (original_count, recovery_count, shard_bytes) = (40, 24, 128);
    let original = original_shards(original_count, shard_bytes, 7);

    assert_eq!(in_place_shard_stride(shard_bytes), shard_bytes);

    let mut work = vec![
        0u8;
        HighRateEncoder::<NoSimd>::in_place_work_bytes(
            original_count,
            recovery_count,
            shard_bytes
        )
    ];
    for (chunk, shard) in work.chunks_exact_mut(shard_bytes).zip(&original) {
        chunk.copy_from_slice(shard);
    }

    HighRateEncoder::encode_in_place(
        &NoSimd::new(),
        original_count,
        recovery_count,
        shard_bytes,
        &mut work,
    )
    .unwrap();

    let expected = reference_recovery::<HighRateEncoder<NoSimd>>(
        original_count,
        recovery_count,
        shard_bytes,
        &original,
    );
    assert_eq!(
        &work[..recovery_count * shard_bytes],
        expected.concat().as_slice()
    );
}

#[test]
fn errors() {
    assert_eq!(
        HighRateEncoder::encode_in_place(&NoSimd::new(), 1, 1, 3, &mut [0u8; 64]).unwrap_err(),
        Error::InvalidShardSize { shard_bytes: 3 }
    );

    let required = HighRateEncoder::<NoSimd>::in_place_work_bytes(4, 4, 64);
    assert_eq!(required, 4 * 64);
    let mut work = vec![0u8; required - 1];
    assert_eq!(
        HighRateEncoder::encode_in_place(&NoSimd::new(), 4, 4, 64, &mut work).unwrap_err(),
        Error::InvalidWorkBufferSize {
            required,
            got: required - 1
        }
    );

    let mut work = vec![0u8; required];
    assert_eq!(
        write_in_place_shard(&mut work, 64, 0, &[0u8; 32]).unwrap_err(),
        Error::DifferentShardSize {
            shard_bytes: 64,
            got: 32
        }
    );
    assert_eq!(
        write_in_place_shard(&mut work, 64, 4, &[0u8; 64]).unwrap_err(),
        Error::InvalidWorkBufferSize {
            required: 5 * 64,
            got: required
        }
    );
    assert_eq!(
        read_in_place_shard(&work, 64, 4).unwrap_err(),
        Error::InvalidWorkBufferSize {
            required: 5 * 64,
            got: required
        }
    );
}

/// Makes sure the naive engine agrees too, so this is not just testing one implementation against
/// itself.
#[test]
fn naive_engine() {
    let (original_count, recovery_count, shard_bytes) = (12, 6, 96);
    let original = original_shards(original_count, shard_bytes, 3);

    let mut work = vec![
        0u8;
        HighRateEncoder::<Naive>::in_place_work_bytes(
            original_count,
            recovery_count,
            shard_bytes
        )
    ];
    for (index, shard) in original.iter().enumerate() {
        write_in_place_shard(&mut work, shard_bytes, index, shard).unwrap();
    }
    HighRateEncoder::encode_in_place(
        &Naive::new(),
        original_count,
        recovery_count,
        shard_bytes,
        &mut work,
    )
    .unwrap();

    let expected = reference_recovery::<HighRateEncoder<NoSimd>>(
        original_count,
        recovery_count,
        shard_bytes,
        &original,
    );
    for (index, expected) in expected.iter().enumerate() {
        assert_eq!(
            read_in_place_shard(&work, shard_bytes, index).unwrap(),
            expected.as_slice()
        );
    }
}

// ======================================================================
// IN-PLACE DECODING

/// Encodes with the regular encoder, then decodes with both the regular decoder and the in-place
/// decoder and makes sure both restore the original shards.
fn check_decode<R>(
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
    missing_original: &[usize],
    present_recovery: &[usize],
    seed: u64,
    work_bytes: usize,
    original_slot: impl Fn(usize) -> usize,
    recovery_slot: impl Fn(usize) -> usize,
    decode: impl FnOnce(&NoSimd, &[bool], &[bool], &mut [u8]) -> Result<(), Error>,
) where
    R: RateEncoder<NoSimd>,
{
    let original = original_shards(original_count, shard_bytes, seed);
    let recovery = reference_recovery::<R>(original_count, recovery_count, shard_bytes, &original);

    let mut original_received = vec![true; original_count];
    for index in missing_original {
        original_received[*index] = false;
    }
    let mut recovery_received = vec![false; recovery_count];
    for index in present_recovery {
        recovery_received[*index] = true;
    }

    let mut work = vec![0u8; work_bytes];
    for (index, shard) in original.iter().enumerate() {
        if original_received[index] {
            write_in_place_shard(&mut work, shard_bytes, original_slot(index), shard).unwrap();
        }
    }
    for (index, shard) in recovery.iter().enumerate() {
        if recovery_received[index] {
            write_in_place_shard(&mut work, shard_bytes, recovery_slot(index), shard).unwrap();
        }
    }

    decode(
        &NoSimd::new(),
        original_received.as_slice(),
        recovery_received.as_slice(),
        &mut work,
    )
    .unwrap();

    for index in missing_original {
        assert_eq!(
            read_in_place_shard(&work, shard_bytes, original_slot(*index)).unwrap(),
            original[*index].as_slice(),
            "restored original shard {index} of {original_count}/{recovery_count} x {shard_bytes}"
        );
    }
}

fn check_decode_all_rates(
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
    missing_original: &[usize],
    present_recovery: &[usize],
    seed: u64,
) {
    if HighRate::<NoSimd>::supports(original_count, recovery_count) {
        check_decode::<HighRateEncoder<NoSimd>>(
            original_count,
            recovery_count,
            shard_bytes,
            missing_original,
            present_recovery,
            seed,
            HighRateDecoder::<NoSimd>::in_place_work_bytes(
                original_count,
                recovery_count,
                shard_bytes,
            ),
            |index| {
                HighRateDecoder::<NoSimd>::in_place_original_slot(
                    original_count,
                    recovery_count,
                    index,
                )
            },
            |index| {
                HighRateDecoder::<NoSimd>::in_place_recovery_slot(
                    original_count,
                    recovery_count,
                    index,
                )
            },
            |engine, original_received, recovery_received, work| {
                HighRateDecoder::decode_in_place(
                    engine,
                    original_count,
                    recovery_count,
                    original_received,
                    recovery_received,
                    shard_bytes,
                    work,
                )
            },
        );
    }

    if LowRate::<NoSimd>::supports(original_count, recovery_count) {
        check_decode::<LowRateEncoder<NoSimd>>(
            original_count,
            recovery_count,
            shard_bytes,
            missing_original,
            present_recovery,
            seed,
            LowRateDecoder::<NoSimd>::in_place_work_bytes(
                original_count,
                recovery_count,
                shard_bytes,
            ),
            |index| {
                LowRateDecoder::<NoSimd>::in_place_original_slot(
                    original_count,
                    recovery_count,
                    index,
                )
            },
            |index| {
                LowRateDecoder::<NoSimd>::in_place_recovery_slot(
                    original_count,
                    recovery_count,
                    index,
                )
            },
            |engine, original_received, recovery_received, work| {
                LowRateDecoder::decode_in_place(
                    engine,
                    original_count,
                    recovery_count,
                    original_received,
                    recovery_received,
                    shard_bytes,
                    work,
                )
            },
        );
    }

    check_decode::<DefaultRateEncoder<NoSimd>>(
        original_count,
        recovery_count,
        shard_bytes,
        missing_original,
        present_recovery,
        seed,
        DefaultRateDecoder::<NoSimd>::in_place_work_bytes(
            original_count,
            recovery_count,
            shard_bytes,
        ),
        |index| {
            DefaultRateDecoder::<NoSimd>::in_place_original_slot(
                original_count,
                recovery_count,
                index,
            )
        },
        |index| {
            DefaultRateDecoder::<NoSimd>::in_place_recovery_slot(
                original_count,
                recovery_count,
                index,
            )
        },
        |engine, original_received, recovery_received, work| {
            DefaultRateDecoder::decode_in_place(
                engine,
                original_count,
                recovery_count,
                original_received,
                recovery_received,
                shard_bytes,
                work,
            )
        },
    );
}

#[test]
fn decode_restores_originals() {
    for (seed, shard_bytes) in [2, 32, 62, 64, 66, 128, 1024, 1026].into_iter().enumerate() {
        let seed = seed as u64;

        check_decode_all_rates(1, 1, shard_bytes, &[0], &[0], seed);
        check_decode_all_rates(3, 2, shard_bytes, &[0, 2], &[0, 1], seed + 100);
        check_decode_all_rates(2, 3, shard_bytes, &[0, 1], &[1, 2], seed + 200);
        check_decode_all_rates(17, 5, shard_bytes, &[1, 4, 9], &[0, 2, 4], seed + 300);
        check_decode_all_rates(
            5,
            17,
            shard_bytes,
            &[0, 1, 2, 3, 4],
            &[3, 5, 7, 11, 13],
            seed + 400,
        );
        check_decode_all_rates(
            64,
            64,
            shard_bytes,
            &(0..32).collect::<Vec<_>>(),
            &(0..32).collect::<Vec<_>>(),
            seed + 500,
        );
        check_decode_all_rates(100, 33, shard_bytes, &[7, 70, 99], &[0, 16, 32], seed + 600);
    }
}

/// Received flags can be given in any `ReceivedShards` representation, bit-packed included.
#[test]
fn decode_with_bit_packed_received() {
    let (original_count, recovery_count, shard_bytes) = (8, 4, 64);
    let original = original_shards(original_count, shard_bytes, 77);
    let recovery = reference_recovery::<HighRateEncoder<NoSimd>>(
        original_count,
        recovery_count,
        shard_bytes,
        &original,
    );

    // Original shards 1 and 5 are missing, recovery shards 0 and 3 were received.
    let mut work = vec![
        0u8;
        HighRateDecoder::<NoSimd>::in_place_work_bytes(
            original_count,
            recovery_count,
            shard_bytes
        )
    ];
    for (index, shard) in original.iter().enumerate() {
        if index != 1 && index != 5 {
            let slot = HighRateDecoder::<NoSimd>::in_place_original_slot(
                original_count,
                recovery_count,
                index,
            );
            write_in_place_shard(&mut work, shard_bytes, slot, shard).unwrap();
        }
    }
    for index in [0, 3] {
        let slot = HighRateDecoder::<NoSimd>::in_place_recovery_slot(
            original_count,
            recovery_count,
            index,
        );
        write_in_place_shard(&mut work, shard_bytes, slot, &recovery[index]).unwrap();
    }

    // Bit `i` of the word is shard `i`.
    let original_received = ReceivedShardBits([0b1101_1101u64]);
    let recovery_received = ReceivedShardBits([0b1001u64]);

    HighRateDecoder::decode_in_place(
        &NoSimd::new(),
        original_count,
        recovery_count,
        original_received,
        recovery_received,
        shard_bytes,
        &mut work,
    )
    .unwrap();

    for index in [1, 5] {
        let slot = HighRateDecoder::<NoSimd>::in_place_original_slot(
            original_count,
            recovery_count,
            index,
        );
        assert_eq!(
            read_in_place_shard(&work, shard_bytes, slot).unwrap(),
            original[index].as_slice()
        );
    }
}

/// Decoding when nothing is missing must succeed and leave received shards alone.
#[test]
fn decode_nothing_missing() {
    let (original_count, recovery_count, shard_bytes) = (8, 4, 64);
    let original = original_shards(original_count, shard_bytes, 11);

    let mut work = vec![
        0u8;
        HighRateDecoder::<NoSimd>::in_place_work_bytes(
            original_count,
            recovery_count,
            shard_bytes
        )
    ];
    for (index, shard) in original.iter().enumerate() {
        let slot = HighRateDecoder::<NoSimd>::in_place_original_slot(
            original_count,
            recovery_count,
            index,
        );
        write_in_place_shard(&mut work, shard_bytes, slot, shard).unwrap();
    }

    HighRateDecoder::decode_in_place(
        &NoSimd::new(),
        original_count,
        recovery_count,
        vec![true; original_count].as_slice(),
        vec![false; recovery_count].as_slice(),
        shard_bytes,
        &mut work,
    )
    .unwrap();

    for (index, shard) in original.iter().enumerate() {
        let slot = HighRateDecoder::<NoSimd>::in_place_original_slot(
            original_count,
            recovery_count,
            index,
        );
        assert_eq!(
            read_in_place_shard(&work, shard_bytes, slot).unwrap(),
            shard.as_slice()
        );
    }
}

#[test]
fn decode_errors() {
    let (original_count, recovery_count, shard_bytes) = (8, 4, 64);
    let required =
        HighRateDecoder::<NoSimd>::in_place_work_bytes(original_count, recovery_count, shard_bytes);

    let mut original_received = vec![true; original_count];
    original_received[0] = false;
    original_received[1] = false;
    let mut recovery_received = vec![false; recovery_count];
    recovery_received[0] = true;

    let mut work = vec![0u8; required];
    assert_eq!(
        HighRateDecoder::decode_in_place(
            &NoSimd::new(),
            original_count,
            recovery_count,
            original_received.as_slice(),
            recovery_received.as_slice(),
            shard_bytes,
            &mut work,
        )
        .unwrap_err(),
        Error::NotEnoughShards {
            original_count,
            original_received_count: 6,
            recovery_received_count: 1,
        }
    );

    recovery_received[1] = true;
    let mut work = vec![0u8; required - 1];
    assert_eq!(
        HighRateDecoder::decode_in_place(
            &NoSimd::new(),
            original_count,
            recovery_count,
            original_received.as_slice(),
            recovery_received.as_slice(),
            shard_bytes,
            &mut work,
        )
        .unwrap_err(),
        Error::InvalidWorkBufferSize {
            required,
            got: required - 1,
        }
    );
}

/// The regular decoder and the in-place decoder must agree, including on a `DefaultEngine`.
#[test]
fn decode_matches_regular_decoder() {
    let (original_count, recovery_count, shard_bytes) = (37, 19, 130);
    let original = original_shards(original_count, shard_bytes, 5);
    let recovery = reference_recovery::<HighRateEncoder<NoSimd>>(
        original_count,
        recovery_count,
        shard_bytes,
        &original,
    );

    let missing = [0, 5, 6, 7, 20, 36];

    let expected = {
        let mut decoder = HighRateDecoder::new(
            original_count,
            recovery_count,
            shard_bytes,
            NoSimd::new(),
            None,
        )
        .unwrap();
        for (index, shard) in original.iter().enumerate() {
            if !missing.contains(&index) {
                decoder.add_original_shard(index, shard).unwrap();
            }
        }
        for (index, shard) in recovery.iter().enumerate().take(missing.len()) {
            decoder.add_recovery_shard(index, shard).unwrap();
        }
        let result = decoder.decode().unwrap();
        let expected = missing
            .iter()
            .map(|index| result.restored_original(*index).unwrap().to_vec())
            .collect::<Vec<_>>();
        expected
    };

    let mut original_received = vec![true; original_count];
    for index in missing {
        original_received[index] = false;
    }
    let mut recovery_received = vec![false; recovery_count];
    for received in recovery_received.iter_mut().take(missing.len()) {
        *received = true;
    }

    let mut work = vec![
        0u8;
        HighRateDecoder::<DefaultEngine>::in_place_work_bytes(
            original_count,
            recovery_count,
            shard_bytes
        )
    ];
    for (index, shard) in original.iter().enumerate() {
        if original_received[index] {
            let slot = HighRateDecoder::<DefaultEngine>::in_place_original_slot(
                original_count,
                recovery_count,
                index,
            );
            write_in_place_shard(&mut work, shard_bytes, slot, shard).unwrap();
        }
    }
    for (index, shard) in recovery.iter().enumerate() {
        if recovery_received[index] {
            let slot = HighRateDecoder::<DefaultEngine>::in_place_recovery_slot(
                original_count,
                recovery_count,
                index,
            );
            write_in_place_shard(&mut work, shard_bytes, slot, shard).unwrap();
        }
    }

    HighRateDecoder::decode_in_place(
        &DefaultEngine::new(),
        original_count,
        recovery_count,
        original_received.as_slice(),
        recovery_received.as_slice(),
        shard_bytes,
        &mut work,
    )
    .unwrap();

    for (expected, index) in expected.iter().zip(missing) {
        let slot = HighRateDecoder::<DefaultEngine>::in_place_original_slot(
            original_count,
            recovery_count,
            index,
        );
        assert_eq!(
            read_in_place_shard(&work, shard_bytes, slot).unwrap(),
            expected.as_slice(),
            "restored original shard {index}"
        );
        assert_eq!(expected.as_slice(), original[index].as_slice());
    }
}

/// The regular low rate decoder API must keep working, now that it shares its implementation
/// with in-place decoding.
#[test]
fn low_rate_decoder_roundtrip() {
    let (original_count, recovery_count, shard_bytes) = (5, 17, 64);
    let original = original_shards(original_count, shard_bytes, 9);
    let recovery = reference_recovery::<LowRateEncoder<NoSimd>>(
        original_count,
        recovery_count,
        shard_bytes,
        &original,
    );

    let mut decoder = LowRateDecoder::new(
        original_count,
        recovery_count,
        shard_bytes,
        NoSimd::new(),
        None,
    )
    .unwrap();
    for (index, shard) in recovery.iter().enumerate().take(original_count) {
        decoder.add_recovery_shard(index, shard).unwrap();
    }
    let result = decoder.decode().unwrap();
    for (index, shard) in original.iter().enumerate() {
        assert_eq!(result.restored_original(index).unwrap(), shard.as_slice());
    }
}

/// The working buffer size must be usable in const context, so that callers with statically known
/// shard counts can put it on the stack or size a heap allocation at compile time.
#[test]
fn work_bytes_is_const() {
    const ORIGINAL_COUNT: usize = 4;
    const RECOVERY_COUNT: usize = 4;
    const SHARD_BYTES: usize = 64;
    const ENCODE_WORK_BYTES: usize =
        HighRateEncoder::<NoSimd>::in_place_work_bytes(ORIGINAL_COUNT, RECOVERY_COUNT, SHARD_BYTES);
    const DECODE_WORK_BYTES: usize =
        HighRateDecoder::<NoSimd>::in_place_work_bytes(ORIGINAL_COUNT, RECOVERY_COUNT, SHARD_BYTES);
    const DEFAULT_ENCODE_WORK_BYTES: usize = DefaultRateEncoder::<NoSimd>::in_place_work_bytes(
        ORIGINAL_COUNT,
        RECOVERY_COUNT,
        SHARD_BYTES,
    );
    const RECOVERY_SLOT: usize =
        HighRateDecoder::<NoSimd>::in_place_recovery_slot(ORIGINAL_COUNT, RECOVERY_COUNT, 1);
    const ORIGINAL_SLOT: usize =
        HighRateDecoder::<NoSimd>::in_place_original_slot(ORIGINAL_COUNT, RECOVERY_COUNT, 1);

    assert_eq!(ENCODE_WORK_BYTES, 4 * 64);
    assert_eq!(DECODE_WORK_BYTES, 8 * 64);
    assert_eq!(DEFAULT_ENCODE_WORK_BYTES, 4 * 64);
    assert_eq!(RECOVERY_SLOT, 1);
    assert_eq!(ORIGINAL_SLOT, 5);

    // Working buffer on the stack, no allocation anywhere
    let original = original_shards(ORIGINAL_COUNT, SHARD_BYTES, 1);
    let mut work = [0u8; ENCODE_WORK_BYTES];
    for (chunk, shard) in work.chunks_exact_mut(SHARD_BYTES).zip(&original) {
        chunk.copy_from_slice(shard);
    }
    HighRateEncoder::encode_in_place(
        &NoSimd::new(),
        ORIGINAL_COUNT,
        RECOVERY_COUNT,
        SHARD_BYTES,
        &mut work,
    )
    .unwrap();

    let expected = reference_recovery::<HighRateEncoder<NoSimd>>(
        ORIGINAL_COUNT,
        RECOVERY_COUNT,
        SHARD_BYTES,
        &original,
    );
    assert_eq!(&work[..], expected.concat().as_slice());
}

// ======================================================================
// CUSTOM SHARD STORAGE

/// [`ShardStorage`] which keeps unused chunks in between the shards, meaning that its layout is
/// deliberately not the flat layout that a `&mut [u8]` working buffer has.
struct PaddedShards {
    shard_count: usize,
    shard_len_64: usize,
    padding_64: usize,
    data: Vec<[u8; 64]>,
}

impl PaddedShards {
    fn new(shard_count: usize, shard_bytes: usize, padding_64: usize) -> Self {
        assert_eq!(shard_bytes % 64, 0);

        let shard_len_64 = shard_bytes / 64;

        Self {
            shard_count,
            shard_len_64,
            padding_64,
            data: vec![[0; 64]; shard_count * (shard_len_64 + padding_64)],
        }
    }

    fn stride(&self) -> usize {
        self.shard_len_64 + self.padding_64
    }

    fn write_shard(&mut self, index: usize, shard: &[u8]) {
        self[index].as_flattened_mut().copy_from_slice(shard);
    }

    fn read_shard(&self, index: usize) -> &[u8] {
        self[index].as_flattened()
    }
}

impl Index<usize> for PaddedShards {
    type Output = [[u8; 64]];
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index * self.stride()..][..self.shard_len_64]
    }
}

impl IndexMut<usize> for PaddedShards {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let stride = self.stride();
        &mut self.data[index * stride..][..self.shard_len_64]
    }
}

// SAFETY: Shards are stored at a fixed stride, hence distinct indices refer to distinct shards of
// equal length, and `dist2_mut()`/`dist4_mut()` return the requested shards.
unsafe impl ShardStorage for PaddedShards {
    fn len(&self) -> usize {
        self.shard_count
    }

    unsafe fn dist2_mut(&mut self, pos: usize, dist: usize) -> (&mut [[u8; 64]], &mut [[u8; 64]]) {
        let (stride, shard_len_64) = (self.stride(), self.shard_len_64);
        let (head, tail) = self.data.split_at_mut((pos + dist) * stride);

        (
            &mut head[pos * stride..][..shard_len_64],
            &mut tail[..shard_len_64],
        )
    }

    unsafe fn dist4_mut(
        &mut self,
        pos: usize,
        dist: usize,
    ) -> (
        &mut [[u8; 64]],
        &mut [[u8; 64]],
        &mut [[u8; 64]],
        &mut [[u8; 64]],
    ) {
        let (stride, shard_len_64) = (self.stride(), self.shard_len_64);
        let (head, tail) = self.data.split_at_mut((pos + dist * 2) * stride);
        let (a, b) = head.split_at_mut((pos + dist) * stride);
        let (c, d) = tail.split_at_mut(dist * stride);

        (
            &mut a[pos * stride..][..shard_len_64],
            &mut b[..shard_len_64],
            &mut c[..shard_len_64],
            &mut d[..shard_len_64],
        )
    }

    fn zero(&mut self, range: impl RangeBounds<usize>) {
        let start = match range.start_bound() {
            Bound::Included(start) => *start,
            Bound::Excluded(start) => start + 1,
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(end) => end + 1,
            Bound::Excluded(end) => *end,
            Bound::Unbounded => self.shard_count,
        };

        for index in start..end {
            self[index].fill([0; 64]);
        }
    }
}

/// The in-place APIs must produce the same results with a shard storage of the caller's own as
/// they do with a flat working buffer.
#[test]
fn custom_shard_storage_roundtrip() {
    let (original_count, recovery_count, shard_bytes) = (8, 4, 128);
    let original = original_shards(original_count, shard_bytes, 42);
    let expected = reference_recovery::<HighRateEncoder<NoSimd>>(
        original_count,
        recovery_count,
        shard_bytes,
        &original,
    );

    // ENCODE

    let stride = in_place_shard_stride(shard_bytes);
    let work_count =
        HighRateEncoder::<NoSimd>::in_place_work_bytes(original_count, recovery_count, shard_bytes)
            / stride;
    let mut work = PaddedShards::new(work_count, shard_bytes, 3);

    for (index, shard) in original.iter().enumerate() {
        work.write_shard(index, shard);
    }

    HighRateEncoder::encode_in_place(
        &NoSimd::new(),
        original_count,
        recovery_count,
        shard_bytes,
        &mut work,
    )
    .unwrap();

    let recovery: Vec<Vec<u8>> = (0..recovery_count)
        .map(|index| work.read_shard(index).to_vec())
        .collect();
    assert_eq!(recovery, expected);

    // DECODE

    let work_count =
        HighRateDecoder::<NoSimd>::in_place_work_bytes(original_count, recovery_count, shard_bytes)
            / stride;
    let mut work = PaddedShards::new(work_count, shard_bytes, 1);

    let missing = [1, 5, 6];
    for (index, shard) in original.iter().enumerate() {
        if !missing.contains(&index) {
            let slot = HighRateDecoder::<NoSimd>::in_place_original_slot(
                original_count,
                recovery_count,
                index,
            );
            work.write_shard(slot, shard);
        }
    }
    for index in 0..missing.len() {
        let slot = HighRateDecoder::<NoSimd>::in_place_recovery_slot(
            original_count,
            recovery_count,
            index,
        );
        work.write_shard(slot, &recovery[index]);
    }

    let original_received: Vec<bool> = (0..original_count)
        .map(|index| !missing.contains(&index))
        .collect();
    let recovery_received: Vec<bool> = (0..recovery_count)
        .map(|index| index < missing.len())
        .collect();

    HighRateDecoder::decode_in_place(
        &NoSimd::new(),
        original_count,
        recovery_count,
        original_received.as_slice(),
        recovery_received.as_slice(),
        shard_bytes,
        &mut work,
    )
    .unwrap();

    for index in missing {
        let slot = HighRateDecoder::<NoSimd>::in_place_original_slot(
            original_count,
            recovery_count,
            index,
        );
        assert_eq!(work.read_shard(slot), original[index].as_slice());
    }
}

/// A working buffer that is too small must be rejected for a custom shard storage too.
#[test]
fn custom_shard_storage_too_small() {
    let (original_count, recovery_count, shard_bytes) = (8, 4, 128);
    let stride = in_place_shard_stride(shard_bytes);
    let required =
        HighRateEncoder::<NoSimd>::in_place_work_bytes(original_count, recovery_count, shard_bytes);

    let mut work = PaddedShards::new(required / stride - 1, shard_bytes, 2);

    assert_eq!(
        HighRateEncoder::encode_in_place(
            &NoSimd::new(),
            original_count,
            recovery_count,
            shard_bytes,
            &mut work,
        )
        .unwrap_err(),
        Error::InvalidWorkBufferSize {
            required,
            got: required - stride
        }
    );
}
