//! Breakdown of where time is spent in `HighRateEncoder`/`HighRateDecoder` and how in-place
//! encoding/decoding with a re-used working buffer compares to it.
//!
//! Shard shapes are the two extremes: many tiny shards and few large shards.
use reed_solomon_simd::engine::DefaultEngine;
use reed_solomon_simd::rate::{
    read_in_place_shard, write_in_place_shard, HighRateDecoder, HighRateEncoder, RateDecoder,
    RateEncoder,
};
use std::time::Instant;

fn bench(name: &str, original_count: usize, recovery_count: usize, shard_bytes: usize) {
    let source = vec![vec![7u8; shard_bytes]; original_count];
    let mut parity = vec![vec![0u8; shard_bytes]; recovery_count];

    let rounds = if shard_bytes > 4096 { 5 } else { 50 };

    // Phase 1: construction (allocation + zeroing of the work buffer)
    let mut t_new = 0.0;
    let mut t_add = 0.0;
    let mut t_encode = 0.0;
    let mut t_out = 0.0;

    for _ in 0..rounds {
        let start = Instant::now();
        let mut encoder = HighRateEncoder::new(
            original_count,
            recovery_count,
            shard_bytes,
            DefaultEngine::new(),
            None,
        )
        .unwrap();
        t_new += start.elapsed().as_secs_f64();

        let start = Instant::now();
        for shard in &source {
            encoder.add_original_shard(shard).unwrap();
        }
        t_add += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let result = encoder.encode().unwrap();
        t_encode += start.elapsed().as_secs_f64();

        let start = Instant::now();
        for (input, output) in result.recovery_iter().zip(parity.iter_mut()) {
            output.copy_from_slice(input);
        }
        t_out += start.elapsed().as_secs_f64();
    }

    let ms = |t: f64| t / rounds as f64 * 1000.0;
    let total = ms(t_new) + ms(t_add) + ms(t_encode) + ms(t_out);
    println!(
        "{name}: new {:.3} ms, add_original {:.3} ms, encode {:.3} ms, copy_out {:.3} ms, total \
        {:.3} ms",
        ms(t_new),
        ms(t_add),
        ms(t_encode),
        ms(t_out),
        total,
    );
}

fn bench_reuse(name: &str, original_count: usize, recovery_count: usize, shard_bytes: usize) {
    let source = vec![vec![7u8; shard_bytes]; original_count];
    let mut parity = vec![vec![0u8; shard_bytes]; recovery_count];

    let rounds = if shard_bytes > 4096 { 5 } else { 50 };

    let mut work = None;
    let mut t_new = 0.0;
    let mut t_add = 0.0;
    let mut t_encode = 0.0;
    let mut t_out = 0.0;

    for _ in 0..rounds {
        let start = Instant::now();
        let mut encoder = HighRateEncoder::new(
            original_count,
            recovery_count,
            shard_bytes,
            DefaultEngine::new(),
            work.take(),
        )
        .unwrap();
        t_new += start.elapsed().as_secs_f64();

        let start = Instant::now();
        for shard in &source {
            encoder.add_original_shard(shard).unwrap();
        }
        t_add += start.elapsed().as_secs_f64();

        let start = Instant::now();
        {
            let result = encoder.encode().unwrap();
            t_encode += start.elapsed().as_secs_f64();

            let start = Instant::now();
            for (input, output) in result.recovery_iter().zip(parity.iter_mut()) {
                output.copy_from_slice(input);
            }
            t_out += start.elapsed().as_secs_f64();
        }
        work = Some(encoder.into_parts().1);
    }

    let ms = |t: f64| t / rounds as f64 * 1000.0;
    let total = ms(t_new) + ms(t_add) + ms(t_encode) + ms(t_out);
    println!(
        "{name} (reused work): new {:.3} ms, add_original {:.3} ms, encode {:.3} ms, copy_out \
        {:.3} ms, total {:.3} ms",
        ms(t_new),
        ms(t_add),
        ms(t_encode),
        ms(t_out),
        total,
    );
}

/// In-place: caller owns the working buffer, copies originals in and recovery out itself.
fn bench_in_place(name: &str, original_count: usize, recovery_count: usize, shard_bytes: usize) {
    let source = vec![vec![7u8; shard_bytes]; original_count];
    let mut parity = vec![vec![0u8; shard_bytes]; recovery_count];

    let rounds = if shard_bytes > 4096 { 5 } else { 50 };

    let engine = DefaultEngine::new();
    let work_bytes = HighRateEncoder::<DefaultEngine>::in_place_work_bytes(
        original_count,
        recovery_count,
        shard_bytes,
    );
    let mut work = vec![0u8; work_bytes];

    let mut t_add = 0.0;
    let mut t_encode = 0.0;
    let mut t_out = 0.0;

    for _ in 0..rounds {
        let start = Instant::now();
        for (index, shard) in source.iter().enumerate() {
            write_in_place_shard(&mut work, shard_bytes, index, shard).unwrap();
        }
        t_add += start.elapsed().as_secs_f64();

        let start = Instant::now();
        HighRateEncoder::encode_in_place(
            &engine,
            original_count,
            recovery_count,
            shard_bytes,
            &mut work,
        )
        .unwrap();
        t_encode += start.elapsed().as_secs_f64();

        let start = Instant::now();
        for (index, output) in parity.iter_mut().enumerate() {
            output.copy_from_slice(read_in_place_shard(&work, shard_bytes, index).unwrap());
        }
        t_out += start.elapsed().as_secs_f64();
    }

    let ms = |t: f64| t / rounds as f64 * 1000.0;
    let total = ms(t_add) + ms(t_encode) + ms(t_out);
    println!(
        "{name} (in-place):    new 0.000 ms, add_original {:.3} ms, encode {:.3} ms, copy_out \
        {:.3} ms, total {:.3} ms",
        ms(t_add),
        ms(t_encode),
        ms(t_out),
        total,
    );
}

/// Fully in-place: originals are produced directly in the working buffer and recovery shards are
/// consumed from it, no copies at all. Only possible when `shard_bytes % 64 == 0`.
fn bench_in_place_zero_copy(
    name: &str,
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
) {
    assert_eq!(shard_bytes % 64, 0);

    let rounds = if shard_bytes > 4096 { 5 } else { 50 };

    let engine = DefaultEngine::new();
    let work_bytes = HighRateEncoder::<DefaultEngine>::in_place_work_bytes(
        original_count,
        recovery_count,
        shard_bytes,
    );
    let mut work = vec![7u8; work_bytes];

    let mut t_encode = 0.0;

    for _ in 0..rounds {
        let start = Instant::now();
        HighRateEncoder::encode_in_place(
            &engine,
            original_count,
            recovery_count,
            shard_bytes,
            &mut work,
        )
        .unwrap();
        t_encode += start.elapsed().as_secs_f64();
    }

    let ms = |t: f64| t / rounds as f64 * 1000.0;
    println!(
        "{name} (in-place, zero copy): encode = total {:.3} ms",
        ms(t_encode)
    );
}

/// Decoder-side breakdown: half of the original shards missing, recovered from recovery shards.
fn bench_decode(name: &str, original_count: usize, recovery_count: usize, shard_bytes: usize) {
    let source = vec![vec![7u8; shard_bytes]; original_count];
    let mut parity = vec![vec![0u8; shard_bytes]; recovery_count];
    {
        let mut encoder = HighRateEncoder::new(
            original_count,
            recovery_count,
            shard_bytes,
            DefaultEngine::new(),
            None,
        )
        .unwrap();
        for shard in &source {
            encoder.add_original_shard(shard).unwrap();
        }
        let result = encoder.encode().unwrap();
        for (input, output) in result.recovery_iter().zip(parity.iter_mut()) {
            output.copy_from_slice(input);
        }
    }

    let missing = original_count / 2;
    let rounds = if shard_bytes > 4096 { 5 } else { 20 };

    let mut t_new = 0.0;
    let mut t_add = 0.0;
    let mut t_decode = 0.0;
    let mut t_out = 0.0;
    let mut restored = vec![vec![0u8; shard_bytes]; missing];

    for _ in 0..rounds {
        let start = Instant::now();
        let mut decoder = HighRateDecoder::new(
            original_count,
            recovery_count,
            shard_bytes,
            DefaultEngine::new(),
            None,
        )
        .unwrap();
        t_new += start.elapsed().as_secs_f64();

        let start = Instant::now();
        for (index, shard) in source.iter().enumerate().skip(missing) {
            decoder.add_original_shard(index, shard).unwrap();
        }
        for (index, shard) in parity.iter().enumerate().take(missing) {
            decoder.add_recovery_shard(index, shard).unwrap();
        }
        t_add += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let result = decoder.decode().unwrap();
        t_decode += start.elapsed().as_secs_f64();

        let start = Instant::now();
        for (index, output) in restored.iter_mut().enumerate() {
            output.copy_from_slice(result.restored_original(index).unwrap());
        }
        t_out += start.elapsed().as_secs_f64();
    }

    let ms = |t: f64| t / rounds as f64 * 1000.0;
    let total = ms(t_new) + ms(t_add) + ms(t_decode) + ms(t_out);
    println!(
        "{name} DECODE: new {:.3} ms, add_shards {:.3} ms, decode {:.3} ms, copy_out {:.3} ms, \
        total {:.3} ms",
        ms(t_new),
        ms(t_add),
        ms(t_decode),
        ms(t_out),
        total,
    );
}

/// In-place decoding with a reusable working buffer.
fn bench_decode_in_place(
    name: &str,
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
) {
    let source = vec![vec![7u8; shard_bytes]; original_count];
    let mut parity = vec![vec![0u8; shard_bytes]; recovery_count];
    {
        let mut encoder = HighRateEncoder::new(
            original_count,
            recovery_count,
            shard_bytes,
            DefaultEngine::new(),
            None,
        )
        .unwrap();
        for shard in &source {
            encoder.add_original_shard(shard).unwrap();
        }
        let result = encoder.encode().unwrap();
        for (input, output) in result.recovery_iter().zip(parity.iter_mut()) {
            output.copy_from_slice(input);
        }
    }

    let missing = original_count / 2;
    let rounds = if shard_bytes > 4096 { 5 } else { 20 };

    let engine = DefaultEngine::new();
    let mut work = vec![
        0u8;
        HighRateDecoder::<DefaultEngine>::in_place_work_bytes(
            original_count,
            recovery_count,
            shard_bytes
        )
    ];

    let mut original_received = vec![true; original_count];
    for received in original_received.iter_mut().take(missing) {
        *received = false;
    }
    let mut recovery_received = vec![false; recovery_count];
    for received in recovery_received.iter_mut().take(missing) {
        *received = true;
    }

    let mut restored = vec![vec![0u8; shard_bytes]; missing];
    let mut t_add = 0.0;
    let mut t_decode = 0.0;
    let mut t_out = 0.0;

    for _ in 0..rounds {
        let start = Instant::now();
        for (index, shard) in source.iter().enumerate().skip(missing) {
            let slot = HighRateDecoder::<DefaultEngine>::in_place_original_slot(
                original_count,
                recovery_count,
                index,
            );
            write_in_place_shard(&mut work, shard_bytes, slot, shard).unwrap();
        }
        for (index, shard) in parity.iter().enumerate().take(missing) {
            let slot = HighRateDecoder::<DefaultEngine>::in_place_recovery_slot(
                original_count,
                recovery_count,
                index,
            );
            write_in_place_shard(&mut work, shard_bytes, slot, shard).unwrap();
        }
        t_add += start.elapsed().as_secs_f64();

        let start = Instant::now();
        HighRateDecoder::decode_in_place(
            &engine,
            original_count,
            recovery_count,
            original_received.as_slice(),
            recovery_received.as_slice(),
            shard_bytes,
            &mut work,
        )
        .unwrap();
        t_decode += start.elapsed().as_secs_f64();

        let start = Instant::now();
        for (index, output) in restored.iter_mut().enumerate() {
            let slot = HighRateDecoder::<DefaultEngine>::in_place_original_slot(
                original_count,
                recovery_count,
                index,
            );
            output.copy_from_slice(read_in_place_shard(&work, shard_bytes, slot).unwrap());
        }
        t_out += start.elapsed().as_secs_f64();
    }

    let ms = |t: f64| t / rounds as f64 * 1000.0;
    let total = ms(t_add) + ms(t_decode) + ms(t_out);
    println!(
        "{name} DECODE (in-place): new 0.000 ms, add_shards {:.3} ms, decode {:.3} ms, copy_out \
        {:.3} ms, total {:.3} ms",
        ms(t_add),
        ms(t_decode),
        ms(t_out),
        total,
    );
}

fn main() {
    bench("record  32768x32B", 32768, 32768, 32);
    bench("segment   128x1MiB", 128, 128, 1024 * 1024);
    bench_reuse("record  32768x32B", 32768, 32768, 32);
    bench_reuse("segment   128x1MiB", 128, 128, 1024 * 1024);
    bench_in_place("record  32768x32B", 32768, 32768, 32);
    bench_in_place("segment   128x1MiB", 128, 128, 1024 * 1024);
    bench_in_place_zero_copy("segment   128x1MiB", 128, 128, 1024 * 1024);
    bench_decode("record  32768x32B", 32768, 32768, 32);
    bench_decode("segment   128x1MiB", 128, 128, 1024 * 1024);
    bench_decode_in_place("record  32768x32B", 32768, 32768, 32);
    bench_decode_in_place("segment   128x1MiB", 128, 128, 1024 * 1024);
}
