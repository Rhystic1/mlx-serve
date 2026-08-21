//! Degenerate-tail loop detection.
//!
//! Pure token analysis: the exact-cycle, long-cycle and near-repeat tiers that
//! decide whether a generation has stopped making progress. Split out of
//! generate.zig for the Windows/Linux port, because it is NOT MLX-specific --
//! a llama.cpp generation loops exactly as readily as an MLX one, and stubbing
//! this out would ship a server with no loop-stop guard at all.
//!
//! A cut is a TRUNCATION: `finish_reason` stays "length" and the cause rides
//! beside it (`finish_details`); see scheduler.loopStopReason.
//!
//! The bars err toward ACQUITTAL by design. A false cut destroys good work; a
//! missed loop merely runs to max_tokens.

const std = @import("std");

pub const degenerate_loop_reps: usize = 16;
// Tier 2 (2026-08-02 shooter wrap-up class): a two-sentence cycle of ~58
// tokens repeated 26 times evaded the 8-token tier. Long periods demand
// fewer reps — 10 verbatim repetitions of a 9..64-token cycle is
// degeneration with overwhelming probability (identical long lines in real
// code repeat a handful of times, not ten).
pub const degenerate_loop_long_max_period: usize = 64;
pub const degenerate_loop_long_reps: usize = 10;

/// Longest EXACT cycle scanned by the first tier.
pub const degenerate_loop_max_period: usize = 8;

// Tier 3 (2026-08-04 agent-traffic class): a restatement loop that VARIES its
// phrasing has no exact cycle at any period, so both tiers above are blind by
// construction. What it does have is a long stretch of output that recycles a
// tiny vocabulary and introduces almost no new n-grams.
//
// Everything about this tier is deliberately reluctant, because unlike the
// exact tiers it is a fuzzy judgement and a false cut truncates a real answer:
// the window is LONG (a legitimate repetitive passage — a table, a block of
// near-identical code, a list scaffold — is finite and ends well inside it),
// the two ratios must BOTH be low (either one alone convicts honest output —
// a numeric table has few distinct tokens, a repeated code scaffold has few
// distinct n-grams), and the window is longer than the exact tiers' reach so
// this tier only ever speaks about spans they have already declined.
pub const near_repeat_window: usize = 1024;
pub const near_repeat_ngram: usize = 4;
pub const near_repeat_max_ngram_ratio: f32 = 0.35;
pub const near_repeat_max_token_ratio: f32 = 0.12;
/// Third ratio (2026-08-05): how much of the window's SECOND half is n-grams
/// its first half never had. Measured across both shapes at the shipped
/// window: restatement loops 0.019-0.022, healthy repetitive output
/// 0.298-0.827 (dense procedural scene code, a markdown table, the voxel
/// artifact that was wrongly cut). 0.10 sits ~4.5x above the loops and ~3x
/// below the closest healthy case.
pub const near_repeat_max_novelty: f32 = 0.10;

/// Open-addressed distinct-counter sized for one window. Stack-resident, so
/// the whole tier is allocation-free and runs in O(window) per decode tick.
fn DistinctSet(comptime cap: usize) type {
    return struct {
        const Self = @This();
        const empty_key: u64 = std.math.maxInt(u64);
        keys: [cap]u64 = @splat(empty_key),
        n: usize = 0,

        /// True when `key` is already present. Read-only; used to ask whether
        /// the window's second half is introducing anything its first half
        /// did not have.
        fn contains(self: *const Self, key: u64) bool {
            const k = if (key == empty_key) 0 else key;
            var i: usize = @intCast(std.hash.Wyhash.hash(0, std.mem.asBytes(&k)) % cap);
            while (true) {
                if (self.keys[i] == empty_key) return false;
                if (self.keys[i] == k) return true;
                i = (i + 1) % cap;
            }
        }

        /// True when `key` was not already present.
        fn insert(self: *Self, key: u64) bool {
            // maxInt is the empty sentinel; fold the one colliding key onto 0.
            const k = if (key == empty_key) 0 else key;
            var i: usize = @intCast(std.hash.Wyhash.hash(0, std.mem.asBytes(&k)) % cap);
            while (true) {
                if (self.keys[i] == empty_key) {
                    self.keys[i] = k;
                    self.n += 1;
                    return true;
                }
                if (self.keys[i] == k) return false;
                i = (i + 1) % cap;
            }
        }
    };
}

/// The two-ratio judgement over ONE window-sized span. Split out so the trim
/// search can slide the same window backwards without re-deriving the rule.
fn nearRepeatWindowIsDegenerate(window: []const u32) bool {
    // Load factor 0.5 keeps the linear probe short even when every entry is
    // distinct (the healthy case, which is also the hot one).
    var toks = DistinctSet(near_repeat_window * 2){};
    for (window) |t| _ = toks.insert(t);
    const token_ratio = @as(f32, @floatFromInt(toks.n)) / @as(f32, @floatFromInt(window.len));
    if (token_ratio > near_repeat_max_token_ratio) return false;

    var grams = DistinctSet(near_repeat_window * 2){};
    var i: usize = 0;
    while (i + near_repeat_ngram <= window.len) : (i += 1) {
        _ = grams.insert(gramHash(window[i .. i + near_repeat_ngram]));
    }
    const n_gram_positions = window.len - near_repeat_ngram + 1;
    const gram_ratio = @as(f32, @floatFromInt(grams.n)) / @as(f32, @floatFromInt(n_gram_positions));
    if (gram_ratio > near_repeat_max_ngram_ratio) return false;

    // Third ratio: is the window still PROGRESSING? Both ratios above are
    // properties of a vocabulary, and procedurally generated code has the same
    // vocabulary profile as a loop — a fixed call template plus a small colour
    // palette (live 2026-08-05: a voxel scene cut at 16241 tokens, the user got
    // no file at all). What a loop does NOT do is keep introducing material:
    // measured, a restatement loop's second half brings 1.9-2.2% n-grams its
    // first half never had, while healthy repetitive output brings 29.8-82.7%.
    // Requiring all THREE keeps the tier's reluctance in the direction that
    // matters — a missed loop still ends at max_tokens, a false cut destroys
    // work that was going fine.
    var first_half = DistinctSet(near_repeat_window * 2){};
    const mid = window.len / 2;
    var fi: usize = 0;
    while (fi + near_repeat_ngram <= mid) : (fi += 1) {
        _ = first_half.insert(gramHash(window[fi .. fi + near_repeat_ngram]));
    }
    var second_half = DistinctSet(near_repeat_window * 2){};
    var novel: usize = 0;
    var distinct_second: usize = 0;
    var si: usize = mid;
    while (si + near_repeat_ngram <= window.len) : (si += 1) {
        const h = gramHash(window[si .. si + near_repeat_ngram]);
        if (!second_half.insert(h)) continue; // count each distinct gram once
        distinct_second += 1;
        if (!first_half.contains(h)) novel += 1;
    }
    if (distinct_second == 0) return true; // nothing new because there is nothing
    const novelty = @as(f32, @floatFromInt(novel)) / @as(f32, @floatFromInt(distinct_second));
    return novelty <= near_repeat_max_novelty;
}

/// One hash for both the ratio pass and the novelty pass — two spellings of
/// this would silently compare different things.
fn gramHash(gram: []const u32) u64 {
    var h: u64 = 0;
    for (gram) |t| h = h *% 0x100000001b3 ^ t;
    return h;
}

/// Detect a NEAR-repeat tail loop: the last `near_repeat_window` tokens keep
/// restating the same thing in slightly different words. Pure; reads only the
/// tail, so cost is independent of total generated length.
pub fn isNearRepeatTailLoop(tokens: []const u32) bool {
    if (tokens.len < near_repeat_window) return false;
    return nearRepeatWindowIsDegenerate(tokens[tokens.len - near_repeat_window ..]);
}

/// How far back the near-repeat trim search steps, and how far it may reach.
/// The window is judged whole at each stop, so the set stays window-sized
/// however long the loop ran — sizing a set to the FULL span instead would
/// put tens of KB on the inference thread's stack.
pub const near_repeat_step: usize = 128;
pub const near_repeat_max_lookback: usize = 8192;

/// A convicted degenerate tail: which tier saw it, and where the degenerate
/// span BEGINS in the generated ids.
///
/// The start is what makes a loop cut recoverable. An agent client re-sends
/// the cut turn's content as history, the model reads its own loop back and
/// resumes it — five loop-stops in a row, each firing sooner than the last
/// (live 2026-08-05, under pi). Emitting only the prefix means
/// the loop cannot round-trip into the next prompt.
pub const DegenerateTail = struct {
    tier: Tier,
    /// First index of the degenerate span; `tokens[0..start]` is what a
    /// client should be shown. For the exact tiers ONE copy of the cycle is
    /// deliberately kept — the truncated answer should still show what the
    /// model was doing when it got stuck, and one copy cannot sustain a loop.
    start: usize,

    pub const Tier = enum { exact_cycle, long_cycle, near_repeat };
};

/// The smallest period in `[min_period, max_period]` whose cycle repeats
/// `reps` times at the tail, or null. `isDegenerateTailLoopRange` is this
/// predicate — one implementation, so the detector and the trim can never
/// disagree about what was convicted.
pub fn exactCyclePeriod(tokens: []const u32, min_period: usize, max_period: usize, reps: usize) ?usize {
    if (max_period == 0 or reps < 2) return null;
    var p: usize = @max(min_period, 1);
    while (p <= max_period) : (p += 1) {
        const span = p * reps;
        if (tokens.len < span) continue;
        const tail = tokens[tokens.len - span ..];
        var periodic = true;
        var i: usize = p;
        while (i < tail.len) : (i += 1) {
            if (tail[i] != tail[i - p]) {
                periodic = false;
                break;
            }
        }
        if (periodic) return p;
    }
    return null;
}

/// Walk the period-`p` cycle backwards past the `reps` that convicted it: a
/// loop that ran 200 times must be trimmed at 200, not at the threshold.
fn trailingCycleStart(tokens: []const u32, p: usize) usize {
    var start = tokens.len - p;
    while (start >= p) {
        if (!std.mem.eql(u32, tokens[start - p .. start], tokens[start .. start + p])) break;
        start -= p;
    }
    return start; // first index of the FIRST copy of the cycle
}

/// Convict a degenerate tail and say where it starts. Tier order matches
/// `scheduler.loopStopReason`: the exact tiers speak first, and the fuzzy
/// near-repeat tier only ever judges spans they have already declined.
pub fn degenerateTail(tokens: []const u32) ?DegenerateTail {
    if (exactCyclePeriod(tokens, 1, degenerate_loop_max_period, degenerate_loop_reps)) |p| {
        return .{ .tier = .exact_cycle, .start = trailingCycleStart(tokens, p) + p };
    }
    if (exactCyclePeriod(
        tokens,
        degenerate_loop_max_period + 1,
        degenerate_loop_long_max_period,
        degenerate_loop_long_reps,
    )) |p| {
        return .{ .tier = .long_cycle, .start = trailingCycleStart(tokens, p) + p };
    }
    if (tokens.len < near_repeat_window) return null;
    if (!nearRepeatWindowIsDegenerate(tokens[tokens.len - near_repeat_window ..])) return null;

    // Slide the window back while it keeps convicting. A restatement loop
    // that has run for 3000 tokens is degenerate for all 3000 — trimming only
    // the last window would hand the client the rest of the loop back.
    var start = tokens.len - near_repeat_window;
    const floor = if (tokens.len > near_repeat_window + near_repeat_max_lookback)
        tokens.len - near_repeat_window - near_repeat_max_lookback
    else
        0;
    while (start >= floor + near_repeat_step) {
        const cand = start - near_repeat_step;
        if (!nearRepeatWindowIsDegenerate(tokens[cand .. cand + near_repeat_window])) break;
        start = cand;
    }
    return .{ .tier = .near_repeat, .start = start };
}
