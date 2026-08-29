//! Cheap per-step video previews for SSE progress events.
//!
//! A full VAE decode every denoise step is not safe (H3 stages the text
//! encoder and DiT because they cannot both stay resident). This is the
//! no-extra-weights fallback from issue #208: a linear Latent2RGB map from
//! 24 (MiniMax-H3) or 128 (LTX) channels to RGB, optional bilinear resize,
//! then JPEG. Hermetic — no MLX.

const std = @import("std");
const jpeg = @import("jpeg.zig");

/// Opt-in request knobs. Default behaviour is off.
pub const Opts = struct {
    enabled: bool = false,
    /// How many temporal slices to show. 1 = one JPEG (mid clip, or frame 0
    /// for I2V). >1 = evenly spaced slices packed as a horizontal JPEG
    /// filmstrip. Animated WebP is a later cut.
    frames: u32 = 1,
    /// Max width or height in px. 0 = latent native size.
    max_side: u32 = 256,

    pub const max_frames: u32 = 8;
    pub const max_side_cap: u32 = 1024;

    pub fn normalize(self: Opts) Opts {
        return .{
            .enabled = self.enabled,
            .frames = if (self.frames == 0) 1 else @min(self.frames, max_frames),
            .max_side = @min(self.max_side, max_side_cap),
        };
    }
};

/// One encoded preview image. Caller owns `jpeg`.
pub const Encoded = struct {
    jpeg: []u8,
    w: u32,
    h: u32,
    mime: []const u8 = "image/jpeg",
};

/// Temporal indices into a clip of length `t`. `first_frame` pins the first
/// slice at 0 (I2V / keyframe). Otherwise a single slice is the midpoint.
pub fn temporalIndices(out: []u32, t: u32, n_want: u32, first_frame: bool) []u32 {
    if (t == 0 or out.len == 0) return out[0..0];
    const n: u32 = @min(n_want, @min(t, @as(u32, @intCast(out.len))));
    if (n == 1) {
        out[0] = if (first_frame) 0 else (t - 1) / 2;
        return out[0..1];
    }
    const denom = n - 1;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        out[i] = (i * (t - 1)) / denom;
    }
    return out[0..n];
}

/// BCFHW latent (B=1) → JPEG. `latent` is C*T*H*W f32, channel-major.
/// `first_frame` selects temporal index 0 when `opts.frames == 1`.
pub fn jpegFromLatent(
    allocator: std.mem.Allocator,
    latent: []const f32,
    c: u32,
    t: u32,
    h: u32,
    w: u32,
    opts: Opts,
    first_frame: bool,
) !Encoded {
    const o = opts.normalize();
    const need = @as(usize, c) * @as(usize, t) * @as(usize, h) * @as(usize, w);
    if (latent.len < need or c == 0 or t == 0 or h == 0 or w == 0) return error.BadLatentShape;

    var idx_buf: [Opts.max_frames]u32 = undefined;
    const idx = temporalIndices(&idx_buf, t, o.frames, first_frame);

    const rgb_native = try allocator.alloc(u8, @as(usize, idx.len) * @as(usize, h) * @as(usize, w) * 3);
    defer allocator.free(rgb_native);
    for (idx, 0..) |ti, fi| {
        latentSliceToRgb(latent, c, t, h, w, ti, rgb_native[fi * @as(usize, h) * @as(usize, w) * 3 ..][0 .. @as(usize, h) * @as(usize, w) * 3]);
    }

    const strip_w = w * @as(u32, @intCast(idx.len));
    const strip_h = h;
    const resized = try resizeMaxSide(allocator, rgb_native, strip_w, strip_h, o.max_side);
    defer if (resized.owned) allocator.free(resized.rgb);

    const jpg = try jpeg.encodeRgb(allocator, resized.rgb, resized.w, resized.h, 72);
    return .{ .jpeg = jpg, .w = resized.w, .h = resized.h };
}

/// SSE `data:` payload (no trailing blank line). Stage is JSON-escaped.
pub fn formatProgressJson(
    allocator: std.mem.Allocator,
    stage: []const u8,
    step: u32,
    total: u32,
    preview: ?Encoded,
) ![]u8 {
    var esc_buf: [256]u8 = undefined;
    const esc = jsonEscape(&esc_buf, stage);
    if (preview) |p| {
        const b64_len = std.base64.standard.Encoder.calcSize(p.jpeg.len);
        const b64 = try allocator.alloc(u8, b64_len);
        defer allocator.free(b64);
        _ = std.base64.standard.Encoder.encode(b64, p.jpeg);
        return std.fmt.allocPrint(
            allocator,
            "{{\"type\":\"progress\",\"stage\":\"{s}\",\"step\":{d},\"total\":{d},\"preview\":\"{s}\",\"mime\":\"{s}\",\"w\":{d},\"h\":{d}}}",
            .{ esc, step, total, b64, p.mime, p.w, p.h },
        );
    }
    return std.fmt.allocPrint(
        allocator,
        "{{\"type\":\"progress\",\"stage\":\"{s}\",\"step\":{d},\"total\":{d}}}",
        .{ esc, step, total },
    );
}

fn jsonEscape(out: []u8, msg: []const u8) []const u8 {
    var n: usize = 0;
    for (msg) |ch| {
        var one: [1]u8 = .{if (ch < 0x20) ' ' else ch};
        const esc: []const u8 = switch (ch) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            else => one[0..1],
        };
        if (n + esc.len > out.len) break;
        @memcpy(out[n..][0..esc.len], esc);
        n += esc.len;
    }
    return out[0..n];
}

/// Linear map from C latent channels → RGB, then a robust mean/std stretch
/// to 0–255. Channel `i` gets a unique hue via the golden angle so structure
/// in any subset of channels still reads as colour rather than a grey smear.
fn latentSliceToRgb(latent: []const f32, c: u32, t: u32, h: u32, w: u32, ti: u32, rgb: []u8) void {
    const hw: usize = @as(usize, h) * @as(usize, w);
    const thw: usize = @as(usize, t) * hw;
    const inv_c = 1.0 / @sqrt(@as(f32, @floatFromInt(c)));
    var sum: f32 = 0;
    var sum2: f32 = 0;
    const count: f32 = @floatFromInt(hw * 3);

    var pix_i: usize = 0;
    while (pix_i < hw) : (pix_i += 1) {
        const rgb3 = mapPixel(latent, c, thw, hw, ti, pix_i, inv_c);
        sum += rgb3[0] + rgb3[1] + rgb3[2];
        sum2 += rgb3[0] * rgb3[0] + rgb3[1] * rgb3[1] + rgb3[2] * rgb3[2];
    }
    const mean = sum / count;
    const var_ = @max(sum2 / count - mean * mean, 0);
    const stddev = @sqrt(var_);
    const scale = 1.0 / (2.0 * stddev + 1e-6);

    pix_i = 0;
    while (pix_i < hw) : (pix_i += 1) {
        const rgb3 = mapPixel(latent, c, thw, hw, ti, pix_i, inv_c);
        rgb[pix_i * 3 + 0] = toU8((rgb3[0] - mean) * scale);
        rgb[pix_i * 3 + 1] = toU8((rgb3[1] - mean) * scale);
        rgb[pix_i * 3 + 2] = toU8((rgb3[2] - mean) * scale);
    }
}

fn mapPixel(latent: []const f32, c: u32, thw: usize, hw: usize, ti: u32, pix: usize, inv_c: f32) [3]f32 {
    var r: f32 = 0;
    var g: f32 = 0;
    var b: f32 = 0;
    var ci: u32 = 0;
    while (ci < c) : (ci += 1) {
        const v = latent[@as(usize, ci) * thw + @as(usize, ti) * hw + pix];
        const angle = @as(f32, @floatFromInt(ci)) * 2.399963229728653;
        r += v * @cos(angle);
        g += v * @cos(angle + 2.0943951023931953);
        b += v * @cos(angle + 4.1887902047863905);
    }
    return .{ r * inv_c, g * inv_c, b * inv_c };
}

fn toU8(x: f32) u8 {
    const y = x * 0.5 + 0.5;
    const z = @min(@max(y, 0.0), 1.0);
    return @intFromFloat(z * 255.0 + 0.5);
}

const Resize = struct {
    rgb: []const u8,
    w: u32,
    h: u32,
    owned: bool,
};

fn resizeMaxSide(allocator: std.mem.Allocator, rgb: []const u8, w: u32, h: u32, max_side: u32) !Resize {
    if (max_side == 0) return .{ .rgb = rgb, .w = w, .h = h, .owned = false };
    const long: u32 = @max(w, h);
    if (long <= max_side) return .{ .rgb = rgb, .w = w, .h = h, .owned = false };
    const nw: u32 = @max(1, (w * max_side) / long);
    const nh: u32 = @max(1, (h * max_side) / long);
    const out = try allocator.alloc(u8, @as(usize, nw) * @as(usize, nh) * 3);
    bilinear(rgb, w, h, out, nw, nh);
    return .{ .rgb = out, .w = nw, .h = nh, .owned = true };
}

fn bilinear(src: []const u8, sw: u32, sh: u32, dst: []u8, dw: u32, dh: u32) void {
    const x_scale = @as(f32, @floatFromInt(sw)) / @as(f32, @floatFromInt(dw));
    const y_scale = @as(f32, @floatFromInt(sh)) / @as(f32, @floatFromInt(dh));
    var y: u32 = 0;
    while (y < dh) : (y += 1) {
        const fy = (@as(f32, @floatFromInt(y)) + 0.5) * y_scale - 0.5;
        const y0 = @as(u32, @intFromFloat(@max(fy, 0)));
        const y1 = @min(y0 + 1, sh - 1);
        const ty = fy - @as(f32, @floatFromInt(y0));
        var x: u32 = 0;
        while (x < dw) : (x += 1) {
            const fx = (@as(f32, @floatFromInt(x)) + 0.5) * x_scale - 0.5;
            const x0 = @as(u32, @intFromFloat(@max(fx, 0)));
            const x1 = @min(x0 + 1, sw - 1);
            const tx = fx - @as(f32, @floatFromInt(x0));
            var ch: u32 = 0;
            while (ch < 3) : (ch += 1) {
                const a = samp(src, sw, x0, y0, ch);
                const b = samp(src, sw, x1, y0, ch);
                const c = samp(src, sw, x0, y1, ch);
                const d = samp(src, sw, x1, y1, ch);
                const top = a + (b - a) * tx;
                const bot = c + (d - c) * tx;
                const v = top + (bot - top) * ty;
                dst[(@as(usize, y) * @as(usize, dw) + @as(usize, x)) * 3 + ch] = @intFromFloat(@min(@max(v, 0), 255) + 0.5);
            }
        }
    }
}

fn samp(src: []const u8, sw: u32, x: u32, y: u32, ch: u32) f32 {
    return @floatFromInt(src[(@as(usize, y) * @as(usize, sw) + @as(usize, x)) * 3 + ch]);
}

test "temporalIndices: one mid slice, I2V uses frame 0" {
    var buf: [8]u32 = undefined;
    try std.testing.expectEqualSlices(u32, &.{5}, temporalIndices(&buf, 11, 1, false));
    try std.testing.expectEqualSlices(u32, &.{0}, temporalIndices(&buf, 11, 1, true));
    try std.testing.expectEqualSlices(u32, &.{ 0, 5, 10 }, temporalIndices(&buf, 11, 3, false));
}

test "Opts.normalize clamps frames and max_side, keeps 0 = native" {
    const a = (Opts{ .frames = 0, .max_side = 0 }).normalize();
    try std.testing.expectEqual(@as(u32, 1), a.frames);
    try std.testing.expectEqual(@as(u32, 0), a.max_side);
    const b = (Opts{ .frames = 99, .max_side = 99999 }).normalize();
    try std.testing.expectEqual(Opts.max_frames, b.frames);
    try std.testing.expectEqual(Opts.max_side_cap, b.max_side);
}

test "jpegFromLatent: 24-ch H3-shaped volume yields a JPEG" {
    const a = std.testing.allocator;
    const c: u32 = 24;
    const t: u32 = 5;
    const h: u32 = 8;
    const w: u32 = 12;
    const n = c * t * h * w;
    const lat = try a.alloc(f32, n);
    defer a.free(lat);
    for (lat, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.017);
    const enc = try jpegFromLatent(a, lat, c, t, h, w, .{ .enabled = true, .frames = 1, .max_side = 256 }, false);
    defer a.free(enc.jpeg);
    try std.testing.expect(enc.jpeg.len > 16);
    try std.testing.expectEqual(@as(u8, 0xFF), enc.jpeg[0]);
    try std.testing.expectEqual(@as(u8, 0xD8), enc.jpeg[1]);
    try std.testing.expectEqual(@as(u32, w), enc.w);
    try std.testing.expectEqual(@as(u32, h), enc.h);
    try std.testing.expectEqualStrings("image/jpeg", enc.mime);
}

test "jpegFromLatent: max_side downscales the long edge" {
    const a = std.testing.allocator;
    const c: u32 = 24;
    const t: u32 = 1;
    const h: u32 = 32;
    const w: u32 = 64;
    const lat = try a.alloc(f32, c * t * h * w);
    defer a.free(lat);
    @memset(lat, 0.25);
    const enc = try jpegFromLatent(a, lat, c, t, h, w, .{ .max_side = 16 }, false);
    defer a.free(enc.jpeg);
    try std.testing.expectEqual(@as(u32, 16), enc.w);
    try std.testing.expectEqual(@as(u32, 8), enc.h);
}

test "jpegFromLatent: preview_frames>1 is a filmstrip, not taller" {
    const a = std.testing.allocator;
    const c: u32 = 8;
    const t: u32 = 4;
    const h: u32 = 4;
    const w: u32 = 4;
    const lat = try a.alloc(f32, c * t * h * w);
    defer a.free(lat);
    @memset(lat, 0);
    const enc = try jpegFromLatent(a, lat, c, t, h, w, .{ .frames = 4, .max_side = 0 }, false);
    defer a.free(enc.jpeg);
    try std.testing.expectEqual(@as(u32, 16), enc.w);
    try std.testing.expectEqual(@as(u32, 4), enc.h);
}

test "formatProgressJson: default event has no preview key" {
    const a = std.testing.allocator;
    const s = try formatProgressJson(a, "Generating", 8, 30, null);
    defer a.free(s);
    try std.testing.expectEqualStrings(
        "{\"type\":\"progress\",\"stage\":\"Generating\",\"step\":8,\"total\":30}",
        s,
    );
    var parsed = try std.json.parseFromSlice(std.json.Value, a, s, .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.object.get("preview") == null);
}

test "formatProgressJson: preview event is parseable JSON with JPEG b64" {
    const a = std.testing.allocator;
    const jpg = [_]u8{ 0xFF, 0xD8, 0xFF, 0x00 };
    const enc = Encoded{ .jpeg = @constCast(&jpg), .w = 256, .h = 144 };
    const s = try formatProgressJson(a, "Generating", 8, 30, enc);
    defer a.free(s);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, s, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqualStrings("progress", o.get("type").?.string);
    try std.testing.expectEqual(@as(i64, 8), o.get("step").?.integer);
    try std.testing.expectEqual(@as(i64, 30), o.get("total").?.integer);
    try std.testing.expectEqualStrings("image/jpeg", o.get("mime").?.string);
    try std.testing.expectEqual(@as(i64, 256), o.get("w").?.integer);
    try std.testing.expectEqual(@as(i64, 144), o.get("h").?.integer);
    const b64 = o.get("preview").?.string;
    const raw = try a.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(b64));
    defer a.free(raw);
    try std.base64.standard.Decoder.decode(raw, b64);
    try std.testing.expectEqualSlices(u8, &jpg, raw);
}

test "jpegFromLatent: 128-ch LTX-shaped volume yields a JPEG" {
    const a = std.testing.allocator;
    const c: u32 = 128;
    const t: u32 = 2;
    const h: u32 = 4;
    const w: u32 = 6;
    const lat = try a.alloc(f32, c * t * h * w);
    defer a.free(lat);
    for (lat, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.1 - 0.8;
    const enc = try jpegFromLatent(a, lat, c, t, h, w, .{ .max_side = 0 }, true);
    defer a.free(enc.jpeg);
    try std.testing.expect(enc.jpeg[0] == 0xFF and enc.jpeg[1] == 0xD8);
}

test {
    _ = @import("jpeg.zig");
}
