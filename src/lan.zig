//! LAN sharing v1 — facade over a portable policy half and a per-host transport.
//!
//! The file was one 1325-line unit until the Windows/Linux port. The split is
//! along the only line that matters: what is Apple-specific.
//!
//!   `lan_policy.zig`         — id parsing, the `routeClass` allowlist, the
//!                              `SharedSet`, name sanitizing, TXT records,
//!                              peer-model rewriting. Pure; builds everywhere;
//!                              carries 11 of the 13 original tests.
//!   `lan_peers.zig`          — the discovered-peer table, the peer-model
//!                              fetch and the streaming proxy tunnel. Portable;
//!                              only DISCOVERY is per-host.
//!   `lan_bonjour.zig`        — dns_sd discovery. Apple only.
//!   `lan_transport_mdns.zig` — discovery over the hand-rolled responder in
//!                              `lan_mdns.zig` (+ `lan_net.zig`), everywhere
//!                              else. avahi-compat was measured and rejected:
//!                              it lacks DNSServiceGetAddrInfo and needs a
//!                              running daemon (windows-plan.md §3.5).
//!
//! Splitting rather than `if (macos)`-ing inside one file keeps the policy
//! tests running on every host: they are where the security-relevant behaviour
//! lives (the keyless gate is `routeClass` x `SharedSet`), and they must not
//! become macOS-only just because the transport under them is.
//!
//! Callers import THIS module and see the original flat API.

const builtin = @import("builtin");

pub const policy = @import("lan_policy.zig");

/// Bonjour exists on Apple platforms only. This keys on the OS rather than on
/// `build_cfg.mlx_enabled` because LAN sharing is orthogonal to the inference
/// backend: a `-Dgguf-only` macOS build can still advertise and discover.
const transport = if (builtin.os.tag == .macos or builtin.os.tag == .ios)
    @import("lan_bonjour.zig")
else
    @import("lan_transport_mdns.zig");

// ── Policy (portable) ──
pub const SERVICE_TYPE = policy.SERVICE_TYPE;
pub const RemoteId = policy.RemoteId;
pub const splitRemoteId = policy.splitRemoteId;
pub const RouteClass = policy.RouteClass;
pub const routeClass = policy.routeClass;
pub const SharedSet = policy.SharedSet;
pub const sanitizeName = policy.sanitizeName;
pub const rewriteModelValue = policy.rewriteModelValue;
pub const unescapeJsonSlashes = policy.unescapeJsonSlashes;
pub const txtBuild = policy.txtBuild;
pub const txtFind = policy.txtFind;
pub const PeerModel = policy.PeerModel;
pub const freePeerModels = policy.freePeerModels;
pub const parsePeerModels = policy.parsePeerModels;

// ── Transport (per-host) ──
pub const Options = transport.Options;
pub const Lan = transport.Lan;
pub const Remote = transport.Remote;
pub const tunnel = transport.tunnel;

test {
    _ = policy;
    _ = transport;
    _ = @import("lan_peers.zig");
    // Reached from the transport as ordinary imports, which does NOT register
    // their tests: the collector only follows `_ =` references out of a test
    // block. Without these two lines `lan_mdns`'s wire-format tests and
    // `lan_net`'s host-clock/interface tests are compiled and never run.
    _ = @import("lan_mdns.zig");
    _ = @import("lan_net.zig");
}
