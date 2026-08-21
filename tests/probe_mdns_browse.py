"""Query the LAN for _mlxserve._tcp and dump exactly what is advertised.

Purpose: capture the ground-truth interop contract from a real macOS mlx-serve
before porting anything. src/lan_policy.zig is the shared spec (SERVICE_TYPE,
txtBuild) and must not drift -- so read what the Mac actually puts on the wire
rather than trusting our own encoder.

Deliberately hand-rolled: no zeroconf/dnspython install, and on Windows the
multicast join has to be done per-interface (INADDR_ANY joins ONE interface,
frequently a Hyper-V/WSL/VPN adapter -- the trap mdns.zig's header records).
"""
import socket, struct, sys, time

MCAST = "224.0.0.251"
PORT = 5353
SERVICE = "_mlxserve._tcp.local"


def encode_name(name):
    out = b""
    for label in name.split("."):
        if label:
            out += bytes([len(label)]) + label.encode()
    return out + b"\x00"


def build_query(name, qtype):
    # id=0 (mDNS ignores it), flags=0 (standard query), 1 question.
    header = struct.pack(">HHHHHH", 0, 0, 1, 0, 0, 0)
    # QU bit clear -> multicast response; class IN.
    return header + encode_name(name) + struct.pack(">HH", qtype, 1)


def read_name(buf, off):
    """Decode a DNS name, following compression pointers.

    Compression MUST be parsed even though we never emit it -- peers emit it,
    and a parser that cannot follow a pointer reads garbage for every record
    after the first.
    """
    parts = []
    jumped = False
    end = off
    hops = 0
    while True:
        if off >= len(buf):
            break
        ln = buf[off]
        if ln == 0:
            off += 1
            if not jumped:
                end = off
            break
        if ln & 0xC0 == 0xC0:
            ptr = ((ln & 0x3F) << 8) | buf[off + 1]
            if not jumped:
                end = off + 2
            off = ptr
            jumped = True
            hops += 1
            if hops > 20:
                break
            continue
        parts.append(buf[off + 1: off + 1 + ln].decode("utf-8", "replace"))
        off += 1 + ln
    return ".".join(parts), end


def parse(buf):
    if len(buf) < 12:
        return []
    qd, an, ns, ar = struct.unpack(">HHHH", buf[4:12])
    off = 12
    for _ in range(qd):
        _, off = read_name(buf, off)
        off += 4
    recs = []
    for _ in range(an + ns + ar):
        if off >= len(buf):
            break
        name, off = read_name(buf, off)
        if off + 10 > len(buf):
            break
        rtype, rclass, ttl, rdlen = struct.unpack(">HHIH", buf[off:off + 10])
        off += 10
        rdata = buf[off:off + rdlen]
        recs.append((name, rtype, rdata, buf, off))
        off += rdlen
    return recs


def local_ipv4s():
    """Every IPv4 this host has. Windows joins the multicast group on ONE
    interface for INADDR_ANY, and on a box with Hyper-V / WSL / VPN adapters
    that is frequently the wrong one -- so join them all explicitly."""
    addrs = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addrs.add(info[4][0])
    except Exception:
        pass
    addrs.add("0.0.0.0")
    return sorted(addrs)


def main():
    qtypes = [(12, "PTR"), (33, "SRV"), (16, "TXT"), (1, "A")]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 0))
    sock.settimeout(0.5)

    ifaces = local_ipv4s()
    print("querying %s on interfaces: %s" % (SERVICE, ", ".join(ifaces)))
    for ip in ifaces:
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                            socket.inet_aton(ip))
        except OSError:
            continue
        for qt, _ in qtypes[:1]:
            try:
                sock.sendto(build_query(SERVICE, qt), (MCAST, PORT))
            except OSError:
                pass

    seen = {}
    deadline = time.time() + 6
    while time.time() < deadline:
        try:
            data, addr = sock.recvfrom(9000)
        except socket.timeout:
            continue
        except OSError:
            break
        for name, rtype, rdata, buf, off in parse(data):
            if "mlxserve" not in name and rtype != 1:
                continue
            key = (addr[0], name, rtype)
            if key in seen:
                continue
            seen[key] = True
            if rtype == 12:
                target, _ = read_name(buf, off)
                print("\nPTR  from %s\n  %s -> %s" % (addr[0], name, target))
                for qt in (33, 16):
                    sock.sendto(build_query(target, qt), (MCAST, PORT))
            elif rtype == 33 and len(rdata) >= 6:
                pri, wt, port = struct.unpack(">HHH", rdata[:6])
                host, _ = read_name(buf, off + 6)
                print("\nSRV  from %s\n  %s\n  host=%s port=%d" % (addr[0], name, host, port))
                sock.sendto(build_query(host, 1), (MCAST, PORT))
            elif rtype == 16:
                print("\nTXT  from %s\n  %s" % (addr[0], name))
                print("  raw bytes (%d): %s" % (len(rdata), rdata.hex()))
                i = 0
                while i < len(rdata):
                    ln = rdata[i]
                    if ln == 0 or i + 1 + ln > len(rdata):
                        break
                    print("    [len=%d] %r" % (ln, rdata[i + 1:i + 1 + ln]))
                    i += 1 + ln
            elif rtype == 1 and len(rdata) == 4 and "mlxserve" in str(seen):
                print("  A    %s -> %s" % (name, socket.inet_ntoa(rdata)))

    if not seen:
        print("\nNOTHING FOUND. Either no peer is advertising _mlxserve._tcp on"
              " this link, or the query never left the right interface.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
