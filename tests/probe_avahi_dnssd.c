// Portability probe: does avahi-compat-libdns_sd actually behave like Apple's
// dns_sd for the way src/lan_bonjour.zig uses it?
//
// windows-plan.md 3.5 (and ../zig-ai/bonjour.md) flags this as UNVERIFIED and
// says to prove it before planning around it. If it holds, Linux LAN sharing
// costs an import gate rather than a hand-rolled mDNS module.
//
// This deliberately does more than check that the symbols link. Linking proves
// the ABI exists; it does not prove avahi-compat implements the SEMANTICS we
// depend on. The two that would silently sink the shortcut:
//
//   - DNSServiceRefSockFD + DNSServiceProcessResult. Our whole event loop is
//     built on getting a pollable fd out of the library and draining it
//     ourselves. A stub that returned -1 would link fine and never deliver a
//     single callback.
//   - The TXT record round-trip. We advertise "v=1" + "t=<token>" in exactly
//     the layout lan_policy.txtBuild emits, and that self-token is what makes
//     proxy loops impossible by construction. If the bytes do not come back
//     verbatim, the security gate is what breaks.
//
// So: register our real SERVICE_TYPE with a real TXT record, browse for it,
// resolve what we find, and check the TXT we get back is byte-identical.
//
//   gcc -o /tmp/probe tests/probe_avahi_dnssd.c -ldns_sd && /tmp/probe
//
// Requires libavahi-compat-libdnssd-dev and a RUNNING avahi-daemon: the compat
// layer is a client shim, so with no daemon every call fails. That is itself
// worth knowing, because it is a deployment dependency Apple's dns_sd does not
// have.
//
// MEASURED 2026-08-21 (Ubuntu 26.04, libavahi-compat-libdnssd1 0.8-18): six of
// the seven entry points are exported; DNSServiceGetAddrInfo is NOT, and is not
// even declared in avahi-compat's dns_sd.h. avahi-compat implements the older
// mDNSResponder API set, and GetAddrInfo postdates it. So this probe must not
// call it directly -- a build that fails to COMPILE answers nothing. It is
// looked up with dlsym (absent -> reported, not fatal) and the A-record lookup
// is then re-attempted through DNSServiceQueryRecord, which IS exported. That
// substitution is the whole delta between "Linux reuses lan_bonjour.zig" and
// "Linux needs the hand-rolled module", so the probe measures it rather than
// asserting it.
#include <dns_sd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/select.h>
#include <arpa/inet.h>
#include <dlfcn.h>

// Not in avahi-compat's dns_sd.h (see the header note), so spelled out here.
// Apple's values; only used against Apple's library, via dlsym.
#define MSV_PROTOCOL_IPv4 0x01
typedef void (DNSSD_API *msv_addrinfo_reply)(DNSServiceRef, DNSServiceFlags,
                                             uint32_t, DNSServiceErrorType,
                                             const char *, const struct sockaddr *,
                                             uint32_t, void *);
typedef DNSServiceErrorType (DNSSD_API *msv_getaddrinfo_fn)(
    DNSServiceRef *, DNSServiceFlags, uint32_t, uint32_t, const char *,
    msv_addrinfo_reply, void *);

// Must match src/lan_policy.zig SERVICE_TYPE. Interop with macOS mlx-serve and
// with zig-ai depends on this exact string.
#define SERVICE_TYPE "_mlxserve._tcp"
#define PROBE_TOKEN  "probe-token-0123456789"

static int  found_service = 0;
static int  resolved      = 0;
static int  txt_matched   = 0;
static int  got_addr      = 0;
static int  got_addr_query = 0;
static char found_name[256];
static char resolved_host[256];

// The TXT that lan_policy.txtBuild produces: length-prefixed "v=1", then
// length-prefixed "t=<token>".
static unsigned char  txt_buf[300];
static unsigned short txt_len;

static void build_txt(void) {
    size_t tl = strlen(PROBE_TOKEN);
    txt_buf[0] = 3;
    memcpy(txt_buf + 1, "v=1", 3);
    txt_buf[4] = (unsigned char)(2 + tl);
    memcpy(txt_buf + 5, "t=", 2);
    memcpy(txt_buf + 7, PROBE_TOKEN, tl);
    txt_len = (unsigned short)(7 + tl);
}

// Drain one ref for up to `secs`, the way lan_bonjour.zig's loop does: get the
// fd, select on it, hand control back to the library.
static int pump(DNSServiceRef ref, int secs, int *done_flag) {
    int fd = DNSServiceRefSockFD(ref);
    if (fd < 0) {
        printf("  FAIL DNSServiceRefSockFD returned %d (no pollable fd)\n", fd);
        return -1;
    }
    for (int i = 0; i < secs * 10 && !*done_flag; i++) {
        fd_set set;
        FD_ZERO(&set);
        FD_SET(fd, &set);
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000;
        int r = select(fd + 1, &set, NULL, NULL, &tv);
        if (r > 0) {
            DNSServiceErrorType e = DNSServiceProcessResult(ref);
            if (e != kDNSServiceErr_NoError) {
                printf("  FAIL DNSServiceProcessResult -> %d\n", e);
                return -1;
            }
        }
    }
    return 0;
}

static void DNSSD_API on_browse(DNSServiceRef r, DNSServiceFlags flags,
                                uint32_t ifidx, DNSServiceErrorType err,
                                const char *name, const char *type,
                                const char *domain, void *ctx) {
    (void)r; (void)ifidx; (void)type; (void)domain; (void)ctx;
    if (err != kDNSServiceErr_NoError) return;
    if (!(flags & kDNSServiceFlagsAdd)) return;
    snprintf(found_name, sizeof(found_name), "%s", name);
    found_service = 1;
}

static void DNSSD_API on_resolve(DNSServiceRef r, DNSServiceFlags flags,
                                 uint32_t ifidx, DNSServiceErrorType err,
                                 const char *fullname, const char *host,
                                 uint16_t port, uint16_t tlen,
                                 const unsigned char *txt, void *ctx) {
    (void)r; (void)flags; (void)ifidx; (void)fullname; (void)ctx;
    if (err != kDNSServiceErr_NoError) return;
    printf("  resolved host=%s port=%u txt_len=%u\n", host, ntohs(port), tlen);
    snprintf(resolved_host, sizeof(resolved_host), "%s", host);
    // Byte-identical TXT is the bar: the self-token rides in here.
    if (tlen == txt_len && memcmp(txt, txt_buf, txt_len) == 0) {
        txt_matched = 1;
    } else {
        printf("  TXT MISMATCH: sent %u bytes, got %u\n", txt_len, tlen);
    }
    resolved = 1;
}

static void DNSSD_API on_addr(DNSServiceRef r, DNSServiceFlags flags,
                              uint32_t ifidx, DNSServiceErrorType err,
                              const char *host, const struct sockaddr *addr,
                              uint32_t ttl, void *ctx) {
    (void)r; (void)flags; (void)ifidx; (void)ttl; (void)ctx;
    if (err != kDNSServiceErr_NoError || !addr) return;
    if (addr->sa_family == AF_INET) {
        char ip[INET_ADDRSTRLEN];
        const struct sockaddr_in *s = (const struct sockaddr_in *)addr;
        inet_ntop(AF_INET, &s->sin_addr, ip, sizeof(ip));
        printf("  addrinfo %s -> %s\n", host, ip);
        got_addr = 1;
    }
}

// The DNSServiceGetAddrInfo substitute: an A-record query. rdata is the raw
// 4-byte address, which is what lan_bonjour.zig ultimately wants anyway -- it
// only ever reads the IPv4 out of the sockaddr GetAddrInfo hands back.
static void DNSSD_API on_query(DNSServiceRef r, DNSServiceFlags flags,
                               uint32_t ifidx, DNSServiceErrorType err,
                               const char *fullname, uint16_t rrtype,
                               uint16_t rrclass, uint16_t rdlen,
                               const void *rdata, uint32_t ttl, void *ctx) {
    (void)r; (void)flags; (void)ifidx; (void)rrclass; (void)ttl; (void)ctx;
    if (err != kDNSServiceErr_NoError) return;
    if (rrtype != kDNSServiceType_A || rdlen != 4) return;
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, rdata, ip, sizeof(ip));
    printf("  queryrecord %s -> %s\n", fullname, ip);
    got_addr_query = 1;
}

int main(void) {
    build_txt();
    printf("probe: avahi-compat-libdns_sd vs the 7 entry points lan_bonjour.zig calls\n");

    // 1. Register -- the advertise half.
    DNSServiceRef reg = NULL;
    DNSServiceErrorType e = DNSServiceRegister(
        &reg, 0, 0, "mlx-serve-probe", SERVICE_TYPE, NULL, NULL,
        htons(18170), txt_len, txt_buf, NULL, NULL);
    if (e != kDNSServiceErr_NoError) {
        printf("  FAIL DNSServiceRegister -> %d (is avahi-daemon running?)\n", e);
        return 1;
    }
    printf("  OK   DNSServiceRegister\n");
    // Let the daemon commit the registration before browsing for it.
    int dummy = 0;
    pump(reg, 2, &dummy);

    // 2. Browse -- the discover half.
    DNSServiceRef br = NULL;
    e = DNSServiceBrowse(&br, 0, 0, SERVICE_TYPE, NULL, on_browse, NULL);
    if (e != kDNSServiceErr_NoError) {
        printf("  FAIL DNSServiceBrowse -> %d\n", e);
        return 1;
    }
    printf("  OK   DNSServiceBrowse\n");
    if (pump(br, 5, &found_service) < 0) return 1;
    if (!found_service) {
        printf("  FAIL browse never saw our own registration\n");
        return 1;
    }
    printf("  OK   DNSServiceRefSockFD + DNSServiceProcessResult (found %s)\n", found_name);

    // 3. Resolve -- name to host/port/TXT.
    DNSServiceRef rs = NULL;
    e = DNSServiceResolve(&rs, 0, 0, found_name, SERVICE_TYPE, "local.", on_resolve, NULL);
    if (e != kDNSServiceErr_NoError) {
        printf("  FAIL DNSServiceResolve -> %d\n", e);
        return 1;
    }
    if (pump(rs, 5, &resolved) < 0) return 1;
    if (!resolved) {
        printf("  FAIL resolve produced no callback\n");
        return 1;
    }
    printf("  OK   DNSServiceResolve\n");
    printf("  %s TXT round-trips byte-identically\n", txt_matched ? "OK  " : "FAIL");

    // 4. Host to IPv4. This is the one entry point avahi-compat may not have,
    //    so ask the loaded library rather than the header.
    const char *host = resolved_host[0] ? resolved_host : "localhost";
    msv_getaddrinfo_fn getaddrinfo_fn =
        (msv_getaddrinfo_fn)dlsym(RTLD_DEFAULT, "DNSServiceGetAddrInfo");
    if (!getaddrinfo_fn) {
        printf("  MISS DNSServiceGetAddrInfo is not exported by this libdns_sd\n");
    } else {
        DNSServiceRef ai = NULL;
        e = getaddrinfo_fn(&ai, 0, 0, MSV_PROTOCOL_IPv4, host, on_addr, NULL);
        if (e != kDNSServiceErr_NoError) {
            printf("  WARN DNSServiceGetAddrInfo -> %d\n", e);
        } else {
            pump(ai, 5, &got_addr);
            printf("  %s DNSServiceGetAddrInfo\n", got_addr ? "OK  " : "WARN");
            DNSServiceRefDeallocate(ai);
        }
    }

    // 4b. The substitute, measured whether or not 4 worked: if this serves,
    //     the missing entry point costs a small shim, not a mDNS stack.
    DNSServiceRef qr = NULL;
    e = DNSServiceQueryRecord(&qr, 0, 0, host, kDNSServiceType_A,
                              kDNSServiceClass_IN, on_query, NULL);
    if (e != kDNSServiceErr_NoError) {
        printf("  FAIL DNSServiceQueryRecord -> %d\n", e);
    } else {
        pump(qr, 5, &got_addr_query);
        printf("  %s DNSServiceQueryRecord (A) as the GetAddrInfo substitute\n",
               got_addr_query ? "OK  " : "FAIL");
        DNSServiceRefDeallocate(qr);
    }

    // 5. Deallocate -- the teardown half.
    DNSServiceRefDeallocate(rs);
    DNSServiceRefDeallocate(br);
    DNSServiceRefDeallocate(reg);
    printf("  OK   DNSServiceRefDeallocate\n");

    // The discovery half is pass/fail; the address half has a middle outcome,
    // and collapsing it would either overstate the shortcut or condemn Linux to
    // a hand-rolled mDNS stack it does not need.
    int core_ok = found_service && resolved && txt_matched;
    printf("\nVERDICT: ");
    if (!core_ok) {
        printf("avahi-compat does NOT serve our usage -- Linux needs the hand-rolled module\n");
        return 1;
    }
    if (got_addr) {
        printf("avahi-compat serves all 7 entry points -- lan_bonjour.zig links unmodified\n");
        return 0;
    }
    if (got_addr_query) {
        printf("avahi-compat serves 6 of 7; DNSServiceGetAddrInfo needs a\n"
               "         DNSServiceQueryRecord(A) shim -- Linux reuses lan_bonjour.zig plus that\n");
        return 0;
    }
    printf("discovery works but no address lookup path does -- investigate before planning\n");
    return 1;
}
