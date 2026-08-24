const std = @import("std");
const builtin = @import("builtin");

comptime {
    // 0.17.0 isn't tagged stable yet (homebrew still ships 0.16.0) — a nightly
    // build from ziglang.org/download is required until it is. 0.16.0's
    // bundled libc++ fails to compile against the macOS 27 beta SDK
    // (`use of undeclared identifier 'INFINITY'` in its vendored <random>);
    // fixed upstream by 0.17.0-dev, which is why the floor moved.
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 17) {
        @compileError(std.fmt.comptimePrint(
            "mlx-serve requires Zig 0.17 (nightly until 0.17.0 stable ships) (have {d}.{d}.{d}). Grab a nightly from https://ziglang.org/download/.",
            .{ builtin.zig_version.major, builtin.zig_version.minor, builtin.zig_version.patch },
        ));
    }
}

pub fn build(b: *std.Build) void {
    // Pin LC_BUILD_VERSION minos to macOS 26.2 — the honest floor: the linked
    // libmlx is built at deployment target 26.2 (NAX kernels, scripts/
    // build-mlx.sh), so on older macOS the binary can't run anyway; failing at
    // the binary with a clear dyld version error beats "loading" and dying on
    // the dylib. Matches app LSMinimumSystemVersion + Package.swift. Guard:
    // tests/test_mlx_staged_nax.sh (binary minos check).
    // On macOS the default target pins LC_BUILD_VERSION minos as described
    // above. Elsewhere the default is the plain native target, except that
    // Windows is pinned to the GNU ABI: lld links directly against a DLL under
    // `-gnu` but REFUSES one under `-msvc` ("bad file type. Did you specify a
    // DLL instead of an import library?"), and llama.cpp's Windows releases
    // ship DLLs with no import libraries. The boundary we link across is
    // llama.h, which is pure C, so the ABI difference is not observable.
    const default_target: std.Target.Query = switch (@import("builtin").os.tag) {
        .macos => .{ .os_version_min = .{ .semver = .{ .major = 26, .minor = 2, .patch = 0 } } },
        .windows => .{ .abi = .gnu },
        else => .{},
    };
    const target = b.standardTargetOptions(.{ .default_target = default_target });
    const optimize = b.standardOptimizeOption(.{});

    const is_darwin = target.result.os.tag == .macos;

    // GGUF-only build: no MLX, no ds4, no ANE, no media generation — the
    // server is driven entirely by the embedded llama.cpp engine.
    //
    // This is what makes Windows and Linux possible at all. MLX has no Windows
    // build and its CUDA backend is Linux-only, and ~600 call sites across
    // transformer.zig/deepseek_v4.zig go through `mlx_fast_metal_kernel`, which
    // is Metal-only by construction. llama.cpp, by contrast, has first-class
    // CUDA support on both platforms. So off-Apple we keep the whole HTTP /
    // tool-calling / discovery / templating stack and swap the inference floor.
    //
    // Defaults to ON everywhere except macOS. It is exposed as a flag rather
    // than derived silently so a Mac can build (and test) the portable
    // configuration without cross-compiling.
    const gguf_only = b.option(
        bool,
        "gguf-only",
        "Build without MLX/ds4/ANE/media-gen; serve GGUF via llama.cpp only (default: on for non-macOS)",
    ) orelse !is_darwin;

    if (gguf_only and is_darwin)
        std.debug.print("[mlx-serve] -Dgguf-only on macOS: MLX, ds4, ANE and media generation are disabled\n", .{});

    // Setting any non-default target field disables Zig's native macOS SDK detection,
    // so we resolve the SDK path ourselves and surface its frameworks dir.
    const macos_sdk_frameworks: ?[]const u8 = blk: {
        if (target.result.os.tag != .macos) break :blk null;
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" },
            &code,
            .inherit,
        ) catch break :blk null;
        const sdk = std.mem.trim(u8, stdout, " \n\r\t");
        if (sdk.len == 0) break :blk null;
        break :blk b.fmt("{s}/System/Library/Frameworks", .{sdk});
    };

    if (is_darwin) {
        verifyBrewDeps(b);
        // Only the MLX build needs the staged mlx/mlx-c pair; a -Dgguf-only
        // macOS build links neither.
        if (!gguf_only) verifyMlxStage(b);
    }
    verifyLlamaStage(b, target.result.os.tag);

    // App version. Release builds pass it explicitly (app/build.sh computes the
    // next CalVer from the GitHub releases and stamps it into app/Info.plist;
    // the release workflow passes the tag). A plain `zig build` used to fall
    // back to a literal "0.1.0-dev", which then showed up as the version in
    // `--version` AND on the console page — so a dev build reported a version
    // that exists nowhere. Default to the checked-in Info.plist stamp instead:
    // one source of truth, already in the repo, and the same string the last
    // real build shipped. Same pattern as the engine pins below.
    const version = b.option([]const u8, "version", "Version string") orelse readAppVersion(b) orelse "0.0.0-dev";

    const mas = b.option(bool, "mas", "MAS build (no curl/model-pull subprocess)") orelse false;

    // Engine-version pins surfaced by `mlx-serve --version` (the macOS app spawns
    // it and parses the output — see src/version.zig). These are the versions
    // that have NO runtime query API (MLX + ggml report themselves at runtime):
    //   --mlx-c-version  pinned mlx-c submodule version; defaults from the
    //                    lib/mlx/.version stamp (written by scripts/build-mlx.sh)
    //   --ds4-commit     pinned ds4 submodule short commit (build.sh: `git rev-parse`)
    //   --llama-tag      llama.cpp release tag; defaults from lib/llama/.version
    //                    (written by scripts/fetch-llama.sh) so a plain dev build
    //                    still reports it. app/build.sh passes all three.
    const mlx_c_version = b.option([]const u8, "mlx-c-version", "Pinned mlx-c version") orelse readMlxcPin(b) orelse "unknown";
    const ds4_commit = b.option([]const u8, "ds4-commit", "Pinned ds4 submodule short commit") orelse "unknown";
    const llama_tag = b.option([]const u8, "llama-tag", "llama.cpp release tag (bNNNN)") orelse readLlamaTag(b) orelse "unknown";

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", version);
    build_options.addOption(bool, "mas", mas);
    build_options.addOption([]const u8, "mlx_c_version", mlx_c_version);
    build_options.addOption([]const u8, "ds4_commit", ds4_commit);
    build_options.addOption([]const u8, "llama_tag", llama_tag);
    // false for the macOS exe/tests; the iOS static-lib step (`zig build ios-lib`)
    // builds its own options with ios=true so the engine swaps the macOS-only
    // ds4 + llama.cpp engines for no-op stubs (iOS serves MLX safetensors only).
    build_options.addOption(bool, "ios", false);
    build_options.addOption(bool, "gguf_only", gguf_only);

    // ds4 Metal kernel sources embedded via @embedFile and exposed as a
    // named module so src/arch/ds4.zig can import them with `@import("ds4_metal_sources")`
    // without traversing the project root.
    const ds4_metal_sources = b.createModule(.{
        // The real module @embedFiles the kernel sources out of the lib/ds4
        // submodule; without ds4 those files are absent (and Metal-only
        // anyway), so an unconditional path here is a hard FileNotFound at
        // compile time rather than a disabled feature.
        .root_source_file = b.path(if (gguf_only)
            "lib/ds4_metal_sources_stub.zig"
        else
            "lib/ds4_metal_sources.zig"),
        .target = target,
        .optimize = optimize,
    });

    const shared: SharedOpts = .{
        .target = target,
        .optimize = optimize,
        .build_options = build_options,
        .ds4_metal_sources = ds4_metal_sources,
        .is_darwin = is_darwin,
        .gguf_only = gguf_only,
        .macos_sdk_frameworks = macos_sdk_frameworks,
    };

    const mod = makeSharedModule(b, b.path("src/main.zig"), shared);

    const exe = b.addExecutable(.{
        .name = "mlx-serve",
        .root_module = mod,
    });

    // Ensure Mach-O header has room for install_name_tool path changes (app bundling)
    exe.headerpad_max_install_names = true;

    b.installArtifact(exe);

    // Windows resolves a DLL from the EXECUTABLE's own directory (there is no
    // rpath), so the whole llama.cpp DLL set has to land beside mlx-serve.exe.
    // The set must stay COMPLETE even though only llama + ggml-base are linked:
    // ggml dlopens its backends by filename at init (ggml-cuda.dll and the
    // per-microarch ggml-cpu-*.dll variants), so a missing one silently
    // downgrades the backend instead of failing the link.
    if (target.result.os.tag == .windows) installLlamaDlls(b);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    const run_step = b.step("run", "Run mlx-serve");
    run_step.dependOn(&run_cmd.step);

    // Unit tests — identical module configuration to the exe, built from the
    // same helper so the two can no longer drift (they were hand-duplicated).
    // Two roots, selected by FILE: src/tests.zig is the portable suite;
    // src/tests_all.zig adds the MLX/ds4/ANE suites. A comptime `if` inside one
    // root does NOT work here -- a dead `if (cond) _ = @import(x)` branch still
    // registers x's tests, so the MLX imports must be absent from the file the
    // GGUF-only build compiles.
    const test_root = if (gguf_only) "src/tests.zig" else "src/tests_all.zig";
    const test_mod = makeSharedModule(b, b.path(test_root), shared);

    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this substring");
    const qwen_preprocess_fixture = b.option(
        []const u8,
        "qwen-preprocess-fixture",
        "CPU reference fixture for the gated Qwen preprocessing parity test",
    );
    const unit_tests = b.addTest(.{
        .root_module = test_mod,
        .filters = if (test_filter) |f| &.{f} else &.{},
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);

    // The test binary runs out of .zig-cache/o/<hash>/, and Windows resolves an
    // imported DLL from the EXECUTABLE's own directory -- where the llama.cpp
    // set is not. Without this the process dies at load, before a single test
    // runs, and the build reports only a bare exit code.
    if (target.result.os.tag == .windows) {
        // Point PATH at the STAGED tree (lib/llama/bin) -- the source of the
        // set `installLlamaDlls` copies. Deliberately NOT the install dir plus a
        // dependency on the install step: that made `zig build test` reinstall
        // mlx-serve.exe, so the whole suite failed AccessDenied whenever a
        // server was running out of zig-out/bin -- i.e. exactly while someone
        // was testing against one. The staged tree is what fetch-llama.sh
        // populates and the tests need nothing else from the install.
        // `getEnvMap` returns the RunStep's inherited environment, so this
        // PREPENDS rather than replacing PATH -- clobbering it would break any
        // tool the tests shell out to.
        const env = run_unit_tests.getEnvMap();
        const prev = env.get("PATH") orelse "";
        run_unit_tests.setEnvironmentVariable(
            "PATH",
            // Relative on purpose: a RunStep's cwd is the build root, and
            // Windows resolves relative PATH entries against it.
            b.fmt("lib\\llama\\bin;{s}", .{prev}),
        );
    }

    // ggml dlopens its compute backends BY FILENAME out of the running
    // executable's directory, which for a test binary is .zig-cache/o/<hash>/
    // -- empty. Without this every model-gated llama test dies `no backends are
    // loaded` even with the libraries reachable: loading a shared object and
    // registering a backend are two different things.
    //
    // NOT Windows-only, which is where this first bit: the staged tree's own
    // directory NAME differs per host (the Windows release ships DLLs under
    // bin/, a CMake build puts shared objects under lib/), so gating the whole
    // thing on Windows left Linux with no backend dir at all -- and an ELF
    // rpath, which does get the libraries LOADED, hides it right up until the
    // first model-gated test. macOS needs nothing: the XCFramework has its
    // backends compiled in, and the shim's call is fenced `#ifndef __APPLE__`.
    //
    // Relative for the same reason as PATH above: a RunStep's cwd is the build
    // root, and the shim hands this straight to
    // ggml_backend_load_all_from_path, which resolves against that cwd.
    if (target.result.os.tag != .macos) {
        run_unit_tests.setEnvironmentVariable(
            "MLX_LLAMA_BACKEND_DIR",
            if (target.result.os.tag == .windows) "lib/llama/bin" else "lib/llama/lib",
        );
    }
    if (qwen_preprocess_fixture) |fixture| {
        run_unit_tests.setEnvironmentVariable("QWEN_PREPROCESS_FIXTURE", fixture);
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/manifest.json", .{fixture}) });
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/source_rgb.bin", .{fixture}) });
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/pixel_values.bin", .{fixture}) });
    }
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // ── vz-agent: the Agent Sandbox's guest-side binary.
    //
    // A standalone static aarch64-linux-musl ELF (~200 KB) that the app injects
    // into the guest rootfs before boot, exactly like `/.vz-init`. It serves the
    // vsock exec protocol (`src/vz_agent.zig`), replacing the hvc1 console shell.
    //
    // It is NOT imported by main.zig — it links nothing but libc and never runs
    // on macOS. Its tests do, though: `serveConnection` is OS-agnostic, so the
    // whole request → spawn → stream → exit path is exercised over a socketpair
    // here on the host. Wire them into `zig build test` explicitly, since the
    // main test module's root never reaches this file.
    addVzAgent(b, target, optimize, test_step);

    // ── iOS on-device engine: a static library (libmlxserve.a) linking the
    //    MLX-only decode path. ds4 + llama.cpp are stubbed (build_options.ios =
    //    true). Two slices: `zig build ios-lib` (device, arm64-iphoneos) and
    //    `zig build ios-lib-sim` (arm64 iphonesimulator). Driven by the iPhone
    //    app project's build scripts (../mlx-iphone/scripts/build-zig-ios.sh),
    //    which supply the matching --sysroot and copy the artifact out of
    //    zig-out/ios/<sdk>/lib. `-Dios-include=<dir>` points at the iOS dist's
    //    include dir for third-party headers (webp); defaults to Homebrew's,
    //    whose versions are pinned identical by verifyBrewDeps.
    // Apple-only: these resolve their SDKs through xcrun, which does not exist
    // off macOS. (addIosLib already returns quietly when xcrun fails, but
    // gating here keeps the step list honest on other hosts.)
    if (is_darwin) {
        const ios_include = b.option([]const u8, "ios-include", "Include dir for webp/stb headers when cross-compiling the iOS lib") orelse "/opt/homebrew/include";
        addIosLib(b, version, ios_include, .{ .step = "ios-lib", .abi = .none, .sdk = "iphoneos" });
        addIosLib(b, version, ios_include, .{ .step = "ios-lib-sim", .abi = .simulator, .sdk = "iphonesimulator" });
    }
}

/// `zig build vz-agent` → `zig-out/guest/vz-agent` (static aarch64 Linux ELF),
/// plus the host-side unit tests wired into `zig build test`.
fn addVzAgent(
    b: *std.Build,
    host_target: std.Build.ResolvedTarget,
    host_optimize: std.builtin.OptimizeMode,
    test_step: *std.Build.Step,
) void {
    // Guest binary. musl + static so it runs on ANY base image — the bundled
    // Debian rootfs for the App Store build, or a user-chosen alpine/slim image.
    const guest_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .linux,
        .abi = .musl,
    });
    const guest = b.addExecutable(.{
        .name = "vz-agent",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/vz_agent.zig"),
            .target = guest_target,
            // Size, not speed: it shuttles bytes between a socket and a pipe.
            .optimize = .ReleaseSmall,
            .link_libc = true,
        }),
    });
    const install = b.addInstallArtifact(guest, .{
        .dest_dir = .{ .override = .{ .custom = "guest" } },
    });
    const step = b.step("vz-agent", "Build the Agent Sandbox guest binary (static aarch64-linux)");
    step.dependOn(&install.step);

    // Host-side tests of the same source.
    //
    // POSIX-only: `serveConnection` is OS-agnostic, but the harness around it
    // is not — it drives the request -> spawn -> stream -> exit path over a
    // socketpair with fork/pipe/waitpid, none of which Windows has. The guest
    // binary itself still cross-compiles everywhere (it is aarch64-linux-musl
    // regardless of host), so only the HOST test arm is gated.
    if (host_target.result.os.tag == .windows) return;

    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/vz_agent.zig"),
            .target = host_target,
            .optimize = host_optimize,
            .link_libc = true,
        }),
    });
    test_step.dependOn(&b.addRunArtifact(tests).step);

    // A macOS-native build of the SAME source, which listens on a unix socket
    // instead of vsock. `GuestExecInteropTests` (Swift) drives it, so the host
    // frame driver and the guest agent are proven against each other without a
    // VM — the golden-byte tests alone can't catch a streaming bug.
    const host_agent = b.addExecutable(.{
        .name = "vz-agent-host",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/vz_agent.zig"),
            .target = host_target,
            .optimize = host_optimize,
            .link_libc = true,
        }),
    });
    const host_step = b.step("vz-agent-host", "Build vz-agent natively (unix-socket mode, for interop tests)");
    host_step.dependOn(&b.addInstallArtifact(host_agent, .{}).step);
}

const IosSlice = struct { step: []const u8, abi: std.Target.Abi, sdk: []const u8 };

fn addIosLib(b: *std.Build, version: []const u8, ios_include: []const u8, slice: IosSlice) void {
    // Min 18.0 to match the MLX metallib (Metal 3.2). abi=.none → device,
    // abi=.simulator → iOS Simulator slice.
    const ios_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .ios,
        .os_version_min = .{ .semver = .{ .major = 18, .minor = 0, .patch = 0 } },
        .abi = slice.abi,
    });

    const ios_options = b.addOptions();
    ios_options.addOption([]const u8, "version", version);
    ios_options.addOption(bool, "ios", true);
    // Mirror the build options the shared engine sources read (server.zig,
    // scheduler.zig). iOS is sandboxed (no curl/model-pull subprocess), so
    // mas=true; ds4 + llama.cpp are stubbed here, so their version pins are
    // unreported. Without these the iOS lib fails to compile ("options has no
    // member named 'mas'/...").
    ios_options.addOption(bool, "mas", true);
    ios_options.addOption([]const u8, "mlx_c_version", "unknown");
    ios_options.addOption([]const u8, "ds4_commit", "unknown");
    ios_options.addOption([]const u8, "llama_tag", "unknown");

    const mod = b.createModule(.{
        .root_source_file = b.path("src/ios_lib.zig"),
        .target = ios_target,
        .optimize = .ReleaseFast,
        .link_libc = true,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = ios_options.createModule() },
        },
    });

    // Apple cross-compiles don't auto-resolve the SDK's libc/frameworks from
    // --sysroot alone, so wire them explicitly (resolved per slice via xcrun).
    //
    // NO iOS SDK → register NOTHING (the `ios-lib` steps just don't exist in
    // this environment) instead of failing the whole configure: app/build.sh
    // pins DEVELOPER_DIR to the CommandLineTools for the macOS link, and CLT
    // ships no iOS SDKs — a @panic here aborted every macOS app build even
    // though nobody asked for an iOS step.
    var code: u8 = undefined;
    const sdk_path = b.runAllowFail(
        &.{ "xcrun", "--sdk", slice.sdk, "--show-sdk-path" },
        &code,
        .ignore, // silent when absent — CLT environments hit this on purpose
    ) catch return;
    const ios_sdk = std.mem.trim(u8, sdk_path, " \n\r\t");
    if (ios_sdk.len == 0) return;
    mod.addSystemIncludePath(.{ .cwd_relative = b.fmt("{s}/usr/include", .{ios_sdk}) });
    mod.addFrameworkPath(.{ .cwd_relative = b.fmt("{s}/System/Library/Frameworks", .{ios_sdk}) });

    // Headers for the @import("jinja_c")/@import("stb") sites (jinja_wrapper.h,
    // stb_image.h, webp/decode.h). The matching static archives are linked by
    // Xcode at final app-link time.
    mod.addIncludePath(b.path("lib/jinja_cpp"));
    mod.addIncludePath(b.path("lib"));
    mod.addIncludePath(.{ .cwd_relative = ios_include });
    mod.addImport("jinja_c", addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), ios_target, .ReleaseFast, ios_sdk));
    mod.addImport("stb", addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), ios_target, .ReleaseFast, ios_sdk));
    mod.addImport("webp", addCHeaderModule(b, .{ .cwd_relative = b.fmt("{s}/webp/decode.h", .{ios_include}) }, .{ .cwd_relative = ios_include }, ios_target, .ReleaseFast, ios_sdk));
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_write_impl.c"), .flags = &.{"-O2"} });
    // xatlas UV unwrapping (C++), used by the Hunyuan3D texture paint stage via
    // src/uvwrap.zig extern decls — compiled into the lib like the macOS exe.
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addIncludePath(b.path("lib/xatlas"));

    const lib = b.addLibrary(.{
        .name = "mlxserve",
        .root_module = mod,
        .linkage = .static,
    });
    lib.bundle_compiler_rt = true;

    const install = b.addInstallArtifact(lib, .{
        .dest_dir = .{ .override = .{ .custom = b.fmt("ios/{s}/lib", .{slice.sdk}) } },
    });
    const step = b.step(slice.step, b.fmt("Build the iOS engine static lib ({s})", .{slice.sdk}));
    step.dependOn(&install.step);
}

/// Translates a single C header into an importable module (`@import("name")`
/// at the call site) via `addTranslateC`, replacing an inline `@cImport` —
/// removed as a language builtin in 0.17.0-dev.
fn addCHeaderModule(
    b: *std.Build,
    header_path: std.Build.LazyPath,
    include_dir: std.Build.LazyPath,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ios_sdk: []const u8,
) *std.Build.Module {
    const translate = b.addTranslateC(.{
        .root_source_file = header_path,
        .target = target,
        .optimize = optimize,
    });
    translate.addIncludePath(include_dir);
    // iOS cross-compile: addTranslateC (0.17's @cImport replacement) does NOT
    // inherit the parent module's SDK include search the way inline @cImport
    // used to, so a header that pulls in <stdio.h>/<inttypes.h> can't find the
    // Apple libc and translation fails. Wire the SDK's system include here.
    // Host builds pass "" (their toolchain resolves the system headers).
    if (ios_sdk.len > 0)
        translate.addSystemIncludePath(.{ .cwd_relative = b.fmt("{s}/usr/include", .{ios_sdk}) });
    return translate.createModule();
}

fn addDs4Sources(b: *std.Build, module: *std.Build.Module) void {
    // Match ds4's Makefile flags (lib/ds4/Makefile lines 10–11). We drop
    // `-mcpu=native` so the produced binary stays portable across Apple
    // Silicon generations — ds4 itself ships portable IR for its Metal
    // kernels, and the C host code is not perf-critical compared to the GPU
    // path. `-Wno-unused-parameter` + `-Wno-unused-variable` keep upstream's
    // warnings from breaking our build without patching the submodule.
    const c_flags = &[_][]const u8{
        "-O3",
        "-ffast-math",
        "-std=c99",
        "-Wno-unused-parameter",
        "-Wno-unused-variable",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-function",
        "-Wno-deprecated-declarations",
    };
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4.c"), .flags = c_flags });
    // ds4.c #includes ds4_distributed.h; the engine/session path links its impl.
    // ds4_gpu.h is implemented in ds4_metal.m; ds4_kvstore/web/help/agent.c and
    // ds4_gpu_args.c are CLI/server-only and not part of the library path
    // mlx-serve embeds (upstream Makefile CORE_OBJS is the authority).
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_distributed.c"), .flags = c_flags });
    // SSD weight-streaming (issue #39): ds4_ssd.c is a standalone TU (#includes
    // only ds4_ssd.h) implementing the streaming expert cache the engine_options
    // ssd_streaming_* fields drive. Added upstream after the previous pin.
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_ssd.c"), .flags = c_flags });
    // Two-machine tensor parallelism + multi-GPU layer placement (pin efdadd4):
    // ds4.c references ds4_tp_* and ds4_compute_layer_placement/ds4_layer_pack_print
    // unconditionally, so both TUs must link even though we never enable TP.
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_tp.c"), .flags = c_flags });
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_layer_pack.c"), .flags = c_flags });
    // Our own shim: exports sizeof/offsetof of the real C structs so the
    // ds4_ffi.zig layout test catches mirror drift (mid-struct-insert class).
    module.addCSourceFile(.{ .file = b.path("src/ds4_layout_check.c"), .flags = c_flags });

    const objc_flags = &[_][]const u8{
        "-O3",
        "-ffast-math",
        "-fobjc-arc",
        "-Wno-unused-parameter",
        "-Wno-unused-variable",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-function",
        "-Wno-deprecated-declarations",
    };
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_metal.m"), .flags = objc_flags });
}

/// ANE prefill offload sources (lib/ane): the private-framework bridge and
/// the per-layer MLP program builder, both ARC objc. Runtime-probed —
/// compiling them in costs nothing on machines without the framework.
fn addAneSources(b: *std.Build, module: *std.Build.Module) void {
    const objc_flags = &[_][]const u8{
        "-O3",
        "-fobjc-arc",
        "-Wno-deprecated-declarations",
    };
    module.addCSourceFile(.{ .file = b.path("lib/ane/ane_bridge.m"), .flags = objc_flags });
    module.addCSourceFile(.{ .file = b.path("lib/ane/ane_mlp.m"), .flags = objc_flags });
    module.addIncludePath(b.path("lib/ane"));
}

fn buildRootHandle(b: *std.Build) std.Io.Dir {
    return b.root.root_dir.handle;
}

/// The llama.cpp tag staged by scripts/fetch-llama.sh (it writes LLAMA_TAG to
/// `lib/llama/.version`). Read at configure time so a plain `zig build` reports
/// the real tag without app/build.sh having to pass `--llama-tag`. Returns null
/// (→ "unknown") when llama hasn't been fetched yet.
fn readLlamaTag(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "lib/llama/.version",
        b.allocator,
        .limited(256),
    ) catch return null;
    const trimmed = std.mem.trim(u8, bytes, " \t\r\n");
    return if (trimmed.len == 0) null else b.dupe(trimmed);
}

fn addLlamaLib(b: *std.Build, module: *std.Build.Module, os_tag: std.Target.Os.Tag) void {
    module.addIncludePath(b.path("lib/llama/include"));

    switch (os_tag) {
        .windows => {
            // The prebuilt Windows release ships DLLs with NO import libraries,
            // so the link target IS the DLL: lld resolves it directly under the
            // gnu ABI (build.zig pins that above; msvc refuses with "bad file
            // type. Did you specify a DLL instead of an import library?").
            // lib/llama/bin therefore doubles as the link-time library path.
            //
            // There is no rpath on Windows: the loader searches the
            // executable's own directory, which is why build() installs the
            // whole DLL set beside mlx-serve.exe. That set must stay complete
            // even though we link only llama -- ggml.dll dlopens its backends
            // (ggml-cuda.dll, the per-microarch ggml-cpu-*.dll) BY FILENAME at
            // init, so a missing one silently downgrades the backend rather
            // than failing the link.
            module.addLibraryPath(b.path("lib/llama/bin"));
            module.linkSystemLibrary("llama", .{ .use_pkg_config = .no });
            // ggml is a SEPARATE DLL in the Windows release (the macOS
            // XCFramework merges llama + ggml + ggml-metal into one dylib, so
            // this has no macOS counterpart). `--version` reports
            // ggml_version()/ggml_commit(), which live in ggml-base.
            module.linkSystemLibrary("ggml-base", .{ .use_pkg_config = .no });
            // The backend REGISTRY (ggml_backend_load_all*) lives in ggml.dll,
            // not ggml-base.dll. The shim calls it explicitly at init rather
            // than relying on the registry's own auto-load, which searches the
            // running executable's directory -- correct for the installed
            // server, empty for a `zig build test` binary under .zig-cache.
            module.linkSystemLibrary("ggml", .{ .use_pkg_config = .no });
        },
        .macos => {
            module.addLibraryPath(b.path("lib/llama/lib"));
            // use_pkg_config = .no: a Homebrew `llama.cpp` install ships a llama.pc that
            // would otherwise hijack this link (pulling in /opt/homebrew's version + its
            // separate libggml). We want exactly the pinned dylib staged in lib/llama/lib.
            module.linkSystemLibrary("llama", .{ .use_pkg_config = .no });
            // Two stagings share this path. fetch-llama.sh's XCFramework is ONE
            // merged dylib (ggml inside). scripts/build-llama-macos.sh (Metal +
            // ggml RPC, rpc-offload-plan.md) stages the SEPARATED layout, where
            // ggml_version()/the backend registry live in libggml-base/libggml
            // -- link them when they are there. Probed at configure time; after
            // switching stagings `rm -rf .zig-cache`.
            if (llamaStagedSeparated(b)) {
                module.linkSystemLibrary("ggml", .{ .use_pkg_config = .no });
                module.linkSystemLibrary("ggml-base", .{ .use_pkg_config = .no });
            }
            // @loader_path resolves against the BINARY's own location at launch,
            // not the launching process's cwd. A bare relative string here (e.g.
            // "lib/llama/lib") gets baked verbatim into LC_RPATH when `zig build`
            // runs with cwd == build root (b.build_root.path is null in that case,
            // so b.path() can't make it absolute) and dyld then resolves that
            // relative string against argv[0]'s cwd, breaking any launch from
            // outside the repo root. An absolute path avoids that but bakes in a
            // machine-specific location, so the build isn't relocatable. This is
            // relative to the binary itself, so it stays correct from any launch
            // cwd and survives copying the whole zig-out + lib tree elsewhere.
            //
            // This module backs two different binaries at two different depths
            // under the build root, so one entry can't serve both: the installed
            // exe lands at zig-out/bin/mlx-serve (@loader_path = zig-out/bin/, 2
            // levels up to root), while `zig build test` runs straight out of
            // .zig-cache/o/<hash>/test (@loader_path = that dir, 3 levels up to
            // root). dyld tries every LC_RPATH entry in order and silently skips
            // ones that don't resolve, so listing both depths here is safe -- each
            // binary finds its own and ignores the other.
            module.addRPath(.{ .cwd_relative = "@loader_path/../../lib/llama/lib" });
            module.addRPath(.{ .cwd_relative = "@loader_path/../../../lib/llama/lib" });
        },
        else => {
            module.addLibraryPath(b.path("lib/llama/lib"));
            module.linkSystemLibrary("llama", .{ .use_pkg_config = .no });
            // Same split as Windows: a CMake-built llama.cpp keeps ggml in its
            // own shared objects, and the shim's explicit backend registration
            // (ggml_backend_load_all*) lives in libggml. `--version` reads
            // ggml_version()/ggml_commit() out of libggml-base.
            module.linkSystemLibrary("ggml", .{ .use_pkg_config = .no });
            module.linkSystemLibrary("ggml-base", .{ .use_pkg_config = .no });
            // ELF equivalent of @loader_path. Same two-depth reasoning as macOS.
            module.addRPath(.{ .cwd_relative = "$ORIGIN/../../lib/llama/lib" });
            module.addRPath(.{ .cwd_relative = "$ORIGIN/../../../lib/llama/lib" });
        },
    }

    // Our clean C shim over llama.h (src/llama_ffi.zig mirrors lib/llama_shim/llama_shim.h).
    // C11 for the one-time backend init.
    module.addIncludePath(b.path("lib/llama_shim"));
    module.addCSourceFile(.{
        .file = b.path("lib/llama_shim/llama_shim.c"),
        .flags = &.{ "-O2", "-std=c11", "-Wno-unused-parameter" },
    });
}

/// Link the self-built mlx + mlx-c staged in lib/mlx by scripts/build-mlx.sh
/// (pinned submodules lib/mlx-src + lib/mlxc-src, deployment target 26.2 so
/// MLX's NAX kernels are compiled in — the Homebrew bottle ships without them
/// and hard-wires is_nax_available() false even on M5). Install names are
/// @rpath/...; the build-tree rpath resolves them in dev, release.yml /
/// app/build.sh rewrite to @executable_path and re-sign for bundles.
/// Guard test: tests/test_mlx_staged_nax.sh.
fn addMlxLib(b: *std.Build, module: *std.Build.Module) void {
    module.addIncludePath(b.path("lib/mlx/include"));
    module.addLibraryPath(b.path("lib/mlx/lib"));
    // use_pkg_config = .no: a leftover Homebrew mlx-c must never hijack this
    // link — we want exactly the staged NAX-enabled pair (same class as the
    // llama.pc hijack above).
    module.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    // See addLlamaLib above: @loader_path is relative to the binary itself,
    // so this stays correct regardless of the launching process's cwd and
    // stays relocatable across machines. Two entries for the same reason —
    // the installed exe and the `zig build test` binary sit at different
    // depths under the build root.
    module.addRPath(.{ .cwd_relative = "@loader_path/../../lib/mlx/lib" });
    module.addRPath(.{ .cwd_relative = "@loader_path/../../../lib/mlx/lib" });
}

/// Configure-time check that scripts/build-mlx.sh has staged the pinned
/// mlx/mlx-c build. Mirrors verifyBrewDeps: fail loudly with the fix, never
/// let the linker produce a confusing -lmlxc error (or silently pick up a
/// leftover brew copy from /opt/homebrew/lib).
fn verifyMlxStage(b: *std.Build) void {
    const stage_ok = blk: {
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/libmlxc.dylib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/mlx.metallib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/.version", .{}) catch break :blk false;
        break :blk true;
    };
    if (!stage_ok) {
        std.debug.print(
            "\n[mlx-serve] lib/mlx is not staged (self-built mlx + mlx-c). Run:\n" ++
                "  git submodule update --init lib/mlx-src lib/mlxc-src && ./scripts/build-mlx.sh\n\n",
            .{},
        );
        std.process.exit(1);
    }
}

/// The pinned mlx-c revision from lib/mlx/.version (written by
/// scripts/build-mlx.sh as "mlx=<sha> mlxc=<sha> target=<ver>"), surfaced in
/// `mlx-serve --version`. Returns null (→ "unknown") when not staged yet.
/// `CFBundleShortVersionString` out of the checked-in app/Info.plist — the one
/// place the current CalVer is committed (app/build.sh stamps it on every real
/// build). Read at configure time so a plain `zig build` reports the same
/// version the last shipped build did, instead of a made-up literal.
fn readAppVersion(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "app/Info.plist",
        b.allocator,
        .limited(64 * 1024),
    ) catch return null;
    const key = "<key>CFBundleShortVersionString</key>";
    const at = std.mem.indexOf(u8, bytes, key) orelse return null;
    const open = std.mem.indexOfPos(u8, bytes, at + key.len, "<string>") orelse return null;
    const start = open + "<string>".len;
    const end = std.mem.indexOfPos(u8, bytes, start, "</string>") orelse return null;
    const v = std.mem.trim(u8, bytes[start..end], " \t\r\n");
    return if (v.len > 0) b.dupe(v) else null;
}

fn readMlxcPin(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "lib/mlx/.version",
        b.allocator,
        .limited(256),
    ) catch return null;
    var it = std.mem.tokenizeScalar(u8, std.mem.trim(u8, bytes, " \t\r\n"), ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, "mlxc=")) return b.dupe(tok["mlxc=".len..]);
    }
    return null;
}

const BrewDep = struct { name: []const u8, min: std.SemanticVersion };

const required_brew_deps = [_]BrewDep{
    // mlx + mlx-c are NOT brew deps anymore: they are pinned submodules built
    // by scripts/build-mlx.sh (see addMlxLib) so the NAX kernels ship enabled.
    .{ .name = "webp", .min = .{ .major = 1, .minor = 6, .patch = 0 } },
};

fn verifyBrewDeps(b: *std.Build) void {
    for (required_brew_deps) |dep| {
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "brew", "list", "--versions", dep.name },
            &code,
            .inherit,
        ) catch {
            std.debug.print(
                "\n[mlx-serve] missing Homebrew dependency '{s}' (>= {d}.{d}.{d}). Install with: brew install webp\n\n",
                .{ dep.name, dep.min.major, dep.min.minor, dep.min.patch },
            );
            std.process.exit(1);
        };
        const trimmed = std.mem.trim(u8, stdout, " \n\r\t");
        const space = std.mem.indexOfScalar(u8, trimmed, ' ') orelse {
            std.debug.print("[mlx-serve] cannot parse `brew list --versions {s}` output: {s}\n", .{ dep.name, trimmed });
            std.process.exit(1);
        };
        var ver_str = trimmed[space + 1 ..];
        // Strip Homebrew revision suffix (e.g., "0.6.0_2" -> "0.6.0").
        if (std.mem.indexOfScalar(u8, ver_str, '_')) |us| ver_str = ver_str[0..us];
        const have = std.SemanticVersion.parse(ver_str) catch {
            std.debug.print("[mlx-serve] cannot parse '{s}' version '{s}'\n", .{ dep.name, ver_str });
            std.process.exit(1);
        };
        if (have.order(dep.min) == .lt) {
            std.debug.print(
                "\n[mlx-serve] Homebrew '{s}' is {d}.{d}.{d}; need >= {d}.{d}.{d}. Run: brew upgrade {s}\n\n",
                .{ dep.name, have.major, have.minor, have.patch, dep.min.major, dep.min.minor, dep.min.patch, dep.name },
            );
            std.process.exit(1);
        }
    }
}

/// Everything the exe module and the test module share. They were hand-
/// duplicated before, which is how they drifted (the test module carried an
/// extra `linkSystemLibrary("c++")` the exe did not); one helper makes that
/// class of bug unbuildable.
const SharedOpts = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Step.Options,
    ds4_metal_sources: *std.Build.Module,
    is_darwin: bool,
    gguf_only: bool,
    macos_sdk_frameworks: ?[]const u8,
};

fn makeSharedModule(b: *std.Build, root: std.Build.LazyPath, o: SharedOpts) *std.Build.Module {
    const t = o.target;
    const opt = o.optimize;

    // webp: Homebrew's on macOS, our own stub everywhere else. `src/server.zig`
    // imports "webp" unconditionally, so the module must always exist -- see
    // lib/webp_stub/webp/decode.h for why a stub rather than a vendored copy.
    const webp_header: std.Build.LazyPath = if (o.is_darwin)
        .{ .cwd_relative = "/opt/homebrew/include/webp/decode.h" }
    else
        b.path("lib/webp_stub/webp/decode.h");
    const webp_include: std.Build.LazyPath = if (o.is_darwin)
        .{ .cwd_relative = "/opt/homebrew/include" }
    else
        b.path("lib/webp_stub");

    const mod = b.createModule(.{
        .root_source_file = root,
        .target = t,
        .optimize = opt,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = o.build_options.createModule() },
            .{ .name = "ds4_metal_sources", .module = o.ds4_metal_sources },
            .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), t, opt, "") },
            .{ .name = "stb", .module = addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), t, opt, "") },
            .{ .name = "webp", .module = addCHeaderModule(b, webp_header, webp_include, t, opt, "") },
        },
    });

    // Jinja2 template engine (from wangzhaode's jinja.cpp + nlohmann/json).
    //
    // macOS links the pre-compiled static archive checked in at
    // lib/jinja_cpp/libjinja.a. That archive is a Mach-O and cannot serve any
    // other host, so every other target compiles the same sources from scratch
    // through Zig's bundled clang -- which also removes the separate rebuild
    // step. Rebuild the macOS archive with: cd lib/jinja_cpp && for f in
    // jinja_wrapper caps lexer parser runtime jinja_string value; do clang++
    // -std=c++17 -O2 -DNDEBUG -I . -c $f.cpp -o obj/$f.o; done && ar rcs
    // libjinja.a obj/*.o
    if (o.is_darwin) {
        mod.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
    } else {
        const jinja_flags = &[_][]const u8{ "-std=c++17", "-O2", "-DNDEBUG" };
        for ([_][]const u8{
            "jinja_wrapper", "caps", "lexer", "parser", "runtime", "jinja_string", "value",
        }) |name| {
            mod.addCSourceFile(.{
                .file = b.path(b.fmt("lib/jinja_cpp/{s}.cpp", .{name})),
                .flags = jinja_flags,
            });
        }
    }
    mod.addIncludePath(b.path("lib/jinja_cpp"));

    // stb_image for JPEG/PNG decoding in the vision pipeline
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    // stb_image_write for PNG encoding (native image-generation endpoint)
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_write_impl.c"), .flags = &.{"-O2"} });
    mod.addIncludePath(b.path("lib"));

    if (!o.is_darwin) {
        mod.addCSourceFile(.{ .file = b.path("lib/webp_stub/webp_stub.c"), .flags = &.{"-O2"} });
        mod.addIncludePath(b.path("lib/webp_stub"));
    }

    // xatlas UV unwrapping (MIT, vendored amalgamation) + C shim for the
    // Hunyuan3D texture paint stage. See lib/xatlas/xatlas_shim.h + src/uvwrap.zig.
    // Portable C++ with no Apple dependency, and src/uvwrap.zig's tests are
    // hermetic (zero MLX), so it is compiled in every configuration.
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addIncludePath(b.path("lib/xatlas"));

    if (!o.gguf_only) {
        // ds4 inference engine for DSV4-Flash (Metal backend, macOS only). See
        // `lib/ds4/` submodule pinned at 613e9b2 and `src/arch/ds4.zig`. Kernel
        // sources are embedded via `lib/ds4_metal_sources.zig` and extracted at
        // runtime to ~/.mlx-serve/ds4-metal/<hash>/.
        addDs4Sources(b, mod);
        mod.addIncludePath(b.path("lib/ds4"));

        // ANE prefill-MLP offload (perf-plan-aug-17 P5): objc bridge to the
        // private AppleNeuralEngine framework (dlopen'd at runtime -- the probe
        // returns unavailable on machines/OSes without it) + the per-layer MLP
        // MIL program builder. See lib/ane/ + src/ane.zig; provenance in NOTICE.
        addAneSources(b, mod);
    }

    // llama.cpp libllama for generic GGUF models. Staged by
    // scripts/fetch-llama.sh into lib/llama/. On macOS that is a single
    // self-contained dylib from the XCFramework (Metal backend); on Windows it
    // is the prebuilt CUDA DLL set. See src/arch/llama.zig.
    addLlamaLib(b, mod, t.result.os.tag);

    if (!o.gguf_only) {
        // mlx + mlx-c: self-built from the pinned submodules (lib/mlx-src,
        // lib/mlxc-src) into lib/mlx by scripts/build-mlx.sh, with NAX kernels
        // enabled (the Homebrew bottle ships without them). MUST come before the
        // /opt/homebrew lib path so a leftover brew mlx-c can never win the link.
        addMlxLib(b, mod);
    }

    if (o.is_darwin) {
        // webp include/lib paths (homebrew)
        mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
        mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
        mod.linkSystemLibrary("webp", .{});

        if (o.macos_sdk_frameworks) |fw_path| {
            mod.addFrameworkPath(.{ .cwd_relative = fw_path });
        }
        mod.linkFramework("IOKit", .{});
        mod.linkFramework("CoreFoundation", .{});
        mod.linkFramework("Foundation", .{});
        // Metal and IOSurface are only referenced by the ds4 Metal backend and
        // the ANE bridge, both of which a -Dgguf-only macOS build drops.
        if (!o.gguf_only) {
            mod.linkFramework("Metal", .{});
            mod.linkFramework("IOSurface", .{});
        }
    }

    return mod;
}

/// Configure-time check that scripts/fetch-llama.sh has staged libllama for
/// this host. Mirrors verifyMlxStage: fail loudly with the fix rather than let
/// the linker emit a bare -lllama error. GGUF is the ONLY engine off Apple, so
/// a missing stage there is fatal rather than a degraded build.
/// True when lib/llama/lib holds the CMake-built separated layout (libggml-base
/// beside libllama) rather than the merged XCFramework dylib.
fn llamaStagedSeparated(b: *std.Build) bool {
    buildRootHandle(b).access(b.graph.io, "lib/llama/lib/libggml-base.dylib", .{}) catch return false;
    return true;
}

fn verifyLlamaStage(b: *std.Build, os_tag: std.Target.Os.Tag) void {
    const probe = switch (os_tag) {
        .macos => "lib/llama/lib/libllama.dylib",
        .windows => "lib/llama/bin/llama.dll",
        else => "lib/llama/lib/libllama.so",
    };
    buildRootHandle(b).access(b.graph.io, probe, .{}) catch {
        std.debug.print(
            "\n[mlx-serve] lib/llama is not staged ({s} missing). Run:\n" ++
                "  ./scripts/fetch-llama.sh\n\n",
            .{probe},
        );
        std.process.exit(1);
    };
}

/// Copy every DLL staged in lib/llama/bin into the install prefix's bin dir.
/// Enumerated at CONFIGURE time from what fetch-llama.sh actually staged, so a
/// llama.cpp release that adds or renames a backend DLL needs no edit here.
fn installLlamaDlls(b: *std.Build) void {
    var dir = buildRootHandle(b).openDir(b.graph.io, "lib/llama/bin", .{ .iterate = true }) catch return;
    defer dir.close(b.graph.io);
    var it = dir.iterate();
    while (it.next(b.graph.io) catch return) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".dll")) continue;
        b.getInstallStep().dependOn(&b.addInstallBinFile(
            b.path(b.fmt("lib/llama/bin/{s}", .{entry.name})),
            b.dupe(entry.name),
        ).step);
    }
}
