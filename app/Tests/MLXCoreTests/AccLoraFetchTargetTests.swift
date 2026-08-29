import XCTest
@testable import MLXCore

/// Acc is a DIFFERENT Hugging Face repo, downloaded INTO the H3 pack dir.
///
/// Same fragment-dir class as Turbo: writing Acc into `models/alibaba-pai/…`
/// would register a fake model. The file belongs beside the pack's DiT.
/// Cancel must never wipe the pack.
@MainActor
final class AccLoraFetchTargetTests: XCTestCase {
    private var tempRoot: String!
    private var savedSession: URLSession!

    private let packRepoId = "ddalcu/fixture-h3-pack"
    private let configBody = Data("{\"model_type\":\"minimax_h3\"}".utf8)
    private let accFile = AccLoraFetch.fl2vaFileName

    override func setUpWithError() throws {
        tempRoot = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-serve-acc-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: tempRoot, withIntermediateDirectories: true)
        savedSession = DownloadSession.shared
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [HuggingFaceStubProtocol.self]
        config.httpMaximumConnectionsPerHost = 20
        DownloadSession.shared = URLSession(configuration: config)
        HuggingFaceStubProtocol.reset()
    }

    override func tearDownWithError() throws {
        DownloadSession.shared = savedSession
        HuggingFaceStubProtocol.reset()
        try? FileManager.default.removeItem(atPath: tempRoot)
    }

    private func makePack(at dir: String) throws {
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try configBody.write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("config.json")))
        try Data("REAL-WEIGHTS".utf8).write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("transformer.safetensors")))
        try Data("tok".utf8).write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("tokenizer.json")))
    }

    private func serveAccRepo() {
        HuggingFaceStubProtocol.serve(repo: AccLoraFetch.hfRepoId, files: [
            ("README.md", Data("readme".utf8)),
            (accFile, Data("ACC".utf8)),
            (AccLoraFetch.ref2vaFileName, Data("WRONG-PARTITION".utf8)),
        ])
    }

    func testAdapterLandsBesideThePackNotInTheAccRepoDir() async throws {
        let packDir = (tempRoot as NSString).appendingPathComponent("fixture-h3-pack")
        try makePack(at: packDir)
        serveAccRepo()
        let manager = DownloadManager(modelsRoot: tempRoot)

        await withCheckedContinuation { cont in
            manager.startAccLora(packRepoId: packRepoId, fileName: accFile) { cont.resume() }
        }

        let fm = FileManager.default
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: (packDir as NSString).appendingPathComponent(accFile))),
                       Data("ACC".utf8), "Acc file did not land beside the pack")
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: (packDir as NSString).appendingPathComponent("transformer.safetensors"))),
                       Data("REAL-WEIGHTS".utf8), "the pack's weights must be untouched")
        XCTAssertFalse(fm.fileExists(atPath: (packDir as NSString).appendingPathComponent("README.md")),
                       "Acc-repo extras must not land in the pack")
        XCTAssertFalse(fm.fileExists(atPath: DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: AccLoraFetch.hfRepoId)),
                       "fetch created a fragment dir at the Acc HF layout path")
        XCTAssertFalse(fm.fileExists(atPath: DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: packRepoId)),
                       "fetch created a fragment dir at the pack destination layout path")
    }

    func testCancelWithoutAFetchIsANoOpOnTheLivePack() throws {
        let packDir = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: packRepoId)
        try makePack(at: packDir)
        let manager = DownloadManager(modelsRoot: tempRoot)

        manager.cancelAccLora(packRepoId: packRepoId)

        XCTAssertTrue(FileManager.default.fileExists(atPath: (packDir as NSString).appendingPathComponent("transformer.safetensors")),
                      "an idle Acc cancel deleted the pack")
    }

    func testCancelMidFetchLeavesThePackAndNoPartials() async throws {
        let packDir = (tempRoot as NSString).appendingPathComponent("fixture-h3-pack")
        try makePack(at: packDir)
        serveAccRepo()
        let manager = DownloadManager(modelsRoot: tempRoot)

        let done = expectation(description: "fetch settled")
        manager.startAccLora(packRepoId: packRepoId, fileName: accFile) { done.fulfill() }
        manager.cancelAccLora(packRepoId: packRepoId)
        await fulfillment(of: [done], timeout: 10)

        let fm = FileManager.default
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: (packDir as NSString).appendingPathComponent("transformer.safetensors"))),
                       Data("REAL-WEIGHTS".utf8))
        XCTAssertTrue(fm.fileExists(atPath: (packDir as NSString).appendingPathComponent("config.json")))
        let strays = (try? fm.contentsOfDirectory(atPath: packDir).filter { $0.contains(".partial") }) ?? []
        XCTAssertTrue(strays.isEmpty, "left behind: \(strays)")
        XCTAssertFalse(fm.fileExists(atPath: DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: AccLoraFetch.hfRepoId)))
    }
}
