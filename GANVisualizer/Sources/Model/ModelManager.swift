import Foundation
import CoreML

struct HFModel: Identifiable, Hashable {
    let name: String
    let size: Int
    let url: URL
    var id: String { name }

    var label: String {
        name
            .replacingOccurrences(of: ".onnx", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "_EndToEndNetwork", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "_", with: " ")
    }

    var sizeLabel: String {
        size > 0 ? String(format: "%.1f MB", Double(size) / 1_048_576) : ""
    }
}

/// Discovers models on the Hugging Face Hub, downloads them, converts ONNX ->
/// CoreML (via the bundled Python converter), caches the compiled model, and
/// hands it to the `InferenceEngine`.
@MainActor
final class ModelManager: ObservableObject {

    static let hfRepo = "lorenzsichert/gan-visualizer"
    static let hfBase = "https://huggingface.co/\(hfRepo)/resolve/main/"
    static let fallback = ["abstract_art.onnx", "abstract_photo.onnx",
                           "abstract_photo_big.onnx", "flowers_big.onnx"]

    @Published var models: [HFModel] = []
    @Published var selected: HFModel?
    @Published var phase: String = "idle"
    @Published var statusText: String = "initializing…"
    @Published var progress: Double?
    @Published var busy = false
    @Published var errorMessage: String?

    let precision = "float16"

    lazy var rootDir: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("GANVisualizer", isDirectory: true)
    }()

    func cacheDir(for m: HFModel) -> URL {
        let slug = m.name.replacingOccurrences(of: ".onnx", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: ".", with: "_")
        return rootDir.appendingPathComponent("Models/\(slug)", isDirectory: true)
    }

    // MARK: - Discovery

    func refresh() async {
        let treeURL = URL(string: "https://huggingface.co/api/models/\(Self.hfRepo)/tree/main")!
        var found: [HFModel] = []
        do {
            let (data, _) = try await URLSession.shared.data(from: treeURL)
            if let arr = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                for e in arr {
                    guard let path = e["path"] as? String, path.lowercased().hasSuffix(".onnx") else { continue }
                    let size = e["size"] as? Int ?? 0
                    guard let url = URL(string: Self.hfBase + path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)!) else { continue }
                    found.append(HFModel(name: path, size: size, url: url))
                }
            }
        } catch {
            NSLog("HF discovery failed: \(error)")
        }
        if found.isEmpty {
            found = Self.fallback.compactMap { name in
                URL(string: Self.hfBase + name).map { HFModel(name: name, size: 0, url: $0) }
            }
        }
        models = found.sorted { $0.name < $1.name }
    }

    // MARK: - Load / convert

    func ensureLoaded(_ m: HFModel, engine: InferenceEngine) async {
        busy = true
        errorMessage = nil
        defer { busy = false }

        let dir = cacheDir(for: m)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let onnx = dir.appendingPathComponent("source.onnx")
        let mlpackage = dir.appendingPathComponent("model.mlpackage")
        let compiled = dir.appendingPathComponent("model.mlmodelc")

        do {
            if !FileManager.default.fileExists(atPath: compiled.path) {
                if !FileManager.default.fileExists(atPath: onnx.path) {
                    set(phase: "download", text: "Downloading \(m.name)…", progress: 0)
                    try await Downloader.download(url: m.url, to: onnx) { [weak self] got, total in
                        let frac = total > 0 ? Double(got) / Double(total) : nil
                        Task { @MainActor in self?.set(phase: "download", text: "Downloading \(m.name)…", progress: frac) }
                    }
                }
                set(phase: "convert", text: "Converting to CoreML…", progress: nil)
                try await Converter.run(onnxURL: onnx, outDir: dir, precision: precision) { [weak self] p in
                    Task { @MainActor in self?.set(phase: p.phase, text: p.text, progress: p.progress) }
                }
                set(phase: "compile", text: "Compiling model…", progress: nil)
                let tmp = try await MLModel.compileModel(at: mlpackage)
                try? FileManager.default.removeItem(at: compiled)
                try FileManager.default.moveItem(at: tmp, to: compiled)
            }

            set(phase: "load", text: "Loading \(m.name)…", progress: nil)
            try engine.load(compiledModelURL: compiled, computeUnits: .all)
            selected = m
            set(phase: "idle", text: "model ready", progress: nil)
        } catch {
            errorMessage = error.localizedDescription
            set(phase: "idle", text: "failed: \(error.localizedDescription)", progress: nil)
            NSLog("ensureLoaded failed: \(error)")
        }
    }

    private func set(phase: String, text: String, progress: Double?) {
        self.phase = phase
        self.statusText = text
        self.progress = progress
    }
}

/// Minimal URLSession download with progress, single-shot.
final class Downloader: NSObject, URLSessionDownloadDelegate {
    private var continuation: CheckedContinuation<URL, Error>?
    private var destination: URL?
    private var onProgress: ((Int64, Int64) -> Void)?

    static func download(url: URL, to dest: URL,
                         onProgress: @escaping (Int64, Int64) -> Void) async throws -> URL {
        let d = Downloader()
        d.destination = dest
        d.onProgress = onProgress
        return try await withCheckedThrowingContinuation { (cont: CheckedContinuation<URL, Error>) in
            d.continuation = cont
            let session = URLSession(configuration: .default, delegate: d, delegateQueue: nil)
            session.downloadTask(with: url).resume()
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        onProgress?(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        guard let dest = destination else { return }
        do {
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try FileManager.default.moveItem(at: location, to: dest)
            continuation?.resume(returning: dest)
        } catch {
            continuation?.resume(throwing: error)
        }
        continuation = nil
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error, continuation != nil {
            continuation?.resume(throwing: error)
            continuation = nil
        }
    }
}
