import Foundation
import CoreML

/// A single file that makes up a hosted CoreML model package. `path` is
/// relative to the root of the Hugging Face repository.
struct HFFile: Hashable, Codable {
    let path: String
    let size: Int
}

struct HFModel: Identifiable, Hashable, Codable {
    /// Folder name under `coreml/`, e.g. `abstract_photo_big`.
    let name: String
    /// Repository-relative path of the model's folder, e.g. `coreml/abstract_photo_big`.
    let path: String
    /// Total download size of all files in the folder, in bytes.
    let size: Int
    /// The files to download (package manifest, spec and weights, plus metadata).
    let files: [HFFile]

    var id: String { name }

    var label: String {
        name.replacingOccurrences(of: "_", with: " ")
    }

    var sizeLabel: String {
        size > 0 ? String(format: "%.1f MB", Double(size) / 1_048_576) : ""
    }
}

/// Author info for one generator family, loaded from `model-credits.json` on
/// the Hub (or the built-in fallback).
struct ModelCredit: Hashable {
    let author: String?
    let links: [String]
}

/// Discovers ready-made CoreML models in the Hugging Face Hub's `coreml/`
/// folder, downloads their `.mlpackage` files, compiles them to `.mlmodelc`,
/// caches the result and hands it to the `InferenceEngine`.
@MainActor
final class ModelManager: ObservableObject {

    static let hfRepo = "lorenzsichert/gan-visualizer"
    static let hfBase = "https://huggingface.co/\(hfRepo)/resolve/main/"
    static let hfTree = "https://huggingface.co/api/models/\(hfRepo)/tree/main/coreml?recursive=true"
    static let hfInfo = "https://huggingface.co/api/models/\(hfRepo)"
    static let hfCredits = "https://huggingface.co/\(hfRepo)/resolve/main/model-credits.json"

    /// Fallback credits used when the Hub file is missing or unreachable.
    static let fallbackCredits: [String: ModelCredit] = [
        "abstract_art": ModelCredit(author: "Michael Friesen",
                                    links: ["https://x.com/MichaelFriese10"]),
        "abstract_photo": ModelCredit(author: "Michael Friesen",
                                      links: ["https://x.com/MichaelFriese10"]),
        "flowers": ModelCredit(author: "Justin Pinkney",
                               links: ["https://www.justinpinkney.com"]),
        "microscope": ModelCredit(author: "Michael Friesen",
                                  links: ["https://x.com/MichaelFriese10"]),
        "textures": ModelCredit(author: "Bernat Cuni",
                                links: ["https://www.cunicode.com"]),
        "trypophobia": ModelCredit(author: "Sid Black",
                                   links: ["https://x.com/realmeatyhuman"]),
        "visionary": ModelCredit(author: "Jeremy Torman",
                                 links: ["https://x.com/TormanJeremy"]),
        "wikiart": ModelCredit(author: "Peter Baylies",
                               links: ["https://x.com/pbaylies"]),
        "wildlife": ModelCredit(author: "Michael Friesen",
                                links: ["https://x.com/MichaelFriese10"]),
    ]

    @Published var models: [HFModel] = []
    @Published var credits: [String: ModelCredit] = ModelManager.fallbackCredits
    @Published var selected: HFModel?
    @Published var phase: String = "idle"
    @Published var statusText: String = "initializing…"
    @Published var progress: Double?
    @Published var busy = false
    @Published var errorMessage: String?

    lazy var rootDir: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("GANVisualizer", isDirectory: true)
    }()

    func cacheDir(for m: HFModel) -> URL {
        let slug = m.name.replacingOccurrences(of: "/", with: "_")
        return rootDir.appendingPathComponent("Models/\(slug)", isDirectory: true)
    }

    /// Per-model cache of the discovered brightness direction.
    func brightnessURL(for m: HFModel) -> URL {
        cacheDir(for: m).appendingPathComponent("brightness-direction.json")
    }

    // MARK: - Discovery

    func refresh() async {
        let creditsTask = Task { await Self.discoverCredits() }
        var found: [HFModel] = []

        // 1. Full recursive listing of `coreml/` — includes file sizes for the
        //    progress bar. This is the authoritative source. The Hub pages the
        //    listing at 100 entries, so follow `rel="next"` until exhausted.
        do {
            let entries = try await fetchTree()
            found = Self.models(fromTree: entries)
        } catch {
            NSLog("HF tree discovery failed: \(error)")
        }

        // 2. The model-info endpoint lists every file in the repo too. It has
        //    no sizes, but it keeps the picker accurate if the tree API is
        //    rate-limited or temporarily unavailable.
        if found.isEmpty {
            do {
                if let url = URL(string: Self.hfInfo) {
                    let (data, response) = try await URLSession.shared.data(from: url)
                    try Self.requireOK(response)
                    if let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let siblings = object["siblings"] as? [[String: Any]] {
                        found = Self.models(fromSiblings: siblings)
                    }
                }
            } catch {
                NSLog("HF model-info discovery failed: \(error)")
            }
        }

        // 3. Last resort: the list discovered on a previous launch. Never a
        //    hardcoded list — new models on the Hub must show up automatically.
        if found.isEmpty { found = cachedModels() }
        if !found.isEmpty { saveCache(found) }
        models = found.sorted { $0.name < $1.name }
        credits = await creditsTask.value
    }

    /// Load author info from `model-credits.json` on the Hub. Accepts both the
    /// repo's `{ "credits": { ... } }` schema and a flat map, and falls back to
    /// the built-in table when the file is missing or malformed.
    private static func discoverCredits() async -> [String: ModelCredit] {
        guard let url = URL(string: hfCredits) else { return fallbackCredits }
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            try requireOK(response)
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return fallbackCredits
            }
            let raw = (object["credits"] as? [String: Any]) ?? object
            var parsed: [String: ModelCredit] = [:]
            for (base, value) in raw {
                guard let entry = value as? [String: Any] else { continue }
                let author = (entry["author"] as? String)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let links = (entry["links"] as? [String])?
                    .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty } ?? []
                let cleanAuthor = (author?.isEmpty == false) ? author : nil
                if cleanAuthor != nil || !links.isEmpty {
                    parsed[base] = ModelCredit(author: cleanAuthor, links: links)
                }
            }
            return parsed.isEmpty ? fallbackCredits : parsed
        } catch {
            return fallbackCredits
        }
    }

    private func fetchTree() async throws -> [[String: Any]] {
        var entries: [[String: Any]] = []
        var next = URL(string: Self.hfTree)
        var visited = Set<URL>()
        while let url = next, visited.insert(url).inserted {
            let (data, response) = try await URLSession.shared.data(from: url)
            try Self.requireOK(response)
            guard let page = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { break }
            entries.append(contentsOf: page)
            next = Self.nextPage(from: response)
        }
        return entries
    }

    /// The Hub returns `Link: <...cursor=...>; rel="next"` when a tree listing
    /// is paginated. Following it is what keeps newly added models visible.
    private static func nextPage(from response: URLResponse) -> URL? {
        guard let http = response as? HTTPURLResponse,
              let link = http.value(forHTTPHeaderField: "Link") else { return nil }
        for raw in link.split(separator: ",") {
            let part = raw.trimmingCharacters(in: .whitespaces)
            guard part.contains("rel=\"next\""),
                  let open = part.firstIndex(of: "<"), let close = part.firstIndex(of: ">") else { continue }
            return URL(string: String(part[part.index(after: open)..<close]))
        }
        return nil
    }

    private static func requireOK(_ response: URLResponse) throws {
        let code = (response as? HTTPURLResponse)?.statusCode ?? -1
        guard code == 200 else {
            throw NSError(domain: "HuggingFace", code: code,
                          userInfo: [NSLocalizedDescriptionKey: "Hub returned HTTP \(code)"])
        }
    }

    /// Group the recursive `coreml/` tree listing into one model per folder.
    /// Only folders containing a `model.mlpackage` are selectable.
    private static func models(fromTree entries: [[String: Any]]) -> [HFModel] {
        let prefix = "coreml/"
        var groups: [String: [HFFile]] = [:]
        for entry in entries {
            guard let path = entry["path"] as? String, path.hasPrefix(prefix),
                  entry["type"] as? String == "file" else { continue }
            let relative = path.dropFirst(prefix.count)
            let folder = relative.split(separator: "/", maxSplits: 1).first.map(String.init) ?? ""
            guard !folder.isEmpty else { continue }
            let size = entry["size"] as? Int ?? 0
            groups[folder, default: []].append(HFFile(path: path, size: size))
        }
        return groups.compactMap { folder, files in
            guard files.contains(where: { $0.path.contains("model.mlpackage/") }) else { return nil }
            return HFModel(name: folder,
                           path: prefix + folder,
                           size: files.reduce(0) { $0 + $1.size },
                           files: files.sorted { $0.path < $1.path })
        }
    }

    /// Build the same list from the `siblings` of the model-info response.
    /// Sizes aren't reported there, so progress is simply indeterminate.
    private static func models(fromSiblings siblings: [[String: Any]]) -> [HFModel] {
        let prefix = "coreml/"
        var groups: [String: [HFFile]] = [:]
        for entry in siblings {
            guard let path = entry["rfilename"] as? String, path.hasPrefix(prefix) else { continue }
            let relative = path.dropFirst(prefix.count)
            let folder = relative.split(separator: "/", maxSplits: 1).first.map(String.init) ?? ""
            guard !folder.isEmpty else { continue }
            groups[folder, default: []].append(HFFile(path: path, size: 0))
        }
        return groups.compactMap { folder, files in
            guard files.contains(where: { $0.path.contains("model.mlpackage/") }) else { return nil }
            return HFModel(name: folder,
                           path: prefix + folder,
                           size: 0,
                           files: files.sorted { $0.path < $1.path })
        }
    }

    // MARK: - Discovery cache

    private var indexURL: URL {
        rootDir.appendingPathComponent("Models/discovered-models.json")
    }

    private func saveCache(_ found: [HFModel]) {
        guard let data = try? JSONEncoder().encode(found) else { return }
        let dir = rootDir.appendingPathComponent("Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try? data.write(to: indexURL)
    }

    private func cachedModels() -> [HFModel] {
        guard let data = try? Data(contentsOf: indexURL),
              let found = try? JSONDecoder().decode([HFModel].self, from: data) else { return [] }
        return found
    }

    // MARK: - Load

    func ensureLoaded(_ m: HFModel, engine: InferenceEngine) async {
        busy = true
        errorMessage = nil
        defer { busy = false }

        let fm = FileManager.default
        let dir = cacheDir(for: m)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let mlpackage = dir.appendingPathComponent("model.mlpackage")
        let compiled = dir.appendingPathComponent("model.mlmodelc")

        do {
            if !fm.fileExists(atPath: compiled.path) {
                if !isCached(m, in: dir) {
                    // Drop any stale/partial package (including artifacts left
                    // by older versions of the app) before downloading.
                    try? fm.removeItem(at: mlpackage)
                    setStatus(phase: "download", text: "Downloading \(m.name)…", progress: 0)
                    try await download(m, into: dir)
                }
                setStatus(phase: "compile", text: "Compiling model…", progress: nil)
                let tmp = try await MLModel.compileModel(at: mlpackage)
                try? fm.removeItem(at: compiled)
                try fm.moveItem(at: tmp, to: compiled)
            }

            setStatus(phase: "load", text: "Loading \(m.name)…", progress: nil)
            try engine.load(compiledModelURL: compiled, computeUnits: .all)
            selected = m
            setStatus(phase: "idle", text: "model ready", progress: nil)
        } catch {
            errorMessage = error.localizedDescription
            setStatus(phase: "idle", text: "failed: \(error.localizedDescription)", progress: nil)
            NSLog("ensureLoaded failed: \(error)")
        }
    }

    /// True when every file is present in the cache (with the expected size
    /// when the Hub reported one).
    private func isCached(_ m: HFModel, in dir: URL) -> Bool {
        let fm = FileManager.default
        for file in m.files {
            let local = dir.appendingPathComponent(relativePath(of: file, in: m))
            guard let attrs = try? fm.attributesOfItem(atPath: local.path) else { return false }
            if file.size > 0 {
                guard let onDisk = attrs[.size] as? NSNumber,
                      onDisk.int64Value == Int64(file.size) else { return false }
            }
        }
        return true
    }

    private func relativePath(of file: HFFile, in model: HFModel) -> String {
        let prefix = model.path + "/"
        if file.path.hasPrefix(prefix) { return String(file.path.dropFirst(prefix.count)) }
        return (file.path as NSString).lastPathComponent
    }

    private func download(_ m: HFModel, into dir: URL) async throws {
        let fm = FileManager.default
        let files = m.files.sorted { $0.size < $1.size }  // small files first
        let total = files.reduce(Int64(0)) { $0 + Int64($1.size) }
        var completed: Int64 = 0

        for file in files {
            let local = dir.appendingPathComponent(relativePath(of: file, in: m))
            let size = Int64(file.size)
            if size > 0, let attrs = try? fm.attributesOfItem(atPath: local.path),
               let onDisk = attrs[.size] as? NSNumber, onDisk.int64Value == size {
                completed += size
                continue
            }
            try fm.createDirectory(at: local.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)
            let prefixBytes = completed
            let encoded = file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
            guard let url = URL(string: Self.hfBase + encoded) else { continue }
            _ = try await Downloader.download(url: url, to: local) { [weak self] written, expected in
                let known = size > 0 ? size : expected
                let denominator = total > 0 ? total : known
                let fraction = denominator > 0
                    ? min(1, max(0, Double(prefixBytes + written) / Double(denominator)))
                    : nil
                Task { @MainActor in
                    self?.setStatus(phase: "download", text: "Downloading \(m.name)…", progress: fraction)
                }
            }
            completed += size
        }
    }

    func setStatus(phase: String, text: String, progress: Double?) {
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
