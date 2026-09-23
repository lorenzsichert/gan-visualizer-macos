import Foundation

enum ConverterError: Error, LocalizedError {
    case scriptMissing
    case launchFailed(String)
    case failed(String)

    var errorDescription: String? {
        switch self {
        case .scriptMissing: return "Converter script is missing from the app bundle."
        case .launchFailed(let m): return "Could not launch the converter: \(m)"
        case .failed(let m): return "Conversion failed: \(m)"
        }
    }
}

/// Runs the bundled Python `convert_model.py` (Hugging Face ONNX -> CoreML
/// `.mlpackage`), streaming its JSON progress lines back to the caller.
enum Converter {

    struct ProgressUpdate {
        var phase: String
        var progress: Double?
        var text: String
    }

    static func scriptURL() -> URL? {
        if let u = Bundle.main.url(forResource: "convert_model", withExtension: "py") { return u }
        // Fallback for running from a build directory / source tree.
        let candidates = [
            "macos/GANVisualizer/Converter/convert_model.py",
            "../macos/GANVisualizer/Converter/convert_model.py",
        ]
        for c in candidates {
            let u = URL(fileURLWithPath: c)
            if FileManager.default.fileExists(atPath: u.path) { return u }
        }
        return nil
    }

    /// Run the converter on a local ONNX file. `onProgress` is called on a
    /// background queue.
    static func run(onnxURL: URL, outDir: URL, precision: String,
                    onProgress: @escaping (ProgressUpdate) -> Void) async throws {
        guard let script = scriptURL() else { throw ConverterError.scriptMissing }
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        let python = ProcessInfo.processInfo.environment["GAN_CONVERTER_PYTHON"] ?? "python3"
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = [python, script.path,
                          "--onnx", onnxURL.path,
                          "--out", outDir.path,
                          "--precision", precision,
                          "--cover"]

        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        let errPipe = Pipe()
        proc.standardError = errPipe

        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try proc.run()
                } catch {
                    cont.resume(throwing: ConverterError.launchFailed(error.localizedDescription))
                    return
                }
                var buffer = Data()
                let handle = pipe.fileHandleForReading
                while true {
                    let data = handle.availableData
                    if data.isEmpty { break }
                    buffer.append(data)
                    while let nl = buffer.firstIndex(of: 0x0A) {
                        let lineData = buffer.subdata(in: buffer.startIndex..<nl)
                        buffer.removeSubrange(buffer.startIndex...nl)
                        if let line = String(data: lineData, encoding: .utf8) {
                            Self.handle(line: line, onProgress: onProgress)
                        }
                    }
                }
                proc.waitUntilExit()
                if proc.terminationStatus != 0 {
                    let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                    let err = String(data: errData, encoding: .utf8) ?? "exit \(proc.terminationStatus)"
                    cont.resume(throwing: ConverterError.failed(String(err.suffix(600))))
                } else {
                    cont.resume()
                }
            }
        }
    }

    private static func handle(line: String, onProgress: (ProgressUpdate) -> Void) {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("{"), let data = trimmed.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            NSLog("converter: \(trimmed)")
            return
        }
        let phase = obj["phase"] as? String ?? "working"
        let text = obj["text"] as? String ?? phase
        let progress = obj["progress"] as? Double
        onProgress(ProgressUpdate(phase: phase, progress: progress, text: text))
    }
}
