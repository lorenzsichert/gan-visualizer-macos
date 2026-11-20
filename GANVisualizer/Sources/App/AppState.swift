import Foundation
import SwiftUI
import Metal
import AppKit
import Combine

@MainActor
final class AppState: ObservableObject {

    let settings = SettingsStore()
    let audio = AudioPipeline()
    let latentEngine = LatentEngine(dim: 512)
    let modelManager = ModelManager()

    var engine: InferenceEngine!
    var renderer: Renderer!

    private var cancellables = Set<AnyCancellable>()

    @Published var fps: Double = 0
    @Published var inferMs: Double = 0
    @Published var input: AudioInput = .off
    @Published var inputError: String?
    @Published var mirrorMode: Int = 2
    @Published var panelCollapsed = false
    @Published var modelReady = false
    @Published var paused = false
    @Published var brightnessReady = false

    private var pulse = [Float](repeating: 0, count: LatentEngine.pulseDim)

    init() {
        // Settings changes (sliders, spectrum drags) must refresh the panel even
        // though the store is not in the SwiftUI environment itself.
        settings.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)

        if let device = MTLCreateSystemDefaultDevice(),
           let eng = InferenceEngine(device: device),
           let rend = Renderer(device: device, engine: eng) {
            engine = eng
            renderer = rend
            rend.onFrameTick = { [weak self] dt, now in self?.tick(dt: dt, now: now) }
            rend.onStats = { [weak self] fps, infer in
                self?.fps = fps
                self?.inferMs = infer
            }
            eng.onError = { [weak self] msg in
                Task { @MainActor in self?.modelManager.errorMessage = msg }
            }
            audio.onError = { [weak self] msg in
                Task { @MainActor in self?.inputError = msg }
            }
        }
        Task { [weak self] in
            guard let self else { return }
            await self.modelManager.refresh()
            if let first = self.modelManager.models.first {
                await self.load(first)
            }
        }
        if let mode = ProcessInfo.processInfo.environment["GAN_LEAK"] {
            if mode != "rss" {
                Task { [weak self] in
                    var lastGen = -1
                    while true {
                        guard let self else { return }
                        if mode != "infer" {
                            _ = self.latentEngine.computeLatent(
                                dt: 1.0 / 60.0, nowSec: ProcessInfo.processInfo.systemUptime,
                                settings: self.settings, audio: self.audio, demo: true, sampleRate: 48000)
                        }
                        if mode != "latent" {
                            var z = [Float](repeating: 0, count: self.engine.currentInputDim())
                            if mode == "infervar" { for i in z.indices { z[i] = Float(i % 7) } }
                            self.engine.submit(z: z, tanh: false, hue: 0)
                        }
                        if let f = self.engine.latestFrame(), f.generation != lastGen {
                            lastGen = f.generation
                            self.engine.acknowledge(f.generation)
                        }
                        try? await Task.sleep(nanoseconds: 16_000_000)
                    }
                }
            }
            Task {
                while true {
                    try? await Task.sleep(nanoseconds: 5_000_000_000)
                    NSLog("GANLEAK \(mode) rss=\(String(format: "%.1f", rssMB()))MB")
                }
            }
        }
    }

    func tick(dt: Double, now: Double) {
        guard engine != nil, engine.isLoaded, !paused else { return }
        var z = latentEngine.computeLatent(
            dt: dt, nowSec: now, settings: settings,
            audio: audio, demo: input == .demo, sampleRate: audio.lastSampleRate)
        latentEngine.buildPulseInjection(into: &pulse, settings: settings)
        if !pulse.isEmpty {
            for i in 0..<z.count { z[i] += pulse[i % pulse.count] }
        }
        engine.submit(z: z, tanh: settings.get("Tanh Output") != 0, hue: latentEngine.lastHue)
    }

    func load(_ model: HFModel) async {
        await modelManager.ensureLoaded(model, engine: engine)
        if modelManager.selected?.id == model.id {
            let dim = engine.currentInputDim()
            latentEngine.resize(dim: dim,
                                truncation: settings.get("Truncation"),
                                motionRandomness: settings.get("Motion Randomness"))
            latentEngine.brightnessDir = nil
            brightnessReady = false
            modelReady = true
            startBrightnessDiscovery(for: model, dim: dim)
        }
    }

    // MARK: - Brightness direction

    private struct BrightnessCache: Codable {
        let dim: Int
        let dir: [Float]
    }

    /// A cached direction is only valid for the latent size it was sampled with.
    private func loadBrightnessCache(for model: HFModel, dim: Int) -> [Float]? {
        let url = modelManager.brightnessURL(for: model)
        guard let data = try? Data(contentsOf: url),
              let cache = try? JSONDecoder().decode(BrightnessCache.self, from: data),
              cache.dim == dim, cache.dir.count == dim else { return nil }
        return cache.dir
    }

    private func saveBrightnessCache(_ dir: [Float], dim: Int, for model: HFModel) {
        let url = modelManager.brightnessURL(for: model)
        let cache = BrightnessCache(dim: dim, dir: dir)
        guard let data = try? JSONEncoder().encode(cache) else { return }
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
    }

    private func startBrightnessDiscovery(for model: HFModel, dim: Int) {
        if let dir = loadBrightnessCache(for: model, dim: dim) {
            latentEngine.brightnessDir = dir
            brightnessReady = true
            modelManager.setStatus(phase: "idle", text: "brightness direction cached", progress: nil)
            return
        }

        let samples = 256
        modelManager.setStatus(phase: "sample",
                               text: "Sampling brightness direction…",
                               progress: 0)
        engine.discoverBrightness(
            samples: samples,
            progress: { [weak self] done, total in
                Task { @MainActor in
                    guard let self, self.modelManager.selected?.id == model.id else { return }
                    self.modelManager.setStatus(
                        phase: "sample",
                        text: "Brightness \(done)/\(total)…",
                        progress: total > 0 ? Double(done) / Double(total) : nil)
                }
            },
            completion: { [weak self] dir in
                Task { @MainActor in
                    guard let self, self.modelManager.selected?.id == model.id else { return }
                    guard let dir, dir.count == dim else {
                        self.modelManager.setStatus(phase: "idle", text: "model ready", progress: nil)
                        return
                    }
                    self.latentEngine.brightnessDir = dir
                    self.brightnessReady = true
                    self.saveBrightnessCache(dir, dim: dim, for: model)
                    self.modelManager.setStatus(
                        phase: "idle",
                        text: "brightness ready (\(samples) samples)",
                        progress: nil)
                }
            })
    }

    func setInput(_ new: AudioInput) {
        guard new != input else { return }
        input = new
        inputError = nil
        Task { [weak self] in
            guard let self else { return }
            self.audio.stop()
            switch new {
            case .off, .demo:
                break
            case .microphone:
                do { try await self.audio.startMicrophone() }
                catch {
                    self.inputError = error.localizedDescription
                    self.input = .off
                }
            case .system:
                do { try await self.audio.startSystemAudio() }
                catch {
                    self.inputError = error.localizedDescription
                    self.input = .off
                }
            }
        }
    }

    func toggleFullscreen() {
        NSApplication.shared.keyWindow?.toggleFullScreen(nil)
    }

    func togglePause() {
        paused.toggle()
    }

    func cycleMirror() {
        mirrorMode = (mirrorMode + 1) % 3
        renderer.mirrorMode = Int32(mirrorMode)
    }
}
