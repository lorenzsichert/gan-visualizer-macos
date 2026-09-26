import Foundation
import SwiftUI
import Metal
import AppKit

@MainActor
final class AppState: ObservableObject {

    let settings = SettingsStore()
    let audio = AudioPipeline()
    let latentEngine = LatentEngine(dim: 512)
    let modelManager = ModelManager()

    var engine: InferenceEngine!
    var renderer: Renderer!

    @Published var fps: Double = 0
    @Published var inferMs: Double = 0
    @Published var input: AudioInput = .off
    @Published var inputError: String?
    @Published var mirrorMode: Int = 2
    @Published var panelCollapsed = false
    @Published var modelReady = false

    private var pulse = [Float](repeating: 0, count: LatentEngine.pulseDim)

    init() {
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
        guard engine != nil, engine.isLoaded else { return }
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
            latentEngine.resize(dim: engine.currentInputDim(),
                                truncation: settings.get("Truncation"),
                                motionRandomness: settings.get("Motion Randomness"))
            modelReady = true
        }
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

    func cycleMirror() {
        mirrorMode = (mirrorMode + 1) % 3
        renderer.mirrorMode = Int32(mirrorMode)
    }
}
