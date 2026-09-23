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
    @Published var audioOn = false
    @Published var demo = false
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
        }
        Task { [weak self] in
            guard let self else { return }
            await self.modelManager.refresh()
            if let first = self.modelManager.models.first {
                await self.load(first)
            }
        }
    }

    func tick(dt: Double, now: Double) {
        guard engine != nil, engine.isLoaded else { return }
        var z = latentEngine.computeLatent(
            dt: dt, nowSec: now, settings: settings,
            audio: audio, demo: demo, sampleRate: audio.lastSampleRate)
        latentEngine.buildPulseInjection(into: &pulse, settings: settings)
        if !pulse.isEmpty {
            for i in 0..<z.count { z[i] += pulse[i % pulse.count] }
        }
        engine.submit(z: z, tanh: settings.get("Tanh Output") != 0, hue: latentEngine.lastHue)
    }

    func load(_ model: HFModel) async {
        await modelManager.ensureLoaded(model, engine: engine)
        if modelManager.selected?.id == model.id {
            latentEngine.resize(dim: engine.inputDim,
                                truncation: settings.get("Truncation"),
                                motionRandomness: settings.get("Motion Randomness"))
            modelReady = true
        }
    }

    func toggleMic() {
        if audioOn {
            audio.stop()
            audioOn = false
        } else {
            Task {
                do {
                    try await audio.start()
                    audioOn = true
                } catch {
                    modelManager.errorMessage = "Microphone unavailable: \(error.localizedDescription)"
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
