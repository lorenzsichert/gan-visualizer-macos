import Foundation
import Metal
import MetalKit
import simd
import SwiftUI

struct DrawUniforms {
    var viewport: SIMD2<Float>
    var imageSize: SIMD2<Float>
    var mirrorMode: Int32
    var blurAmount: Float
}

/// Metal renderer: draws the latest inferred frame as a full-window textured
/// quad, applying the horizontal/vertical reflected mirror tiling in the
/// fragment shader. Also drives the per-frame latent computation (via
/// `onFrameTick`) so audio -> inference -> display stay decoupled.
final class Renderer: NSObject, MTKViewDelegate {

    let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let drawPipeline: MTLRenderPipelineState
    private let sampler: MTLSamplerState
    private let fallbackTexture: MTLTexture
    private weak var engine: InferenceEngine?

    /// Called at the start of every draw with (dt, now) in seconds.
    var onFrameTick: ((Double, Double) -> Void)?
    /// Called ~2×/s with (fps, medianInferMs).
    var onStats: ((Double, Double) -> Void)?

    var mirrorMode: Int32 = 2

    private var lastTime: Double = CACurrentMediaTime()
    private var lastGeneration = -1
    private var currentTexture: MTLTexture?
    private var frames = 0
    private var statWindow: Double = CACurrentMediaTime()
    private var inferSum = 0.0
    private var inferCount = 0

    init?(device: MTLDevice, engine: InferenceEngine) {
        guard let cq = device.makeCommandQueue(),
              let lib = device.makeDefaultLibrary(),
              let vfn = lib.makeFunction(name: "vs_fullscreen"),
              let ffn = lib.makeFunction(name: "fs_draw") else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        guard let pipeline = try? device.makeRenderPipelineState(descriptor: desc) else { return nil }

        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear
        sd.magFilter = .linear
        sd.sAddressMode = .clampToEdge
        sd.tAddressMode = .clampToEdge
        guard let samp = device.makeSamplerState(descriptor: sd) else { return nil }

        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                          width: 2, height: 2, mipmapped: false)
        td.usage = [.shaderRead]
        guard let fallback = device.makeTexture(descriptor: td) else { return nil }

        self.device = device
        self.commandQueue = cq
        self.drawPipeline = pipeline
        self.sampler = samp
        self.fallbackTexture = fallback
        self.engine = engine
        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        let dt = min(max(now - lastTime, 1e-4), 0.1)
        lastTime = now

        onFrameTick?(dt, now)

        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cb = commandQueue.makeCommandBuffer() else { return }

        // Keep drawing the most recent frame until a newer one is produced;
        // otherwise draws between two produced frames would show the fallback.
        var texture = currentTexture ?? fallbackTexture
        let size = engine?.currentImageSize() ?? (width: 512, height: 512)
        let imgW = Swift.max(1, size.width)
        let imgH = Swift.max(1, size.height)
        var consumedGeneration: Int?
        if let frame = engine?.latestFrame(), frame.generation != lastGeneration {
            currentTexture = frame.texture
            texture = frame.texture
            lastGeneration = frame.generation
            consumedGeneration = frame.generation
        }

        if let enc = cb.makeRenderCommandEncoder(descriptor: rpd) {
            enc.setRenderPipelineState(drawPipeline)
            var u = DrawUniforms(
                viewport: SIMD2<Float>(Float(view.drawableSize.width), Float(view.drawableSize.height)),
                imageSize: SIMD2<Float>(Float(imgW), Float(imgH)),
                mirrorMode: mirrorMode,
                blurAmount: mirrorMode >= 2 ? Swift.max(1, Float(imgW) / 40) : 0)
            enc.setFragmentBytes(&u, length: MemoryLayout<DrawUniforms>.stride, index: 0)
            enc.setFragmentTexture(texture, index: 0)
            enc.setFragmentSamplerState(sampler, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }
        cb.present(drawable)
        if let generation = consumedGeneration {
            cb.addCompletedHandler { [weak self] _ in
                self?.engine?.acknowledge(generation)
            }
        }
        cb.commit()

        // Stats: ~2×/s.
        frames += 1
        let ms = engine?.currentInferMs() ?? 0
        if ms > 0 { inferSum += ms; inferCount += 1 }
        if now - statWindow >= 0.5 {
            let fps = Double(frames) / (now - statWindow)
            let infer = inferCount > 0 ? inferSum / Double(inferCount) : 0
            inferSum = 0; inferCount = 0
            onStats?(fps, infer)
            frames = 0
            statWindow = now
        }
    }
}

/// SwiftUI wrapper around an `MTKView` driven by `Renderer`.
struct VisualizerView: NSViewRepresentable {
    let renderer: Renderer

    func makeCoordinator() -> Renderer { renderer }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: renderer.device)
        view.colorPixelFormat = .bgra8Unorm
        view.preferredFramesPerSecond = 60
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.framebufferOnly = true
        view.delegate = renderer
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}
}
