import Foundation
import CoreML
import Metal
import MetalKit

/// Owns the CoreML model and runs predictions "latest-wins": the render loop
/// submits the newest latent, stale work is dropped, and exactly one prediction
/// is in flight (paced by the renderer's acknowledgement). The NCHW model output
/// is converted to an RGBA8 Metal texture by a compute shader (with the hue
/// channel-roll applied), so the main thread only draws.
final class InferenceEngine {

    struct Uniforms {
        var H: UInt32
        var W: UInt32
        var hue: Int32
        var doTanh: Int32
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline16: MTLComputePipelineState
    private let pipeline32: MTLComputePipelineState
    private let queue = DispatchQueue(label: "com.jakobsichert.GANVisualizer.inference")
    private let ackSem = DispatchSemaphore(value: 0)
    private let lock = NSLock()

    private var model: MLModel?
    private(set) var inputDim = 512
    private(set) var outC = 3, outH = 512, outW = 512
    private var inputName = "var"
    private var outputName = "img"
    private(set) var computeUnits: MLComputeUnits = .all

    // Mailbox (protected by `lock`)
    private var pendingZ: [Float]?
    private var pendingTanh = false
    private var pendingHue = 0
    private var isRunning = false

    // Published frame (protected by `lock`)
    private var textures: [MTLTexture] = []
    private var textureIndex = 0
    private var publishedTexture: MTLTexture?
    private var publishedGeneration = 0

    private var planesBuffer: MTLBuffer?
    private var planesCapacity = 0
    private var zArray: MLMultiArray?

    /// Last measured prediction + conversion time (ms).
    private(set) var lastInferMs: Double = 0
    var onInferTime: ((Double) -> Void)?

    init?(device: MTLDevice) {
        guard let cq = device.makeCommandQueue(),
              let lib = device.makeDefaultLibrary(),
              let k16 = lib.makeFunction(name: "planesToRGBA16"),
              let k32 = lib.makeFunction(name: "planesToRGBA32"),
              let p16 = try? device.makeComputePipelineState(function: k16),
              let p32 = try? device.makeComputePipelineState(function: k32) else { return nil }
        self.device = device
        self.commandQueue = cq
        self.pipeline16 = p16
        self.pipeline32 = p32
    }

    var isLoaded: Bool { model != nil }

    // MARK: - Loading

    func load(compiledModelURL: URL, computeUnits: MLComputeUnits) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let m = try MLModel(contentsOf: compiledModelURL, configuration: config)
        self.model = m
        self.computeUnits = computeUnits

        let desc = m.modelDescription
        if desc.inputDescriptionsByName["var"] != nil { inputName = "var" }
        else if let first = desc.inputDescriptionsByName.keys.first { inputName = first }
        if desc.outputDescriptionsByName["img"] != nil { outputName = "img" }
        else if let first = desc.outputDescriptionsByName.keys.first { outputName = first }

        if let c = desc.inputDescriptionsByName[inputName]?.multiArrayConstraint, c.shape.count > 1 {
            inputDim = c.shape[1].intValue
        }
        if let c = desc.outputDescriptionsByName[outputName]?.multiArrayConstraint, c.shape.count == 4 {
            outC = Swift.max(1, c.shape[1].intValue)
            outH = c.shape[2].intValue
            outW = c.shape[3].intValue
        }
        zArray = try? MLMultiArray(shape: [1, NSNumber(value: inputDim)], dataType: .float32)
        prepareOutputs()
    }

    private func prepareOutputs() {
        var newTextures: [MTLTexture] = []
        for _ in 0..<2 {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                             width: outW, height: outH, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            if let tex = device.makeTexture(descriptor: d) { newTextures.append(tex) }
        }
        let bytes = outC * outH * outW * 4
        if planesCapacity < bytes {
            planesBuffer = device.makeBuffer(length: bytes, options: .storageModeShared)
            planesCapacity = bytes
        }
        lock.lock()
        textures = newTextures
        publishedTexture = nil
        textureIndex = 0
        lock.unlock()
    }

    // MARK: - Submission

    func submit(z: [Float], tanh: Bool, hue: Int) {
        lock.lock()
        pendingZ = z
        pendingTanh = tanh
        pendingHue = hue
        let start = !isRunning
        if start { isRunning = true }
        lock.unlock()
        if start { queue.async { [weak self] in self?.pump() } }
    }

    /// Called by the renderer once it has consumed a published frame.
    func acknowledge() { ackSem.signal() }

    func latestFrame() -> (texture: MTLTexture, generation: Int)? {
        lock.lock(); defer { lock.unlock() }
        guard let t = publishedTexture else { return nil }
        return (t, publishedGeneration)
    }

    // MARK: - Pump

    private func pump() {
        while true {
            lock.lock()
            guard let z = pendingZ else { isRunning = false; lock.unlock(); return }
            pendingZ = nil
            let tanh = pendingTanh
            let hue = pendingHue
            lock.unlock()

            runFrame(z: z, tanh: tanh, hue: hue)
            ackSem.wait()
        }
    }

    private func runFrame(z: [Float], tanh: Bool, hue: Int) {
        guard let model, let zArray, let planesBuffer else { return }
        let t0 = CFAbsoluteTimeGetCurrent()
        do {
            let n = Swift.min(inputDim, z.count)
            let zp = zArray.dataPointer.bindMemory(to: Float.self, capacity: inputDim)
            for i in 0..<inputDim { zp[i] = i < n ? z[i] : 0 }
            let provider = try MLDictionaryFeatureProvider(
                dictionary: [inputName: MLFeatureValue(multiArray: zArray)])
            let out = try model.prediction(from: provider)
            guard let arr = out.featureValue(for: outputName)?.multiArrayValue else { return }
            encode(arr, into: planesBuffer, tanh: tanh, hue: hue)
            lastInferMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            onInferTime?(lastInferMs)
        } catch {
            NSLog("GANVisualizer inference error: \(error)")
        }
    }

    private func encode(_ arr: MLMultiArray, into buffer: MTLBuffer, tanh: Bool, hue: Int) {
        let elements = outC * outH * outW
        let isHalf = arr.dataType == .float16
        let byteCount = elements * (isHalf ? 2 : 4)
        if isHalf {
            memcpy(buffer.contents(), arr.dataPointer, byteCount)
        } else {
            memcpy(buffer.contents(), arr.dataPointer, byteCount)
        }

        lock.lock()
        let tex = textures.isEmpty ? nil : textures[textureIndex % Swift.max(1, textures.count)]
        textureIndex += 1
        lock.unlock()
        guard let texture = tex, let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }

        var u = Uniforms(H: UInt32(outH), W: UInt32(outW), hue: Int32(hue), doTanh: tanh ? 1 : 0)
        enc.setComputePipelineState(isHalf ? pipeline16 : pipeline32)
        enc.setBuffer(buffer, offset: 0, index: 0)
        enc.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setTexture(texture, index: 0)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let groups = MTLSize(width: (outW + 15) / 16, height: (outH + 15) / 16, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        lock.lock()
        publishedTexture = texture
        publishedGeneration += 1
        lock.unlock()
    }
}
