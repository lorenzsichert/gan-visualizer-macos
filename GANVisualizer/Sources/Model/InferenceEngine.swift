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

    // Shared state (all guarded by `lock`).
    private var model: MLModel?
    private var inputDim = 512
    private var outC = 3, outH = 512, outW = 512
    private var inputName = "var"
    private var outputName = "img"
    private var epoch = 0
    private(set) var computeUnits: MLComputeUnits = .all

    // Mailbox.
    private var pendingZ: [Float]?
    private var pendingTanh = false
    private var pendingHue = 0
    private var isRunning = false

    // Published frame.
    private var textures: [MTLTexture] = []
    private var textureIndex = 0
    private var publishedTexture: MTLTexture?
    private var publishedGeneration = 0

    private var planesBuffer: MTLBuffer?
    private var planesCapacity = 0
    private var zArray: MLMultiArray?
    private var lastInferMs: Double = 0

    var onError: ((String) -> Void)?

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

    // MARK: - Queries

    var isLoaded: Bool {
        lock.lock(); defer { lock.unlock() }
        return model != nil
    }

    func currentImageSize() -> (width: Int, height: Int) {
        lock.lock(); defer { lock.unlock() }
        return (outW, outH)
    }

    func currentInferMs() -> Double {
        lock.lock(); defer { lock.unlock() }
        return lastInferMs
    }

    func currentInputDim() -> Int {
        lock.lock(); defer { lock.unlock() }
        return inputDim
    }

    // MARK: - Loading

    /// Replace the active model. Safe to call while predictions are in flight:
    /// the new state is committed atomically, a partially-finished prediction
    /// from the previous model is discarded, and any pump waiting on a frame
    /// that will never be displayed is released.
    func load(compiledModelURL: URL, computeUnits: MLComputeUnits) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let m = try MLModel(contentsOf: compiledModelURL, configuration: config)

        let desc = m.modelDescription
        var iName = "var", oName = "img"
        if desc.inputDescriptionsByName["var"] != nil { iName = "var" }
        else if let first = desc.inputDescriptionsByName.keys.first { iName = first }
        if desc.outputDescriptionsByName["img"] != nil { oName = "img" }
        else if let first = desc.outputDescriptionsByName.keys.first { oName = first }

        var dim = 512
        if let c = desc.inputDescriptionsByName[iName]?.multiArrayConstraint, c.shape.count > 1 {
            dim = c.shape[1].intValue
        }
        var C = 3, H = 512, W = 512
        if let c = desc.outputDescriptionsByName[oName]?.multiArrayConstraint, c.shape.count == 4 {
            C = Swift.max(1, c.shape[1].intValue)
            H = c.shape[2].intValue
            W = c.shape[3].intValue
        }
        guard C > 0, H > 0, W > 0 else {
            throw NSError(domain: "GANVisualizer", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "model has an invalid output shape"])
        }
        let za = try? MLMultiArray(shape: [1, NSNumber(value: dim)], dataType: .float32)

        var newTextures: [MTLTexture] = []
        for _ in 0..<2 {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                             width: W, height: H, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            if let t = device.makeTexture(descriptor: d) { newTextures.append(t) }
        }

        let bytes = C * H * W * 4
        let buf: MTLBuffer?
        let newCapacity: Int
        if planesCapacity >= bytes, let existing = planesBuffer {
            buf = existing
            newCapacity = planesCapacity
        } else {
            buf = device.makeBuffer(length: bytes, options: .storageModeShared)
            newCapacity = bytes
        }

        // Commit everything atomically so the pump sees a consistent model.
        lock.lock()
        epoch += 1
        model = m
        inputName = iName
        outputName = oName
        inputDim = dim
        outC = C; outH = H; outW = W
        zArray = za
        textures = newTextures
        textureIndex = 0
        planesBuffer = buf
        planesCapacity = newCapacity
        publishedTexture = nil
        publishedGeneration += 1
        self.computeUnits = computeUnits
        lock.unlock()

        // Release a pump blocked on the (now discarded) previous frame, then
        // restart it if a latent is already waiting.
        drainAcks()
        ackSem.signal()
        kick()
    }

    // MARK: - Submission

    func submit(z: [Float], tanh: Bool, hue: Int) {
        lock.lock()
        pendingZ = z
        pendingTanh = tanh
        pendingHue = hue
        lock.unlock()
        kick()
    }

    /// Called by the renderer once it has consumed a published frame.
    func acknowledge() { ackSem.signal() }

    func latestFrame() -> (texture: MTLTexture, generation: Int)? {
        lock.lock(); defer { lock.unlock() }
        guard let t = publishedTexture else { return nil }
        return (t, publishedGeneration)
    }

    // MARK: - Pump

    private func kick() {
        lock.lock()
        if pendingZ != nil && !isRunning {
            isRunning = true
            lock.unlock()
            queue.async { [weak self] in self?.pump() }
        } else {
            lock.unlock()
        }
    }

    private func drainAcks() {
        while ackSem.wait(timeout: .now()) == .success {}
    }

    private func pump() {
        while true {
            lock.lock()
            guard let z = pendingZ else {
                isRunning = false
                lock.unlock()
                return
            }
            pendingZ = nil
            let tanh = pendingTanh
            let hue = pendingHue
            lock.unlock()

            let published = runFrame(z: z, tanh: tanh, hue: hue)
            // Only wait for the renderer when we actually produced a frame; a
            // discarded (stale-epoch) frame must not block the pump.
            if published { ackSem.wait() }
        }
    }

    /// Returns true if a frame was published (and therefore must be acked).
    private func runFrame(z: [Float], tanh: Bool, hue: Int) -> Bool {
        lock.lock()
        guard let model = self.model, let zArray = self.zArray,
              let buffer = planesBuffer, !textures.isEmpty else {
            lock.unlock()
            return false
        }
        let snapEpoch = epoch
        let dim = inputDim
        let iName = inputName
        let oName = outputName
        let C = outC, H = outH, W = outW
        let texture = textures[textureIndex % textures.count]
        textureIndex += 1
        lock.unlock()

        let t0 = CFAbsoluteTimeGetCurrent()
        do {
            let n = Swift.min(dim, z.count)
            let zp = zArray.dataPointer.bindMemory(to: Float.self, capacity: dim)
            for i in 0..<dim { zp[i] = i < n ? z[i] : 0 }
            let provider = try MLDictionaryFeatureProvider(
                dictionary: [iName: MLFeatureValue(multiArray: zArray)])
            let out = try model.prediction(from: provider)
            guard let arr = out.featureValue(for: oName)?.multiArrayValue else { return false }
            encode(arr, into: buffer, texture: texture, C: C, H: H, W: W, tanh: tanh, hue: hue)
            let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            lock.lock(); lastInferMs = ms; lock.unlock()
        } catch {
            onError?("inference failed: \(error.localizedDescription)")
            return false
        }

        lock.lock()
        let changed = (epoch == snapEpoch)
        if changed {
            publishedTexture = texture
            publishedGeneration += 1
        }
        lock.unlock()
        return changed
    }

    private func encode(_ arr: MLMultiArray, into buffer: MTLBuffer, texture: MTLTexture,
                        C: Int, H: Int, W: Int, tanh: Bool, hue: Int) {
        let elements = C * H * W
        let isHalf = arr.dataType == .float16
        let bytes = elements * (isHalf ? 2 : 4)
        guard bytes <= buffer.length else { return }
        memcpy(buffer.contents(), arr.dataPointer, bytes)

        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        var u = Uniforms(H: UInt32(H), W: UInt32(W), hue: Int32(hue), doTanh: tanh ? 1 : 0)
        enc.setComputePipelineState(isHalf ? pipeline16 : pipeline32)
        enc.setBuffer(buffer, offset: 0, index: 0)
        enc.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setTexture(texture, index: 0)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let groups = MTLSize(width: (W + 15) / 16, height: (H + 15) / 16, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}
