import Foundation
import AVFoundation
import Accelerate
import CoreAudio
import CoreMedia
import ScreenCaptureKit
import CoreGraphics

enum AudioInput: String, CaseIterable, Identifiable {
    case off, microphone, system, demo
    var id: String { rawValue }
    var label: String {
        switch self {
        case .off: return "Off"
        case .microphone: return "Mic"
        case .system: return "System"
        case .demo: return "Demo"
        }
    }
}

enum AudioError: Error, LocalizedError {
    case permissionDenied
    case noInput
    case screenPermission
    case captureFailed(String)

    var errorDescription: String? {
        switch self {
        case .permissionDenied: return "Microphone access was denied."
        case .noInput: return "No audio input device is available."
        case .screenPermission:
            return "Screen Recording permission is required for system audio. Enable it in System Settings › Privacy & Security › Screen Recording, then relaunch."
        case .captureFailed(let m): return m
        }
    }
}

/// The audio->spectrum DSP shared by every source: a sliding 2048-sample window,
/// DC-removed, Hann-windowed, real-FFT'd into 1025 magnitude bins (×1024/2048),
/// mirroring the web app's AudioWorklet. Runs identically for mic and system
/// audio, so both paths cost the same.
final class SpectrumAnalyzer {

    static let fftSize = 2048
    static let bins = fftSize / 2 + 1
    private static let magScale: Float = 0.25 // vDSP zrip is 2× the DFT; web uses 0.5·|X|

    private let lock = NSLock()
    private var spectrumBuf = [Float](repeating: 0, count: bins)

    private let n = fftSize
    private let log2n: vDSP_Length = 11
    private var fftSetup: FFTSetup?
    private var window = [Float](repeating: 0, count: fftSize)
    private var ring = [Float](repeating: 0, count: fftSize)
    private var pos = 0
    private var filled = 0
    private var windowed = [Float](repeating: 0, count: fftSize)
    private var realp = [Float](repeating: 0, count: fftSize / 2)
    private var imagp = [Float](repeating: 0, count: fftSize / 2)
    private var mags = [Float](repeating: 0, count: fftSize / 2)

    init() {
        fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))
        for i in 0..<n {
            window[i] = 0.5 * (1 - cos(2 * Float.pi * Float(i) / Float(n - 1)))
        }
    }

    deinit {
        if let setup = fftSetup { vDSP_destroy_fftsetup(setup) }
    }

    func reset() {
        pos = 0
        filled = 0
        for i in ring.indices { ring[i] = 0 }
        lock.lock()
        for i in spectrumBuf.indices { spectrumBuf[i] = 0 }
        lock.unlock()
    }

    /// Feed mono samples from any source (called on that source's thread).
    func process(_ samples: UnsafePointer<Float>, _ count: Int) {
        guard count > 0 else { return }
        let nn = n
        for i in 0..<count {
            ring[pos] = samples[i]
            pos = (pos + 1) % nn
            if filled < nn { filled += 1 }
        }
        if filled < nn { return }
        compute()
    }

    func copySpectrum(into dst: inout [Float]) {
        lock.lock()
        let count = Swift.min(dst.count, spectrumBuf.count)
        for i in 0..<count { dst[i] = spectrumBuf[i] }
        lock.unlock()
    }

    private func compute() {
        guard let setup = fftSetup else { return }
        let nn = n
        var mean: Float = 0
        for k in 0..<nn { mean += ring[(pos + k) % nn] }
        mean /= Float(nn)
        for k in 0..<nn {
            let idx = (pos + k) % nn
            windowed[k] = (ring[idx] - mean) * window[k]
        }

        realp.withUnsafeMutableBufferPointer { rp in
            imagp.withUnsafeMutableBufferPointer { ip in
                var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                windowed.withUnsafeBufferPointer { wp in
                    wp.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: nn / 2) { cp in
                        vDSP_ctoz(cp, 2, &split, 1, vDSP_Length(nn / 2))
                    }
                }
                vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                vDSP_zvabs(&split, 1, &mags, 1, vDSP_Length(nn / 2))

                let s = SpectrumAnalyzer.magScale
                lock.lock()
                spectrumBuf[0] = abs(rp[0]) * s
                spectrumBuf[nn / 2] = abs(ip[0]) * s
                for i in 1..<(nn / 2) { spectrumBuf[i] = mags[i] * s }
                lock.unlock()
            }
        }
    }
}

/// Microphone capture + system-audio capture, both funnelling into one
/// `SpectrumAnalyzer` so the DSP (and its cost) is identical.
final class AudioPipeline {

    static let fftSize = SpectrumAnalyzer.fftSize
    static let bins = SpectrumAnalyzer.bins

    let analyzer = SpectrumAnalyzer()

    private(set) var running = false
    private(set) var lastSampleRate: Double = 48000
    private(set) var msgCount = 0

    private var engine: AVAudioEngine?
    private var sinkNode: AVAudioSinkNode?
    private var tapHelper: CoreAudioTapCapture?
    private var stream: SCStream?
    private var systemHelper: SystemAudioCapture?
    // Audio must preempt everything else, so its delivery queue runs at
    // `.userInteractive`; the (unused, 1 fps, 2×2) video output gets its own
    // lower-priority queue so video can never back-pressure the audio path.
    private let systemQueue = DispatchQueue(
        label: "com.jakobsichert.GANVisualizer.systemaudio", qos: .userInteractive)
    private let screenQueue = DispatchQueue(
        label: "com.jakobsichert.GANVisualizer.systemvideo", qos: .utility)

    var onError: ((String) -> Void)?

    var active: Bool { running }

    func copySpectrum(into dst: inout [Float]) { analyzer.copySpectrum(into: &dst) }

    // MARK: - Microphone

    func startMicrophone() async throws {
        stop()
        let granted = await Self.requestMicPermission()
        guard granted else { throw AudioError.permissionDenied }

        let engine = AVAudioEngine()
        let input = engine.inputNode
        // An `AVAudioSinkNode` performs no format conversion, so it must be
        // connected with the input node's output-scope (hardware) format.
        let format = input.outputFormat(forBus: 0)
        guard format.channelCount > 0, format.sampleRate > 0 else { throw AudioError.noInput }

        lastSampleRate = format.sampleRate
        analyzer.reset()

        // `inputNode.installTap` batches delivery into large buffers on macOS
        // (commonly ~100 ms, and it ignores the requested buffer size), which
        // would put a hard floor under the visualizer's reaction time. A sink
        // node instead receives the raw hardware IO buffer — a handful of
        // milliseconds — directly on the CoreAudio real-time thread, so the
        // sliding-window spectrum is refreshed with the least possible delay.
        // The block only does the preallocated analyzer work (no allocation).
        let analyzer = self.analyzer
        let sink = AVAudioSinkNode { [weak self] _, frameCount, inputData -> OSStatus in
            let abl = UnsafeMutableAudioBufferListPointer(
                UnsafeMutablePointer(mutating: inputData))
            guard let buffer = abl.first, let mData = buffer.mData else { return noErr }
            let frames = Swift.min(Int(frameCount),
                                   Int(buffer.mDataByteSize) / MemoryLayout<Float>.size)
            guard frames > 0 else { return noErr }
            analyzer.process(mData.assumingMemoryBound(to: Float.self), frames)
            self?.msgCount += 1
            return noErr
        }
        engine.attach(sink)
        engine.connect(input, to: sink, format: format)
        engine.prepare()
        try engine.start()
        self.engine = engine
        self.sinkNode = sink
        running = true
    }

    // MARK: - System audio

    /// Starts the lowest-latency system-audio source available: a Core Audio
    /// process tap on macOS 14.2+ (sub-20 ms), falling back to ScreenCaptureKit
    /// (~20–50 ms) when the tap is unavailable or cannot be created.
    func startSystemAudio() async throws {
        stop()

        if #available(macOS 14.2, *) {
            let helper = CoreAudioTapCapture(analyzer: analyzer)
            helper.onSampleRate = { [weak self] sr in self?.lastSampleRate = sr }
            helper.onBuffer = { [weak self] in self?.msgCount += 1 }
            do {
                try helper.start()
                analyzer.reset()
                tapHelper = helper
                running = true
                return
            } catch {
                NSLog("GANVisualizer: Core Audio tap unavailable (\(error.localizedDescription)); falling back to ScreenCaptureKit")
            }
        }

        try await startScreenCaptureAudio()
    }

    private func startScreenCaptureAudio() async throws {
        guard Self.requestScreenRecordingPermission() else {
            throw AudioError.screenPermission
        }

        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        } catch {
            // Preflight passed, so this is a real capture failure rather than a
            // permission denial — surface the underlying reason.
            throw AudioError.captureFailed(
                "Could not read screen content for system audio: \(error.localizedDescription)")
        }
        guard let display = content.displays.first else { throw AudioError.captureFailed("no display available for system audio") }

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.sampleRate = 48000
        config.channelCount = 1
        config.excludesCurrentProcessAudio = true
        config.showsCursor = false
        // Minimal video footprint: we only want the audio.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        // Keep only two frames buffered; the default (3) adds avoidable
        // capture latency for no benefit here.
        config.queueDepth = 2

        let helper = SystemAudioCapture(analyzer: analyzer)
        helper.onSampleRate = { [weak self] sr in self?.lastSampleRate = sr }
        helper.onBuffer = { [weak self] in self?.msgCount += 1 }
        helper.onError = { [weak self] msg in self?.onError?(msg) }
        let stream = SCStream(filter: filter, configuration: config, delegate: helper)
        try stream.addStreamOutput(helper, type: .audio, sampleHandlerQueue: systemQueue)
        // Also drain the (tiny) video output on its own queue: ScreenCaptureKit
        // otherwise reports a dropped frame for every frame, and keeping the
        // video path off the audio queue guarantees it can never delay audio.
        try stream.addStreamOutput(helper, type: .screen, sampleHandlerQueue: screenQueue)
        try await stream.startCapture()

        lastSampleRate = 48000
        analyzer.reset()
        self.systemHelper = helper
        self.stream = stream
        running = true
    }

    // MARK: - Stop

    func stop() {
        if let engine {
            engine.stop()
            if let sink = sinkNode {
                engine.disconnectNodeOutput(engine.inputNode)
                engine.detach(sink)
            }
        }
        engine = nil
        sinkNode = nil

        if #available(macOS 14.2, *) {
            tapHelper?.stop()
        }
        tapHelper = nil

        if let stream {
            let s = stream
            Task { try? await s.stopCapture() }
        }
        stream = nil
        systemHelper = nil

        running = false
        analyzer.reset()
    }

    private static func requestScreenRecordingPermission() -> Bool {
        if CGPreflightScreenCaptureAccess() { return true }
        // Shows the TCC prompt on first use and returns the decision. ScreenCap-
        // tureKit may still need an app relaunch before it delivers content.
        return CGRequestScreenCaptureAccess()
    }

    private static func requestMicPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return true
        case .notDetermined:
            return await withCheckedContinuation { cont in
                AVCaptureDevice.requestAccess(for: .audio) { ok in cont.resume(returning: ok) }
            }
        default:
            return false
        }
    }
}

/// Receives PCM from ScreenCaptureKit and hands it to the shared analyzer.
private final class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private let analyzer: SpectrumAnalyzer
    var onSampleRate: ((Double) -> Void)?
    var onBuffer: (() -> Void)?
    var onError: ((String) -> Void)?

    init(analyzer: SpectrumAnalyzer) {
        self.analyzer = analyzer
        super.init()
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        // The sample queue runs continuously, so drain per-buffer autoreleased
        // CoreMedia objects instead of letting them accumulate.
        autoreleasepool {
            guard type == .audio, sampleBuffer.isValid else { return }
            guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
                  let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc) else { return }
            // Only float PCM is expected from ScreenCaptureKit.
            guard (asbd.pointee.mFormatFlags & kAudioFormatFlagIsFloat) != 0 else { return }

            let channels = Int(asbd.pointee.mChannelsPerFrame)
            let frames = CMSampleBufferGetNumSamples(sampleBuffer)
            guard channels > 0, frames > 0 else { return }

            let abl = AudioBufferList.allocate(maximumBuffers: channels)
            defer { free(abl.unsafeMutablePointer) }
            var blockBuffer: CMBlockBuffer?
            let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
                sampleBuffer,
                bufferListSizeNeededOut: nil,
                bufferListOut: abl.unsafeMutablePointer,
                bufferListSize: AudioBufferList.sizeInBytes(maximumBuffers: channels),
                blockBufferAllocator: kCFAllocatorDefault,
                blockBufferMemoryAllocator: kCFAllocatorDefault,
                flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
                blockBufferOut: &blockBuffer)
            guard status == noErr else { return }

            let buffer = abl[0]
            guard let mData = buffer.mData else { return }
            let count = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
            guard count > 0 else { return }
            analyzer.process(mData.bindMemory(to: Float.self, capacity: count), count)
            onSampleRate?(asbd.pointee.mSampleRate)
            onBuffer?()
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        onError?("System audio capture stopped: \(error.localizedDescription)")
    }
}

/// Receives PCM from a Core Audio process tap (macOS 14.2+,
/// `AudioHardwareCreateProcessTap`). The tap is exposed to clients through a
/// private, tap-only aggregate device whose IO block runs on the Core Audio
/// real-time thread — the lowest-latency public route to system output, with
/// none of ScreenCaptureKit's sample-handler buffering.
private final class CoreAudioTapCapture {

    private let analyzer: SpectrumAnalyzer
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var ioProcID: AudioDeviceIOProcID?

    var onSampleRate: ((Double) -> Void)?
    var onBuffer: (() -> Void)?

    init(analyzer: SpectrumAnalyzer) {
        self.analyzer = analyzer
    }

    @available(macOS 14.2, *)
    func start() throws {
        let description = CATapDescription()
        description.name = "GAN Visualizer System Audio"
        description.isPrivate = true
        description.muteBehavior = .unmuted
        // Global tap of every other process, mixed to mono. The app renders no
        // audio, so there is nothing to feed back into the tap.
        description.isExclusive = true
        description.processes = []
        description.isMixdown = true
        description.isMono = true

        var tap = AudioObjectID(kAudioObjectUnknown)
        let tapStatus = AudioHardwareCreateProcessTap(description, &tap)
        guard tapStatus == noErr, tap != kAudioObjectUnknown else {
            throw AudioError.captureFailed(
                "Core Audio tap creation failed (status \(tapStatus)) — is System Audio Recording allowed?")
        }
        tapID = tap

        do {
            let format = try Self.tapFormat(tap)
            guard (format.mFormatFlags & kAudioFormatFlagIsFloat) != 0,
                  format.mChannelsPerFrame > 0, format.mSampleRate > 0 else {
                throw AudioError.captureFailed("Core Audio tap returned an unsupported format")
            }
            onSampleRate?(format.mSampleRate)

            aggregateID = try Self.createAggregateDevice(tapUID: try Self.tapUID(tap))

            // Passing no dispatch queue invokes the block directly on Core
            // Audio's real-time thread, with no extra scheduling hop.
            let analyzer = self.analyzer
            var proc: AudioDeviceIOProcID?
            let ioStatus = AudioDeviceCreateIOProcIDWithBlock(&proc, aggregateID, nil) {
                [weak self] _, inputData, _, _, _ in
                let abl = UnsafeMutableAudioBufferListPointer(
                    UnsafeMutablePointer(mutating: inputData))
                // Mono mixdown, so the first buffer holds the samples.
                guard let buffer = abl.first, let mData = buffer.mData else { return }
                let count = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
                guard count > 0 else { return }
                analyzer.process(mData.assumingMemoryBound(to: Float.self), count)
                self?.onBuffer?()
            }
            guard ioStatus == noErr, let proc else {
                throw AudioError.captureFailed("Core Audio IO proc creation failed (status \(ioStatus))")
            }
            ioProcID = proc

            let startStatus = AudioDeviceStart(aggregateID, proc)
            guard startStatus == noErr else {
                throw AudioError.captureFailed("Core Audio tap start failed (status \(startStatus))")
            }
        } catch {
            stop()
            throw error
        }
    }

    @available(macOS 14.2, *)
    func stop() {
        if aggregateID != kAudioObjectUnknown, let ioProcID {
            AudioDeviceStop(aggregateID, ioProcID)
            AudioDeviceDestroyIOProcID(aggregateID, ioProcID)
        }
        ioProcID = nil
        if aggregateID != kAudioObjectUnknown {
            AudioHardwareDestroyAggregateDevice(aggregateID)
            aggregateID = AudioObjectID(kAudioObjectUnknown)
        }
        if tapID != kAudioObjectUnknown {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
        }
    }

    @available(macOS 14.2, *)
    private static func tapFormat(_ tap: AudioObjectID) throws -> AudioStreamBasicDescription {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var asbd = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        let status = AudioObjectGetPropertyData(tap, &address, 0, nil, &size, &asbd)
        guard status == noErr else {
            throw AudioError.captureFailed("Could not read Core Audio tap format (status \(status))")
        }
        return asbd
    }

    @available(macOS 14.2, *)
    private static func tapUID(_ tap: AudioObjectID) throws -> String {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var value: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        let status = withUnsafeMutablePointer(to: &value) {
            AudioObjectGetPropertyData(tap, &address, 0, nil, &size, $0)
        }
        guard status == noErr, let value else {
            throw AudioError.captureFailed("Could not read Core Audio tap UID (status \(status))")
        }
        return value.takeRetainedValue() as String
    }

    @available(macOS 14.2, *)
    private static func createAggregateDevice(tapUID: String) throws -> AudioObjectID {
        // Tap-only aggregate (no main sub-device) so it is not pinned to the
        // default output device's clock/rate.
        let description: [String: Any] = [
            kAudioAggregateDeviceNameKey: "GAN Visualizer System Audio",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapUIDKey: tapUID,
                    kAudioSubTapDriftCompensationKey: true,
                ],
            ],
        ]
        var aggregate = AudioObjectID(kAudioObjectUnknown)
        let status = AudioHardwareCreateAggregateDevice(description as CFDictionary, &aggregate)
        guard status == noErr, aggregate != kAudioObjectUnknown else {
            throw AudioError.captureFailed("Core Audio aggregate device creation failed (status \(status))")
        }
        return aggregate
    }
}
