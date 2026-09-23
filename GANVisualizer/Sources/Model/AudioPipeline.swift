import Foundation
import AVFoundation
import Accelerate
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
    private var stream: SCStream?
    private var systemHelper: SystemAudioCapture?
    private let systemQueue = DispatchQueue(label: "com.jakobsichert.GANVisualizer.systemaudio")

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
        let format = input.outputFormat(forBus: 0)
        guard format.channelCount > 0, format.sampleRate > 0 else { throw AudioError.noInput }

        lastSampleRate = format.sampleRate
        analyzer.reset()
        input.installTap(onBus: 0, bufferSize: 128, format: format) { [weak self] buffer, _ in
            guard let self, let ch = buffer.floatChannelData, buffer.frameLength > 0 else { return }
            self.analyzer.process(ch[0], Int(buffer.frameLength))
            self.msgCount += 1
        }
        engine.prepare()
        try engine.start()
        self.engine = engine
        running = true
    }

    // MARK: - System audio

    func startSystemAudio() async throws {
        stop()
        if !CGPreflightScreenCaptureAccess() {
            _ = CGRequestScreenCaptureAccess()
        }

        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        } catch {
            throw AudioError.screenPermission
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

        let helper = SystemAudioCapture(analyzer: analyzer)
        helper.onSampleRate = { [weak self] sr in self?.lastSampleRate = sr }
        helper.onBuffer = { [weak self] in self?.msgCount += 1 }
        helper.onError = { [weak self] msg in self?.onError?(msg) }
        let stream = SCStream(filter: filter, configuration: config, delegate: helper)
        try stream.addStreamOutput(helper, type: .audio, sampleHandlerQueue: systemQueue)
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
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        engine = nil

        if let stream {
            let s = stream
            Task { try? await s.stopCapture() }
        }
        stream = nil
        systemHelper = nil

        running = false
        analyzer.reset()
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

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        onError?("System audio capture stopped: \(error.localizedDescription)")
    }
}
