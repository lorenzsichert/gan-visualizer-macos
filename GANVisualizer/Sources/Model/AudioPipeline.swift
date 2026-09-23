import Foundation
import AVFoundation
import Accelerate

enum AudioError: Error {
    case permissionDenied
    case noInput
}

/// Microphone capture + off-thread spectrum, mirroring the web app's
/// `AudioWorkletProcessor`: a sliding 2048-sample window, DC-removed,
/// Hann-windowed, real-FFT'd into 1025 magnitude bins (×1024/2048 reference).
final class AudioPipeline {

    static let fftSize = 2048
    static let bins = fftSize / 2 + 1          // 1025
    private static let magScale: Float = 0.25  // vDSP zrip is 2× the DFT; web uses 0.5·|X|

    private let engine = AVAudioEngine()
    private let lock = NSLock()
    private var spectrumBuf = [Float](repeating: 0, count: bins)

    private(set) var running = false
    private(set) var lastSampleRate: Double = 48000
    private(set) var msgCount = 0

    // FFT scratch
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

    var active: Bool { running }

    func start() async throws {
        if running { return }
        let granted = await Self.requestMicPermission()
        guard granted else { throw AudioError.permissionDenied }

        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.channelCount > 0, format.sampleRate > 0 else { throw AudioError.noInput }

        lastSampleRate = format.sampleRate
        pos = 0
        filled = 0
        for i in ring.indices { ring[i] = 0 }

        input.installTap(onBus: 0, bufferSize: 128, format: format) { [weak self] buffer, time in
            self?.handle(buffer: buffer, time: time)
        }
        engine.prepare()
        try engine.start()
        running = true
    }

    func stop() {
        if !running { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        running = false
        lock.lock()
        for i in spectrumBuf.indices { spectrumBuf[i] = 0 }
        lock.unlock()
    }

    /// Copy the latest 1025-bin spectrum into `dst`.
    func copySpectrum(into dst: inout [Float]) {
        lock.lock()
        let count = Swift.min(dst.count, spectrumBuf.count)
        for i in 0..<count { dst[i] = spectrumBuf[i] }
        lock.unlock()
    }

    // MARK: - Capture

    private func handle(buffer: AVAudioPCMBuffer, time: AVAudioTime) {
        guard let chData = buffer.floatChannelData, buffer.frameLength > 0 else { return }
        let frames = Int(buffer.frameLength)
        let ch = chData[0]
        let nn = n
        for i in 0..<frames {
            ring[pos] = ch[i]
            pos = (pos + 1) % nn
            if filled < nn { filled += 1 }
        }
        if filled < nn { return }
        compute()
        msgCount += 1
    }

    private func compute() {
        guard let setup = fftSetup else { return }
        let nn = n
        // Mean (DC) removal before windowing keeps offsets out of the low bins.
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

                let s = AudioPipeline.magScale
                lock.lock()
                spectrumBuf[0] = abs(rp[0]) * s
                spectrumBuf[nn / 2] = abs(ip[0]) * s
                for i in 1..<(nn / 2) { spectrumBuf[i] = mags[i] * s }
                lock.unlock()
            }
        }
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
