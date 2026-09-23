import Foundation

/// Per-frame audio -> latent pipeline, a direct port of the web app's
/// `main.js computeLatent` (itself a port of the Python `main.py update_frame`
/// StyleGAN branch), plus the LSD latent modulation and pulse injection.
final class LatentEngine {

    static let bins = AudioPipeline.bins          // 1025
    static let pulseDim = 512
    static let refMag: Float = 128

    private(set) var dim = 512

    private var lookup: [Int]
    private var a: [Float]
    private var smoothed = [Float](repeating: 0, count: bins)
    private var prevSpectrum = [Float](repeating: 0, count: bins)
    private var spectrum = [Float](repeating: 0, count: bins)
    private var audioNoise: [Float]
    private var pulseSpectrum = [Float](repeating: 0, count: pulseDim)
    private var pulseScratch = [Float](repeating: 0, count: pulseDim)
    private var aWeightGain: [Float]?
    private var aWeightSR: Double = 0
    private var audioRandTimer: Double = 0
    private var lastHueFlux: Float = 0
    private(set) var lastPulseMag: Float = 0
    private(set) var lastHue: Int = 0

    private(set) var lsd: LSDLatent?
    var brightnessDir: [Float]?

    init(dim: Int) {
        self.dim = dim
        self.lookup = (0..<dim).map { $0 % Self.bins }
        self.a = LSDLatent.randn(dim)
        self.audioNoise = [Float](repeating: 0, count: dim)
        self.lsd = LSDLatent(dim: dim, truncation: 1.0, motionRandomness: 0.5)
    }

    func resize(dim: Int, truncation: Double, motionRandomness: Double) {
        self.dim = dim
        lookup = (0..<dim).map { $0 % Self.bins }
        a = LSDLatent.randn(dim)
        for i in smoothed.indices { smoothed[i] = 0 }
        for i in prevSpectrum.indices { prevSpectrum[i] = 0 }
        for i in pulseSpectrum.indices { pulseSpectrum[i] = 0 }
        audioNoise = [Float](repeating: 0, count: dim)
        lsd = LSDLatent(dim: dim, truncation: truncation, motionRandomness: motionRandomness)
    }

    /// A-weighting curve in dB (0 dB at 1 kHz).
    static func aWeightDb(_ f: Float) -> Float {
        let f2 = f * f
        let num = pow(12194, 2) * f2 * f2
        let den = (f2 + pow(20.6, 2)) * sqrt((f2 + pow(107.7, 2)) * (f2 + pow(737.9, 2))) * (f2 + pow(12194, 2))
        return 20 * log10(num / den) + 2.0
    }

    private func genDemoSpectrum(_ t: Double, _ out: inout [Float]) {
        let beat1 = Float(0.5 + 0.5 * sin(t * 2.1))
        let beat2 = Float(0.5 + 0.5 * sin(t * 3.7))
        for i in out.indices { out[i] = 0 }
        let peaks: [(Float, Float, Float)] = [
            (32, 0.9 * beat1, 12), (80, 0.7 * beat1, 16),
            (180, 0.6 * beat2, 24), (360, 0.5 * beat2, 32),
            (600, 0.35, 40), (880, 0.3 * Float(0.5 + 0.5 * sin(t * 5)), 48),
        ]
        for p in peaks {
            for i in out.indices {
                let d = (Float(i) - p.0) / p.2
                out[i] += p.1 * exp(-d * d * 0.5)
            }
        }
        for i in out.indices { out[i] += 0.002 + Float.random(in: 0..<1) * 0.01 }
    }

    /// Compute the latent for one frame. Returns the LSD noise array (reused).
    func computeLatent(dt: Double, nowSec: Double, settings: SettingsStore,
                       audio: AudioPipeline, demo: Bool, sampleRate: Double) -> [Float] {
        // Raw spectrum: mic, demo, or silence.
        if audio.active {
            audio.copySpectrum(into: &spectrum)
        } else if demo {
            genDemoSpectrum(nowSec, &spectrum)
        } else {
            for i in spectrum.indices { spectrum[i] = 0 }
        }

        // Preamp (broadband dB gain applied to the raw magnitudes).
        let preampDb = settings.get("Preamp Gain")
        if preampDb != 0 {
            let g = Float(pow(10, preampDb / 20))
            for i in spectrum.indices { spectrum[i] *= g }
        }

        let sr = Float(audio.active ? audio.lastSampleRate : (sampleRate > 0 ? sampleRate : 48000))
        let srD = Double(sr)
        let fftSize = Float(AudioPipeline.fftSize)

        if aWeightGain == nil || aWeightSR != srD {
            if aWeightGain == nil { aWeightGain = [Float](repeating: 0, count: Self.bins) }
            for i in 0..<Self.bins {
                aWeightGain![i] = Float(pow(10, Double(LatentEngine.aWeightDb(Float(i) * sr / fftSize)) / 20))
            }
            aWeightSR = srD
        }
        let aw = aWeightGain!

        // Smoothing (EMA over the A-weighted spectrum).
        let smoothingFactor = settings.get("Smoothing Factor")
        let smoothing = Float(1 - exp((-dt * 10) / Swift.max(smoothingFactor, 1e-6)))
        for i in 0..<Self.bins {
            smoothed[i] = smoothing * smoothed[i] + (1 - smoothing) * spectrum[i] * aw[i]
        }

        // Randomize latent vector (swap two lookup entries periodically).
        let ar = settings.get("Audio Randomization")
        if ar != 0 {
            audioRandTimer += dt
            if audioRandTimer >= 0.5 / ar {
                audioRandTimer = 0
                let c = Int.random(in: 0..<dim)
                let d = Int.random(in: 0..<dim)
                lookup.swapAt(c, d)
            }
        }

        // Band filters.
        let bfFreq = Float(Swift.max(settings.get("Brightness Freq"), 1))
        let bfWidth = Float(Swift.max(settings.get("Brightness Width"), 0.05))
        let bfNorm = 1 / (2 * bfWidth * bfWidth)
        let mfFreq = Float(Swift.max(settings.get("Motion Freq"), 1))
        let mfWidth = Float(Swift.max(settings.get("Motion Width"), 0.05))
        let mfNorm = 1 / (2 * mfWidth * mfWidth)
        let pfFreq = Float(Swift.max(settings.get("Pulse Freq"), 1))
        let pfWidth = Float(Swift.max(settings.get("Pulse Width"), 0.05))
        let pfNorm = 1 / (2 * pfWidth * pfWidth)

        var lowPassBright: Float = 0
        for i in 1..<Self.bins {
            let d = log2((Float(i) * sr) / (fftSize * bfFreq))
            let v = spectrum[i] * aw[i] * exp(-d * d * bfNorm)
            if v > lowPassBright { lowPassBright = v }
        }

        // Blue pulse filter -> z-space injection (EMA by Pulse Smooth).
        let pulsePs = Float(pow(settings.get("Pulse Smooth"), 60 * dt))
        for i in 0..<Swift.min(Self.pulseDim, Self.bins) {
            let f = (Float(i) * sr) / fftSize
            let d = log2(f / pfFreq)
            let w = exp(-d * d * pfNorm)
            let target = spectrum[i] * aw[i] * w
            pulseSpectrum[i] = pulseSpectrum[i] * pulsePs + target * (1 - pulsePs)
        }

        let invDt = Float(1 / Swift.max(dt, 1e-6))
        var lowPassDrift: Float = 0
        var motionAmp: Float = 0
        for i in 0..<Self.bins {
            let f = (spectrum[i] - prevSpectrum[i]) * invDt
            if f > lowPassDrift { lowPassDrift = f }
            let dm = log2((Float(i) * sr) / (fftSize * mfFreq))
            let vm = f * exp(-dm * dm * mfNorm)
            if vm > motionAmp { motionAmp = vm }
        }
        prevSpectrum = spectrum
        lowPassDrift = Swift.max(lowPassDrift, 0)
        motionAmp = Swift.max(motionAmp, 0)

        let lowPass = Float(pow(Double(motionAmp), settings.get("Motion Power"))) * 0.0001
        lastHueFlux = lowPassDrift * 0.001

        // Latent composition (StyleGAN branch).
        let cutoff = Int(settings.get("Cutoff").rounded())
        let audioWeight = Float(settings.get("Audio Weight"))
        let noiseWeight = Float(settings.get("Noise Weight"))
        for i in 0..<dim {
            let v: Float = i < cutoff ? 0 : smoothed[lookup[i]] * audioWeight
            audioNoise[i] = v
        }

        var params = LSDLatent.Params()
        params.pulseAmp = 0
        params.motionAmp = lowPass
        params.music = audioNoise
        params.pulseMode = settings.get("Pulse Mode")
        params.pulseReact = settings.get("Pulse React")
        params.pulsePower = settings.get("Pulse Power")
        params.brightnessReact = settings.get("Brightness React")
        params.brightnessDir = brightnessDir
        params.brightnessAmp = Swift.max(lowPassBright, 0)
        params.motionReact = settings.get("Motion React")
        params.motionRandomness = settings.get("Motion Randomness")
        params.truncation = settings.get("Truncation")
        params.fps = 1 / Swift.max(dt, 1e-6)
        params.pulseSmooth = settings.get("Pulse Smooth")
        params.brightnessSmooth = settings.get("Brightness Smooth")
        params.motionSmooth = settings.get("Motion Smooth")

        let lz = lsd!.step(params)
        lastHue = Int((lastHueFlux * Float(settings.get("Hue Shift"))).rounded())
        return lz
    }

    /// Pulse-filtered spectrum scaled by Pulse React; added into latent z.
    func buildPulseInjection(into out: inout [Float], settings: SettingsStore) {
        lastPulseMag = 0
        let n = Self.pulseDim
        let gain = Float(settings.get("Pulse React")) / Self.refMag
        for i in 0..<n {
            let v = pulseSpectrum[i] * gain
            out[i] = v
            if abs(v) > lastPulseMag { lastPulseMag = abs(v) }
        }
    }
}
