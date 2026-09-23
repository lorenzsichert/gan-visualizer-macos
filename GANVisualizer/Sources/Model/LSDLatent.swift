import Foundation

/// Port of the Python `lsd.py` / JS `lsd.js` LSDLatent class
/// (lucid-sonic-dreams style latent-space modulation).
///
///  - base latent is a truncated normal in [-2, 2] scaled by `truncation`
///  - pulse pushes the latent along the audio/brightness direction (smoothed EMA)
///  - motion is a per-dimension signed random walk (smoothed EMA, accumulated)
///  - motion signs bounce at +/- 2*truncation
///  - randomness factors re-roll every 4 seconds (time-based)
///  - all EMAs and the motion step are normalized to a 60 fps reference
final class LSDLatent {

    struct Params {
        var pulseAmp: Float = 0
        var motionAmp: Float = 0
        var music: [Float]? = nil
        var pulseMode: Double = 0
        var pulseReact: Double = 0.5
        var pulsePower: Double = 1.0
        var brightnessReact: Double = 0.5
        var brightnessDir: [Float]? = nil
        var brightnessAmp: Float? = nil
        var motionReact: Double = 0.5
        var motionRandomness: Double = 0.5
        var truncation: Double = 1.0
        var fps: Double = 60.0
        var pulseSmooth: Double = 0.75
        var brightnessSmooth: Double = 0.75
        var motionSmooth: Double = 0.75
    }

    let dim: Int
    private var baseUnit: [Float]
    private var noise: [Float]
    private var pulseNoise: [Float]
    private var audioNoise: [Float]
    private var brightnessNoise: [Float]
    private var motionNoise: [Float]
    private var motionCum: [Float]
    private var motionSigns: [Float]
    private var randFactors: [Float]
    private var randPush: [Float]
    private var lastMotionRandomness: Double
    private var frame = 0
    private var time: Double = 0

    init(dim: Int, truncation: Double = 1.0, motionRandomness: Double = 0.5) {
        self.dim = dim
        baseUnit = []
        noise = []
        pulseNoise = []
        audioNoise = []
        brightnessNoise = []
        motionNoise = []
        motionCum = []
        motionSigns = []
        randFactors = []
        randPush = []
        lastMotionRandomness = motionRandomness
        reset(truncation: truncation, motionRandomness: motionRandomness)
    }

    func reset(truncation: Double = 1.0, motionRandomness: Double = 0.5) {
        let d = dim
        baseUnit = (0..<d).map { _ in Self.truncatedNormal(-2, 2) }
        noise = baseUnit
        pulseNoise = [Float](repeating: 0, count: d)
        audioNoise = [Float](repeating: 0, count: d)
        brightnessNoise = [Float](repeating: 0, count: d)
        motionNoise = [Float](repeating: 0, count: d)
        motionCum = [Float](repeating: 0, count: d)
        motionSigns = (0..<d).map { _ in Float.random(in: 0..<1) < 0.5 ? 1 : -1 }
        randFactors = (0..<d).map { _ in Float.random(in: 0..<1) < 0.5 ? 1.0 : Float(1.0 - motionRandomness) }
        randPush = (0..<d).map { _ in Float.random(in: 0..<1) }
        lastMotionRandomness = motionRandomness
        frame = 0
        time = 0
    }

    func step(_ p: Params) -> [Float] {
        let d = dim
        let first = frame == 0
        frame += 1
        let dt = 1.0 / Swift.max(p.fps, 1e-6)
        time += dt

        if p.motionRandomness != lastMotionRandomness {
            lastMotionRandomness = p.motionRandomness
            for i in 0..<d {
                randFactors[i] = Float.random(in: 0..<1) < 0.5 ? 1.0 : Float(1.0 - p.motionRandomness)
            }
        }

        if !first && time >= 4.0 {
            time = 0
            for i in 0..<d {
                randFactors[i] = Float.random(in: 0..<1) < 0.5 ? 1.0 : Float(1.0 - p.motionRandomness)
            }
        }

        let m = Float(p.motionReact * 20.0 / p.fps)
        var pAmp = Float(p.pulseAmp)
        if p.pulsePower != 1.0 { pAmp = Float(pow(Double(pAmp), p.pulsePower) * 0.1) }

        var pulseAdd = [Float](repeating: 0, count: d)
        for i in 0..<d { pulseAdd[i] = randPush[i] * Float(p.pulseReact) * pAmp }

        var audioAdd = [Float](repeating: 0, count: d)
        if let music = p.music {
            audioAdd = normalizeAs(music, d).map { $0 * Float(p.pulseReact) * pAmp }
        }

        var brightAdd = [Float](repeating: 0, count: d)
        if let dir = p.brightnessDir {
            let bAmp = p.brightnessAmp ?? pAmp
            brightAdd = normalizeAs(dir, d).map { $0 * Float(p.brightnessReact) * bAmp }
        }

        var motionAdd = [Float](repeating: 0, count: d)
        for i in 0..<d { motionAdd[i] = m * p.motionAmp * motionSigns[i] * randFactors[i] }

        if !first {
            let ps = Float(pow(p.pulseSmooth, 60.0 / p.fps))
            let bs = Float(pow(p.brightnessSmooth, 60.0 / p.fps))
            let ms = Float(pow(p.motionSmooth, 60.0 / p.fps))
            for i in 0..<d {
                pulseAdd[i] = pulseNoise[i] * ps + pulseAdd[i] * (1 - ps)
                audioAdd[i] = audioNoise[i] * ps + audioAdd[i] * (1 - ps)
                brightAdd[i] = brightnessNoise[i] * bs + brightAdd[i] * (1 - bs)
                motionAdd[i] = motionNoise[i] * ms + motionAdd[i] * (1 - ms)
            }
        }

        pulseNoise = pulseAdd
        audioNoise = audioAdd
        brightnessNoise = brightAdd
        motionNoise = motionAdd
        for i in 0..<d { motionCum[i] += motionAdd[i] }

        let mode = Int(p.pulseMode.rounded())
        var push: [Float]
        switch mode {
        case 1: push = audioAdd
        case 2: push = brightAdd
        case 3: push = zip(pulseAdd, brightAdd).map(+)
        case 4: push = zip(audioAdd, brightAdd).map(+)
        default: push = pulseAdd
        }

        let truncation = Float(p.truncation)
        for i in 0..<d {
            noise[i] = truncation * baseUnit[i] + push[i] + motionCum[i]
        }

        let lo = -2.0 * truncation
        let hi = 2.0 * truncation
        for i in 0..<d {
            if noise[i] - m < lo { motionSigns[i] = 1.0 }
            else if noise[i] + m >= hi { motionSigns[i] = -1.0 }
        }

        return noise
    }

    /// Unit-normalize and resize to `dim`.
    private func normalizeAs(_ x: [Float], _ dim: Int) -> [Float] {
        var v = [Float](repeating: 0, count: dim)
        let n = Swift.min(x.count, dim)
        for i in 0..<n { v[i] = x[i] }
        var norm: Float = 0
        for i in 0..<dim { norm += v[i] * v[i] }
        norm = sqrt(norm)
        if norm > 1e-12 { for i in 0..<dim { v[i] /= norm } }
        return v
    }

    /// Box-Muller standard normal with rejection to the [lo, hi] range.
    static func truncatedNormal(_ lo: Float, _ hi: Float) -> Float {
        for _ in 0..<1000 {
            let u = Float.random(in: -1..<1)
            let v = Float.random(in: -1..<1)
            let s = u * u + v * v
            if s >= 1 || s == 0 { continue }
            let z = u * sqrt((-2 * log(s)) / s)
            if z >= lo && z <= hi { return z }
        }
        return 0
    }

    static func randn(_ n: Int) -> [Float] {
        (0..<n).map { _ in truncatedNormal(-8, 8) }
    }
}
