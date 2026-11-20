import Foundation
import SwiftUI

/// A single tunable parameter, mirroring the JS `SETTINGS` entries.
struct SettingDef {
    let group: String
    var value: Double
    let min: Double
    let max: Double
    let step: Double
    let dec: Int
    let hidden: Bool

    init(_ group: String, _ value: Double, _ min: Double, _ max: Double,
         _ step: Double, _ dec: Int, hidden: Bool = false) {
        self.group = group
        self.value = value
        self.min = min
        self.max = max
        self.step = step
        self.dec = dec
        self.hidden = hidden
    }
}

/// Observable store for every parameter, persisted to `UserDefaults`.
/// A direct port of `js/settings.js` / the Python `midi.settings`.
/// Not actor-isolated: the render loop reads values off the UI thread while the
/// UI writes them on the main thread.
final class SettingsStore: ObservableObject {

    /// Insertion-ordered key list (Swift dictionaries are unordered).
    static let order: [String] = [
        "Pulse React", "Pulse Freq", "Pulse Width",
        "Brightness React", "Brightness Freq", "Brightness Width", "Brightness Smooth",
        "Motion React", "Motion Freq", "Motion Width",
        "Pulse Smooth", "Hue Shift", "Motion Power", "Smoothing Factor",
        "Noise Weight", "Audio Weight", "Audio Randomization", "Preamp Gain",
        "Lowpass Sensivity", "Lowpass Cutoff", "Cutoff", "Truncation",
        "Pulse Mode", "Pulse Power", "Motion Randomness", "Motion Smooth",
        "Tanh Output",
    ]

    static let defs: [String: SettingDef] = [
        "Pulse React":        SettingDef("Pulse", 0.47, 0, 100.0, 0.001, 2),
        "Pulse Freq":         SettingDef("Pulse", 1090, 50, 16000, 5, 0),
        "Pulse Width":        SettingDef("Pulse", 0.47, 0.25, 4, 0.01, 2),
        "Brightness React":   SettingDef("Pulse", 1.90, 0, 20.0, 0.001, 2),
        "Brightness Freq":    SettingDef("Pulse", 6268, 50, 16000, 5, 0),
        "Brightness Width":   SettingDef("Pulse", 0.65, 0.25, 4, 0.01, 2),
        "Brightness Smooth":  SettingDef("Pulse", 0.86, 0, 1, 0.01, 2),
        "Motion React":       SettingDef("Motion", 0.57, 0, 1.0, 0.001, 2),
        "Motion Freq":        SettingDef("Motion", 135, 50, 16000, 5, 0),
        "Motion Width":       SettingDef("Motion", 1.30, 0.25, 4, 0.01, 2),
        "Pulse Smooth":       SettingDef("Pulse", 0.86, 0, 1, 0.01, 2),
        "Hue Shift":          SettingDef("Post", 0.45, 0, 10.0, 0.01, 2),
        "Motion Power":       SettingDef("Motion", 1.39, 1.0, 2.0, 0.01, 2, hidden: true),
        "Smoothing Factor":   SettingDef("Timing", 1.0, 0, 1, 0.01, 2),
        "Noise Weight":       SettingDef("Input", 1.48, 0, 10, 0.01, 2, hidden: true),
        "Audio Weight":       SettingDef("Input", 0.1, 0, 2.0, 0.01, 2, hidden: true),
        "Audio Randomization": SettingDef("Input", 0.0, 0, 1, 0.01, 2, hidden: true),
        "Preamp Gain":        SettingDef("Audio", -12.0, -30, 60, 0.5, 1),
        "Lowpass Sensivity":  SettingDef("Audio", 0.47, 0, 5.0, 0.01, 2, hidden: true),
        "Lowpass Cutoff":     SettingDef("Audio", 128, 0, 256, 1, 0, hidden: true),
        "Cutoff":             SettingDef("Audio", 0, 0, 100, 1, 0, hidden: true),
        "Truncation":         SettingDef("Pulse", 1.0, 0, 2, 0.01, 2, hidden: true),
        "Pulse Mode":         SettingDef("Pulse", 3.0, 0, 4, 1, 0, hidden: true),
        "Pulse Power":        SettingDef("Pulse", 1.0, 0.5, 3, 0.01, 2, hidden: true),
        "Motion Randomness":  SettingDef("Motion", 0.50, 0, 1, 0.01, 2, hidden: true),
        "Motion Smooth":      SettingDef("Motion", 0.74, 0, 1, 0.01, 2),
        "Tanh Output":        SettingDef("Post", 0, 0, 1, 1, 0, hidden: true),
    ]

    private static let storageKey = "gan.settings.v1"
    private var values: [String: Double]

    init() {
        var initial: [String: Double] = [:]
        for (k, d) in Self.defs { initial[k] = d.value }
        if let saved = UserDefaults.standard.dictionary(forKey: Self.storageKey) {
            for (k, v) in saved {
                if let n = v as? Double, Self.defs[k] != nil { initial[k] = n }
            }
        }
        self.values = initial
    }

    func get(_ name: String) -> Double {
        values[name] ?? Self.defs[name]?.value ?? 0
    }

    func set(_ name: String, _ raw: Double) {
        guard let def = Self.defs[name] else { return }
        values[name] = Swift.min(Swift.max(raw, def.min), def.max)
        UserDefaults.standard.set(values, forKey: Self.storageKey)
        objectWillChange.send()
    }

    /// A two-way binding for a `Slider`.
    func binding(_ name: String) -> Binding<Double> {
        Binding(
            get: { [weak self] in self?.get(name) ?? 0 },
            set: { [weak self] in self?.set(name, $0) }
        )
    }
}
