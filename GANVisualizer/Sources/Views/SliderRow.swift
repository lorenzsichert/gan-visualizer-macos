import SwiftUI

struct SliderRow: View {
    @EnvironmentObject var state: AppState
    let name: String

    var body: some View {
        if let def = SettingsStore.defs[name] {
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(name).font(.caption)
                    Spacer()
                    Text(formatted(def))
                        .font(.caption).monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                Slider(value: state.settings.binding(name), in: def.min...def.max, step: def.step)
                    .controlSize(.small)
            }
        }
    }

    private func formatted(_ def: SettingDef) -> String {
        let v = state.settings.get(name)
        return def.dec > 0 ? String(format: "%.\(def.dec)f", v) : String(Int(v.rounded()))
    }
}
