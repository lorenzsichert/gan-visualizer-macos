import SwiftUI

struct ControlPanel: View {
    @EnvironmentObject var state: AppState
    @State private var tab = "general"

    private let tabs: [(id: String, label: String, color: Color)] = [
        ("general", "General", Color(red: 0.91, green: 0.63, blue: 0.23)),
        ("motion", "Motion", Color(red: 1.0, green: 0.85, blue: 0.40)),
        ("pulse", "Pulse", Color(red: 0.33, green: 0.76, blue: 0.95)),
        ("brightness", "Brightness", Color.white),
    ]

    var body: some View {
        VStack(spacing: 0) {
            header
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    modelSection
                    statsSection
                    actionsSection
                    slidersSection
                }
                .padding(14)
            }
        }
        .background(.ultraThinMaterial)
        .overlay(alignment: .leading) { Divider() }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Text("Controls").font(.headline)
            Spacer()
            Button {
                withAnimation(.easeInOut(duration: 0.2)) { state.panelCollapsed = true }
            } label: {
                Image(systemName: "chevron.right.2")
            }
            .buttonStyle(.plain)
            .help("Hide panel")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.thinMaterial)
        .overlay(alignment: .bottom) { Divider() }
    }

    // MARK: - Models

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Generator").font(.subheadline).bold()
            ForEach(state.modelManager.models) { m in
                Button {
                    Task { await state.load(m) }
                } label: {
                    HStack {
                        Image(systemName: state.modelManager.selected?.id == m.id
                              ? "largecircle.fill.circle" : "circle")
                            .foregroundStyle(state.modelManager.selected?.id == m.id ? Color.accentColor : .secondary)
                        Text(m.label)
                            .lineLimit(1)
                        Spacer()
                        if !m.sizeLabel.isEmpty {
                            Text(m.sizeLabel).font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(state.modelManager.busy)
            }
            if state.modelManager.models.isEmpty {
                Text("Discovering models…").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Stats

    private var statsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            stat("Status", state.modelManager.statusText, accent: state.modelReady)
            if let p = state.modelManager.progress {
                ProgressView(value: p).controlSize(.small)
            } else if state.modelManager.phase != "idle" {
                ProgressView().controlSize(.small)
            }
            HStack(spacing: 12) {
                stat("FPS", state.fps > 0 ? String(Int(state.fps.rounded())) : "--", accent: state.fps > 0)
                stat("Infer", state.inferMs > 0 ? String(format: "%.1fms", state.inferMs) : "--", accent: false)
                stat("Audio", audioLabel, accent: state.input != .off)
            }
            if let err = state.inputError {
                Text(err).font(.caption2).foregroundStyle(.red).lineLimit(3)
            }
            if let err = state.modelManager.errorMessage {
                Text(err).font(.caption2).foregroundStyle(.red).lineLimit(3)
            }
        }
    }

    private func stat(_ label: String, _ value: String, accent: Bool) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value)
                .font(.caption).monospacedDigit()
                .foregroundStyle(accent ? Color(red: 0.91, green: 0.63, blue: 0.23) : .primary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Actions

    private var actionsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Input").font(.caption2).foregroundStyle(.secondary)
            Picker("Input", selection: Binding(get: { state.input }, set: { state.setInput($0) })) {
                ForEach(AudioInput.allCases) { option in
                    Text(option.label).tag(option)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            HStack(spacing: 8) {
                Button {
                    state.toggleFullscreen()
                } label: {
                    Label("Fullscreen", systemImage: "arrow.up.left.and.arrow.down.right")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)

                Button {
                    state.cycleMirror()
                } label: {
                    Label(mirrorLabel, systemImage: "arrow.left.and.right.righttriangle.left.righttriangle.right")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(state.mirrorMode != 0 ? .accentColor : nil)
            }
        }
    }

    private var audioLabel: String {
        switch state.input {
        case .off: return "off"
        case .demo: return "demo"
        case .microphone: return "mic"
        case .system: return "system"
        }
    }

    private var mirrorLabel: String {
        switch state.mirrorMode {
        case 0: return "Mirror"
        case 1: return "Mirror"
        default: return "Mirror+Blur"
        }
    }

    // MARK: - Sliders

    private var slidersSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 4) {
                ForEach(tabs, id: \.id) { t in
                    Button {
                        tab = t.id
                    } label: {
                        Text(t.label)
                            .font(.caption).bold()
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 5)
                            .background(
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(tab == t.id ? t.color.opacity(0.20) : Color.clear)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 6)
                                    .stroke(tab == t.id ? t.color.opacity(0.7) : Color.clear, lineWidth: 1)
                            )
                    }
                    .buttonStyle(.plain)
                }
            }
            ForEach(state.settings.visibleKeys(tab: tab), id: \.self) { key in
                SliderRow(name: key)
            }
        }
    }
}
