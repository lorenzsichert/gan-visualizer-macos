import SwiftUI
import AppKit

// MARK: - Theme

extension Color {
    init(hex: UInt32, opacity: Double = 1) {
        let r = Double((hex >> 16) & 0xFF) / 255
        let g = Double((hex >> 8) & 0xFF) / 255
        let b = Double(hex & 0xFF) / 255
        self.init(.sRGB, red: r, green: g, blue: b, opacity: opacity)
    }
}

enum PanelTheme {
    static let bg = Color(hex: 0x0A0B0D)
    static let surface = Color(hex: 0x121316)
    static let surface2 = Color(hex: 0x17181C)
    static let surface3 = Color(hex: 0x1E2025)
    static let line = Color(hex: 0x25272D)
    static let lineStrong = Color(hex: 0x34363D)
    static let text = Color(hex: 0xE8EAEE)
    static let textMuted = Color(hex: 0x9AA1AB)
    static let textFaint = Color(hex: 0x60666F)
    static let accent = Color(hex: 0xE8EAEE)
    static let accentLine = Color.white.opacity(0.40)
    static let pulse = Color(hex: 0x53C1F1)
    static let motion = Color(hex: 0xFFD966)
    static let brightness = Color.white

    static func tabSoft(_ tab: String) -> Color {
        switch tab {
        case "motion": return motion.opacity(0.16)
        case "pulse": return pulse.opacity(0.16)
        case "brightness": return brightness.opacity(0.14)
        default: return Color.white.opacity(0.14)
        }
    }

    static func tabColor(_ tab: String) -> Color {
        switch tab {
        case "motion": return motion
        case "pulse": return pulse
        case "brightness": return brightness
        default: return text
        }
    }
}

private struct ControlTab: Identifiable {
    let id: String
    let label: String
    var color: Color { PanelTheme.tabColor(id) }
    var soft: Color { PanelTheme.tabSoft(id) }
}

// MARK: - Panel

struct ControlPanel: View {
    @EnvironmentObject var state: AppState
    @State private var tab = "general"
    @State private var openTab: String?
    @State private var expandedFamily: String?

    private let tabs: [ControlTab] = [
        ControlTab(id: "general", label: "General"),
        ControlTab(id: "motion", label: "Motion"),
        ControlTab(id: "pulse", label: "Pulse"),
        ControlTab(id: "brightness", label: "Bright"),
    ]

    var body: some View {
        VStack(spacing: 0) {
            header
            ScrollView(.vertical, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 22) {
                    StatsSection()
                    ActionsSection()
                    SpectrumRow(tab: $tab, openTab: $openTab)
                    HueSlider {
                        tab = "general"
                        if openTab != nil { openTab = "general" }
                    }
                    TabsSection(tab: $tab, openTab: $openTab, tabs: tabs)
                    ModelGallerySection(expandedFamily: $expandedFamily)
                }
                .padding(.horizontal, 16)
                .padding(.top, 14)
                .padding(.bottom, 20)
            }
        }
        .frame(width: 320)
        .background(
            ZStack {
                Rectangle().fill(.ultraThinMaterial)
                PanelTheme.surface.opacity(0.88)
            }
        )
        .overlay(alignment: .leading) {
            Rectangle().fill(PanelTheme.line).frame(width: 1)
        }
        .shadow(color: .black.opacity(0.45), radius: 24, x: -24, y: 0)
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text("Controls")
                .font(.system(size: 11))
                .tracking(1.5)
                .foregroundStyle(PanelTheme.textMuted)
            Spacer()
            Button {
                withAnimation(.easeInOut(duration: 0.22)) { state.panelCollapsed = true }
            } label: {
                Image(systemName: "chevron.right.2")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(PanelTheme.textMuted)
                    .frame(width: 30, height: 30)
                    .background(
                        RoundedRectangle(cornerRadius: 7)
                            .stroke(PanelTheme.lineStrong, lineWidth: 1)
                    )
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Hide panel")
        }
        .padding(.leading, 16)
        .padding(.trailing, 10)
        .frame(height: 52)
        .overlay(alignment: .bottom) {
            Rectangle().fill(PanelTheme.line).frame(height: 1)
        }
    }
}

// MARK: - Telemetry

private struct StatsSection: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            VStack(spacing: 1) {
                StatusCell(
                    text: state.modelManager.statusText,
                    progress: state.modelManager.progress,
                    busy: state.modelManager.phase != "idle")
                HStack(spacing: 1) {
                    StatCell(
                        label: "FPS",
                        value: state.fps > 0 ? String(Int(state.fps.rounded())) : "--",
                        on: state.fps > 0)
                    StatCell(
                        label: "Infer",
                        value: state.inferMs > 0 ? String(format: "%.1fms", state.inferMs) : "--",
                        on: false)
                    StatCell(label: "Audio", value: audioLabel, on: state.input != .off)
                }
            }
            .background(PanelTheme.line)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(PanelTheme.line, lineWidth: 1)
            )

            if let error = state.inputError ?? state.modelManager.errorMessage {
                Text(error)
                    .font(.system(size: 9.5))
                    .foregroundStyle(Color(hex: 0xE0604F))
                    .lineLimit(3)
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
}

private struct StatCell: View {
    let label: String
    let value: String
    let on: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 9.5))
                .tracking(1.3)
                .foregroundStyle(PanelTheme.textFaint)
            Text(value)
                .font(.system(size: 12, design: .monospaced))
                .monospacedDigit()
                .foregroundStyle(on ? PanelTheme.accent : PanelTheme.textMuted)
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(PanelTheme.surface2)
    }
}

private struct StatusCell: View {
    let text: String
    let progress: Double?
    let busy: Bool

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                PanelTheme.surface2
                if let p = progress {
                    LinearGradient(
                        colors: [Color.white.opacity(0.26), Color.white.opacity(0.10)],
                        startPoint: .leading, endPoint: .trailing)
                        .frame(width: geo.size.width * CGFloat(min(1, max(0, p))))
                        .shadow(color: .white.opacity(0.28), radius: 3, x: 2)
                } else if busy {
                    SweepBar()
                }
                HStack(spacing: 12) {
                    Text("Status")
                        .font(.system(size: 9.5))
                        .tracking(1.3)
                        .foregroundStyle(PanelTheme.textFaint)
                        .fixedSize()
                    Spacer(minLength: 0)
                    Text(text)
                        .font(.system(size: 12, design: .monospaced))
                        .monospacedDigit()
                        .foregroundStyle(PanelTheme.textMuted)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
                .padding(.horizontal, 10)
            }
        }
        .frame(height: 36)
    }
}

private struct SweepBar: View {
    @State private var phase: CGFloat = -0.4

    var body: some View {
        GeometryReader { geo in
            LinearGradient(
                colors: [.clear, Color.white.opacity(0.34), .clear],
                startPoint: .leading, endPoint: .trailing)
                .frame(width: geo.size.width * 0.38)
                .offset(x: phase * geo.size.width)
                .onAppear {
                    withAnimation(.easeInOut(duration: 1.1).repeatForever(autoreverses: false)) {
                        phase = 1.2
                    }
                }
        }
        .clipped()
    }
}

// MARK: - Actions

private struct ActionsSection: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(spacing: 8) {
            PanelActionButton(
                title: state.input == .microphone ? "Microphone: ON" : "Enable Microphone",
                systemImage: "mic",
                primary: true,
                active: state.input == .microphone
            ) {
                state.setInput(state.input == .microphone ? .off : .microphone)
            }

            PanelActionButton(
                title: state.input == .system ? "System Audio: ON" : "System Audio",
                systemImage: "display",
                active: state.input == .system
            ) {
                state.setInput(state.input == .system ? .off : .system)
            }

            PanelActionButton(
                title: "Demo signal",
                systemImage: "waveform",
                active: state.input == .demo
            ) {
                state.setInput(state.input == .demo ? .off : .demo)
            }

            PanelActionButton(
                title: state.paused ? "Resume" : "Pause",
                systemImage: state.paused ? "play.fill" : "pause.fill",
                active: state.paused
            ) {
                state.togglePause()
            }

            PanelActionButton(title: "Fullscreen", systemImage: "arrow.up.left.and.arrow.down.right") {
                state.toggleFullscreen()
            }

            PanelActionButton(
                title: mirrorLabel,
                systemImage: "arrow.left.and.right",
                active: state.mirrorMode != 0
            ) {
                state.cycleMirror()
            }
        }
    }

    private var mirrorLabel: String {
        state.mirrorMode >= 2 ? "Mirror + Blur" : "Mirror"
    }
}

private struct PanelActionButton: View {
    let title: String
    let systemImage: String
    var primary = false
    var active = false
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 9) {
                Image(systemName: systemImage)
                    .font(.system(size: 13, weight: .regular))
                    .frame(width: 16, height: 16)
                    .foregroundStyle(iconColor)
                Text(title)
                    .font(.system(size: 12))
                    .lineLimit(1)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 12)
            .frame(maxWidth: .infinity)
            .frame(height: 36)
            .foregroundStyle(foreground)
            .background(
                RoundedRectangle(cornerRadius: 7)
                    .fill(background)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 7)
                    .stroke(border, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
    }

    private var foreground: Color {
        if primary {
            return active ? PanelTheme.accent : PanelTheme.bg
        }
        return active ? PanelTheme.accent : PanelTheme.text
    }

    private var iconColor: Color {
        if primary {
            return active ? PanelTheme.accent : PanelTheme.bg
        }
        if active { return PanelTheme.accent }
        return hovering ? PanelTheme.textMuted : PanelTheme.textFaint
    }

    private var background: Color {
        if primary {
            return active ? .clear : PanelTheme.accent
        }
        return hovering ? PanelTheme.surface3 : PanelTheme.surface2
    }

    private var border: Color {
        if primary {
            return active ? PanelTheme.accentLine : PanelTheme.accent
        }
        if active { return PanelTheme.accentLine }
        return hovering ? PanelTheme.textFaint : PanelTheme.lineStrong
    }
}

// MARK: - Spectrum + preamp

private struct PreampHint: Equatable {
    let value: Double
    let y: CGFloat
}

private struct SpectrumRow: View {
    @Binding var tab: String
    @Binding var openTab: String?
    @State private var preampHint: PreampHint?

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            VStack(spacing: 4) {
                PreampSlider(hint: $preampHint)
                InputIndicator()
            }
            SpectrumGraph(activeTab: tab) { id in
                tab = id
                // Dragging a band switches tabs but keeps the explainer open
                // only if one was already open, matching the web panel.
                if openTab != nil { openTab = id }
            }
        }
        // The hint lives above the whole row so the spectrum canvas cannot
        // draw over it. Canvas content composites in its own layer and would
        // otherwise cover a sibling overlay.
        .overlay(alignment: .topLeading) {
            if let hint = preampHint {
                PreampReadout(value: hint.value)
                    .fixedSize()
                    .offset(x: 22, y: hint.y)
                    .allowsHitTesting(false)
            }
        }
    }
}

private struct PreampSlider: View {
    @EnvironmentObject var state: AppState
    @Binding var hint: PreampHint?
    @State private var hovering = false
    @State private var dragging = false

    private var def: SettingDef {
        SettingsStore.defs["Preamp Gain"] ?? SettingDef("Audio", -12, -30, 60, 0.5, 1)
    }

    private var enabled: Bool {
        state.input == .microphone || state.input == .system
    }

    var body: some View {
        GeometryReader { geo in
            let h = geo.size.height
            let value = state.settings.get("Preamp Gain")
            let span = def.max - def.min
            let t = CGFloat((value - def.min) / (span > 0 ? span : 1))
            let fillHeight = max(6, h * t)

            ZStack(alignment: .bottom) {
                Capsule()
                    .fill(Color.white.opacity(0.07))
                    .overlay(Capsule().stroke(Color.white.opacity(0.06), lineWidth: 1))
                    .frame(width: 8)
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [Color.white.opacity(0.82), .white],
                            startPoint: .bottom, endPoint: .top)
                    )
                    .frame(width: 8, height: fillHeight)
                    .shadow(color: .white.opacity(0.2), radius: 5)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { drag in
                        dragging = true
                        let frac = 1 - min(1, max(0, drag.location.y / max(h, 1)))
                        let raw = def.min + Double(frac) * span
                        let next = quantize(raw, step: def.step)
                        state.settings.set("Preamp Gain", next)
                        let fill = max(6, h * CGFloat((next - def.min) / (span > 0 ? span : 1)))
                        hint = PreampHint(value: next, y: readoutY(height: h, fill: fill))
                    }
                    .onEnded { _ in
                        dragging = false
                        if hovering {
                            hint = PreampHint(value: state.settings.get("Preamp Gain"),
                                              y: readoutY(height: h, fill: fillHeight))
                        } else {
                            hint = nil
                        }
                    }
            )
            .onHover { over in
                hovering = over
                if over {
                    hint = PreampHint(value: value,
                                      y: readoutY(height: h, fill: fillHeight))
                } else if !dragging {
                    hint = nil
                }
            }
        }
        .frame(width: 14, height: 90)
        .opacity(enabled ? 1 : 0.3)
        .allowsHitTesting(enabled)
        .onChange(of: enabled) { _, on in
            if !on {
                hovering = false
                dragging = false
                hint = nil
            }
        }
        .onDisappear { hint = nil }
    }

    private func readoutY(height: CGFloat, fill: CGFloat) -> CGFloat {
        min(max(height - fill - 14, 0), height - 42)
    }

    private func quantize(_ value: Double, step: Double) -> Double {
        guard step > 0 else { return value }
        return (value / step).rounded() * step
    }
}

private struct PreampReadout: View {
    let value: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text("PREAMP")
                .font(.system(size: 8))
                .tracking(1.1)
                .foregroundStyle(PanelTheme.textFaint)
            Text(formattedDb(value))
                .font(.system(size: 10, design: .monospaced))
                .monospacedDigit()
                .foregroundStyle(PanelTheme.text)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(PanelTheme.bg.opacity(0.92))
        .clipShape(RoundedRectangle(cornerRadius: 7))
        .overlay(
            RoundedRectangle(cornerRadius: 7)
                .stroke(PanelTheme.lineStrong, lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.45), radius: 10, y: 8)
    }

    private func formattedDb(_ value: Double) -> String {
        let number = String(format: "%.1f", abs(value))
        let sign = value < 0 ? "−" : value > 0 ? "+" : ""
        return "\(sign)\(number) dB"
    }
}

private struct InputIndicator: View {
    @EnvironmentObject var state: AppState
    @State private var hovering = false

    private var icon: String {
        switch state.input {
        case .system: return "display"
        case .demo: return "waveform"
        case .microphone: return "mic"
        case .off: return "mic.slash"
        }
    }

    private var on: Bool { state.input != .off }

    private var label: String {
        if state.input == .microphone { return "Microphone on. Click to turn off." }
        if state.input == .system { return "System audio on. Click to switch to microphone." }
        if state.input == .demo { return "Demo signal on. Click to enable microphone." }
        return "Microphone off. Click to enable microphone."
    }

    var body: some View {
        Button {
            if state.input == .microphone { state.setInput(.off) }
            else { state.setInput(.microphone) }
        } label: {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .regular))
                .foregroundStyle(on || hovering ? PanelTheme.text : PanelTheme.textFaint)
                .opacity(on ? 1 : 0.7)
                .frame(width: 20, height: 20)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .help(label)
    }
}

private struct HueSlider: View {
    @EnvironmentObject var state: AppState
    var onSelect: () -> Void = {}
    @State private var hovering = false
    @State private var dragging = false

    private var def: SettingDef {
        SettingsStore.defs["Hue Shift"] ?? SettingDef("Post", 0.45, 0, 10, 0.01, 2)
    }

    var body: some View {
        GeometryReader { geo in
            let value = state.settings.get("Hue Shift")
            let span = def.max - def.min
            let t = CGFloat((value - def.min) / (span > 0 ? span : 1))
            let clamped = min(1, max(0, t))
            let trackX: CGFloat = 35
            let trackW = max(geo.size.width - trackX, 1)

            ZStack(alignment: .topLeading) {
                Image(systemName: "waveform")
                    .font(.system(size: 19, weight: .regular))
                    .foregroundStyle(PanelTheme.text)
                    .frame(width: 25, height: 25)

                Capsule()
                    .fill(Color.white.opacity(0.07))
                    .overlay(Capsule().stroke(Color.white.opacity(0.06), lineWidth: 1))
                    .frame(width: trackW, height: 8)
                    .offset(x: trackX, y: 8.5)

                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [Color.white.opacity(0.82), .white],
                            startPoint: .leading, endPoint: .trailing)
                    )
                    .frame(width: max(4, trackW * clamped), height: 8)
                    .offset(x: trackX, y: 8.5)
                    .shadow(color: .white.opacity(0.2), radius: 5)

                // The interaction layer spans the whole slider, and the track
                // inset is subtracted from the drag location. Attaching the
                // gesture to an offset view reports coordinates that do not
                // line up with the visible track.
                Color.clear
                    .frame(width: geo.size.width, height: 25)
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { drag in
                                if !dragging { onSelect() }
                                dragging = true
                                let x = drag.location.x - trackX
                                let frac = Double(min(1, max(0, x / trackW)))
                                let raw = def.min + frac * span
                                state.settings.set("Hue Shift", quantize(raw, step: def.step))
                            }
                            .onEnded { _ in dragging = false }
                    )

                if hovering || dragging {
                    readout(value: value)
                        .fixedSize()
                        .offset(x: min(max(trackX + clamped * trackW - 45, 0), geo.size.width - 96),
                                y: -31)
                        .allowsHitTesting(false)
                }
            }
        }
        .frame(height: 25)
        .onHover { hovering = $0 }
        .help("Hue Shift: how far the colors shift as the music changes.")
    }

    private func readout(value: Double) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text("HUE SHIFT")
                .font(.system(size: 8))
                .tracking(1.1)
                .foregroundStyle(PanelTheme.textFaint)
            Text(String(format: "%.\(def.dec)f", value))
                .font(.system(size: 10, design: .monospaced))
                .monospacedDigit()
                .foregroundStyle(PanelTheme.text)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(PanelTheme.bg.opacity(0.92))
        .clipShape(RoundedRectangle(cornerRadius: 7))
        .overlay(
            RoundedRectangle(cornerRadius: 7)
                .stroke(PanelTheme.lineStrong, lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.45), radius: 10, y: 8)
    }

    private func quantize(_ value: Double, step: Double) -> Double {
        guard step > 0 else { return min(max(value, def.min), def.max) }
        let stepped = (value / step).rounded() * step
        return min(max(stepped, def.min), def.max)
    }
}

/// Drives the spectrum at a fixed 60 Hz. SwiftUI's TimelineView is coalesced
/// with other UI updates, so a small live canvas can stall; this tick force-
/// updates the view every display interval regardless of the rest of the app.
private final class SpectrumTicker: ObservableObject {
    @Published private(set) var frame: UInt64 = 0
    private var timer: Timer?

    init() {
        let timer = Timer(timeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            self?.frame &+= 1
        }
        timer.tolerance = 0.001
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    deinit {
        timer?.invalidate()
    }
}

private struct FilterBand {
    let id: String
    let color: Color
    let freq: Float
    let width: Float
    let react: Float
    let reactMin: Float
    let reactMax: Float
}

private struct SpectrumGraph: View {
    @EnvironmentObject var state: AppState
    let activeTab: String
    let onSelect: (String) -> Void

    private let fMin: Float = 50
    private let fMax: Float = 18000
    private let refMag: Float = 128
    private let refDb: Float = -80
    private let samples = 1024

    @StateObject private var ticker = SpectrumTicker()

    @State private var dragStart: CGPoint?
    @State private var lastPoint: CGPoint?
    @State private var dragMoved = false
    @State private var widthMode = false
    @State private var widthStart: Float = 1
    @State private var activeBandID: String?
    @State private var longPressTask: Task<Void, Never>?
    @State private var flashBandID: String?
    @State private var flashStart: Date?

    var body: some View {
        GeometryReader { geo in
            Canvas(rendersAsynchronously: false) { context, size in
                draw(&context, size, frame: ticker.frame)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        if dragStart == nil { begin(at: value.location, size: geo.size) }
                        update(at: value.location, size: geo.size)
                    }
                    .onEnded { _ in endDrag() }
            )
        }
        .frame(height: 110)
        .background(PanelTheme.surface2)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(PanelTheme.line, lineWidth: 1)
        )
        .help("A-weighted spectrum. Drag a colored point to retune its band: horizontal = frequency, vertical = reaction. Hold a point for a moment, then slide up or down to change width.")
    }

    private var logMin: Float { log(fMin) }
    private var logSpan: Float { log(fMax / fMin) }

    private func draw(_ ctx: inout GraphicsContext, _ size: CGSize, frame: UInt64) {
        _ = frame
        guard size.width > 1, size.height > 1 else { return }

        let bins = AudioPipeline.bins
        var spec = [Float](repeating: 0, count: bins)
        if state.input == .demo && !state.audio.active {
            LatentEngine.genDemoSpectrum(ProcessInfo.processInfo.systemUptime, &spec)
        } else {
            state.audio.copySpectrum(into: &spec)
        }

        let sr = Float(max(state.audio.lastSampleRate, 44100))
        let nyquist = max(sr / 2, fMin * 1.01)
        let displayMax = min(fMax, nyquist)
        let logMinD = log(fMin)
        let logSpanD = log(displayMax / fMin)
        let lastBin = Float(max(bins - 1, 1))
        let preamp = Float(pow(10, state.settings.get("Preamp Gain") / 20))

        let top: CGFloat = 6
        let baseY = size.height
        let plotH = max(baseY - top, 1)
        let denom = Float(max(samples - 1, 1))

        // Resample the raw spectrum onto a fixed log-frequency grid. Max
        // magnitude per slice, A-weighted, then referenced to the same 128
        // full-scale magnitude the web app uses. Without that reference the
        // curve pins to the top and looks tilted.
        var levels = [CGFloat](repeating: 0, count: samples)
        for i in 0..<samples {
            let t0 = Float(i) / denom
            let t1 = Float(i + 1) / denom
            let f0 = exp(logMinD + logSpanD * t0)
            let f1 = exp(logMinD + logSpanD * t1)
            let b0 = max(1, min(bins - 1, Int((f0 / nyquist) * lastBin)))
            let b1 = min(bins - 1, max(b0 + 1, Int(ceil((f1 / nyquist) * lastBin))))

            var mag: Float = 0
            if b0 <= b1 {
                for b in b0...b1 where spec[b] > mag { mag = spec[b] }
            }
            let scaled = max(mag * preamp / refMag, 1e-9)
            let db = 20 * log10(scaled) + LatentEngine.aWeightDb(f0)
            levels[i] = CGFloat(min(1, max(0, (db - refDb) / -refDb)))
        }

        let smooth = blur3(levels)
        let bands = bands()
        let flashElapsed = flashStart.map { Date().timeIntervalSince($0) } ?? 10
        let flashProgress = min(1, max(0, flashElapsed / 0.62))

        // Bands first, exactly like the web app: soft Gaussian envelope, center
        // hairline and the grab knob at the envelope peak.
        for band in bands where band.freq > 0 {
            let active = band.id == activeTab
            let flashing = flashBandID == band.id && flashProgress < 1
            let highlighted = active || flashing
            let fc = min(max(band.freq, fMin), displayMax)
            let sigma = max(band.width, 0.05)
            let inv = 1 / (2 * sigma * sigma)
            let range = max(band.reactMax - band.reactMin, 1e-6)
            let gain = CGFloat(min(1, max(0, (band.react - band.reactMin) / range)))

            var bell = Path()
            for i in 0..<samples {
                let t = Float(i) / denom
                let fq = exp(logMinD + logSpanD * t)
                let d = log2(fq / fc)
                let env = CGFloat(exp(-d * d * inv)) * gain
                let x = size.width * CGFloat(i) / CGFloat(samples - 1)
                let y = baseY - env * plotH
                if i == 0 { bell.move(to: CGPoint(x: x, y: y)) }
                else { bell.addLine(to: CGPoint(x: x, y: y)) }
            }

            var bellFill = bell
            bellFill.addLine(to: CGPoint(x: size.width, y: baseY))
            bellFill.addLine(to: CGPoint(x: 0, y: baseY))
            bellFill.closeSubpath()
            ctx.fill(bellFill, with: .color(band.color.opacity(highlighted ? 0.07 : 0.05)))
            ctx.stroke(bell, with: .color(band.color.opacity(highlighted ? 0.42 : 0.26)),
                       lineWidth: 1)

            let tx = (log(fc) - logMinD) / logSpanD
            let cx = CGFloat(min(1, max(0, tx))) * size.width
            let dotY = baseY - gain * plotH

            var hairline = Path()
            hairline.move(to: CGPoint(x: cx, y: baseY))
            hairline.addLine(to: CGPoint(x: cx, y: dotY))
            ctx.stroke(hairline, with: .color(band.color.opacity(0.30)), lineWidth: 1)

            var dot = Path()
            dot.addEllipse(in: CGRect(x: cx - 4.5, y: dotY - 4.5, width: 9, height: 9))
            ctx.fill(dot, with: .color(band.color))

            let flashPulse = CGFloat(sin(Double.pi * flashProgress))
            let ringRadius: CGFloat = flashing ? 8 + 9 * (1 - flashPulse) : 8
            let ringAlpha: Double = flashing
                ? Double(flashPulse)
                : (highlighted ? 0.45 : 0.28)
            var ring = Path()
            ring.addEllipse(in: CGRect(x: cx - ringRadius, y: dotY - ringRadius,
                                       width: ringRadius * 2, height: ringRadius * 2))
            ctx.stroke(ring, with: .color(band.color.opacity(ringAlpha)),
                       lineWidth: flashing ? 2 : 1)
        }

        // Filled area under the spectrum, fading toward the bottom.
        var fill = Path()
        fill.move(to: CGPoint(x: 0, y: baseY))
        for i in 0..<samples {
            let x = size.width * CGFloat(i) / CGFloat(samples - 1)
            let y = baseY - smooth[i] * plotH
            fill.addLine(to: CGPoint(x: x, y: y))
        }
        fill.addLine(to: CGPoint(x: size.width, y: baseY))
        fill.closeSubpath()
        let stops = [
            Gradient.Stop(color: Color.white.opacity(0.38), location: 0),
            Gradient.Stop(color: Color.white.opacity(0.16), location: 0.55),
            Gradient.Stop(color: Color.white.opacity(0.04), location: 1),
        ]
        ctx.fill(
            fill,
            with: .linearGradient(
                Gradient(stops: stops),
                startPoint: CGPoint(x: 0, y: top),
                endPoint: CGPoint(x: 0, y: baseY)))

        // Crisp white edge, drawn through segment midpoints so no corners show.
        var curve = Path()
        var started = false
        var px: CGFloat = 0
        var py: CGFloat = 0
        for i in 0..<samples {
            let x = size.width * CGFloat(i) / CGFloat(samples - 1)
            let y = baseY - smooth[i] * plotH
            if !started {
                if smooth[i] < 1e-3 { continue }
                curve.move(to: CGPoint(x: x, y: y))
                px = x
                py = y
                started = true
                continue
            }
            let mx = (px + x) / 2
            let my = (py + y) / 2
            curve.addQuadCurve(to: CGPoint(x: mx, y: my),
                               control: CGPoint(x: px, y: py))
            px = x
            py = y
        }
        if started { curve.addLine(to: CGPoint(x: px, y: py)) }
        ctx.stroke(curve, with: .color(Color.white.opacity(0.92)), lineWidth: 1)

        // Highlighted band, applied to the display spectrum, in its color.
        if let active = bands.first(where: { $0.id == activeTab }), active.freq > 0 {
            let fc = min(max(active.freq, fMin), displayMax)
            let sigma = max(active.width, 0.05)
            let inv = 1 / (2 * sigma * sigma)
            let step = logSpanD / Float(max(samples - 1, 1))
            var weights = [CGFloat](repeating: 0, count: samples)
            var first: Int?
            var last: Int?
            for i in 0..<samples {
                let d = (logMinD + step * Float(i) - log(fc)) / Float(log(2.0))
                let w = CGFloat(exp(-d * d * inv))
                weights[i] = w
                if smooth[i] * w >= 1e-3 {
                    if first == nil { first = i }
                    last = i
                }
            }

            if let i0 = first, let i1 = last, i1 >= i0 {
                let xAt = { (i: Int) in size.width * CGFloat(i) / CGFloat(samples - 1) }
                let yAt = { (i: Int) in baseY - smooth[i] * weights[i] * plotH }

                var bandFill = Path()
                bandFill.move(to: CGPoint(x: xAt(i0), y: baseY))
                for i in i0...i1 { bandFill.addLine(to: CGPoint(x: xAt(i), y: yAt(i))) }
                bandFill.addLine(to: CGPoint(x: xAt(i1), y: baseY))
                bandFill.closeSubpath()
                ctx.fill(bandFill, with: .color(active.color.opacity(0.18)))

                var bandCurve = Path()
                for i in i0...i1 {
                    let point = CGPoint(x: xAt(i), y: yAt(i))
                    if i == i0 { bandCurve.move(to: point) }
                    else { bandCurve.addLine(to: point) }
                }
                ctx.stroke(bandCurve, with: .color(active.color.opacity(0.95)), lineWidth: 2)
            }
        }
    }

    /// Three box-blur passes with a frequency-tapered radius, the Swift port of
    /// the web view's `smoothLevel`. Low frequencies get a wide blur (smooth
    /// bass, hides FFT stair-steps); treble keeps its detail.
    private func blur3(_ src: [CGFloat]) -> [CGFloat] {
        let n = src.count
        guard n > 2 else { return src }

        let sigmaLow: CGFloat = 20
        let sigmaHigh: CGFloat = 0.625
        var radii = [Int](repeating: 0, count: n)
        for x in 0..<n {
            let t = CGFloat(x) / CGFloat(n - 1)
            let sigma = sigmaLow + (sigmaHigh - sigmaLow) * t
            radii[x] = max(0, Int((((4 * sigma * sigma + 1).squareRoot() - 1) / 2).rounded()))
        }
        let rMax = radii.max() ?? 0
        if rMax == 0 { return src }

        var a = src
        var b = [CGFloat](repeating: 0, count: n)
        var prefix = [CGFloat](repeating: 0, count: n + 2 * rMax + 1)

        for _ in 0..<3 {
            var acc: CGFloat = 0
            prefix[0] = 0
            for i in 0..<rMax { acc += a[0]; prefix[i + 1] = acc }
            for i in 0..<n { acc += a[i]; prefix[rMax + 1 + i] = acc }
            for i in 0..<rMax { acc += a[n - 1]; prefix[rMax + n + 1 + i] = acc }

            for x in 0..<n {
                let r = radii[x]
                let lo = x + rMax - r
                let count = 2 * r + 1
                b[x] = (prefix[lo + count] - prefix[lo]) / CGFloat(count)
            }
            swap(&a, &b)
        }
        return a
    }

    private func bands() -> [FilterBand] {
        [
            FilterBand(
                id: "pulse", color: PanelTheme.pulse,
                freq: Float(state.settings.get("Pulse Freq")),
                width: Float(state.settings.get("Pulse Width")),
                react: Float(state.settings.get("Pulse React")),
                reactMin: Float(SettingsStore.defs["Pulse React"]?.min ?? 0),
                reactMax: Float(SettingsStore.defs["Pulse React"]?.max ?? 150)),
            FilterBand(
                id: "brightness", color: PanelTheme.brightness,
                freq: Float(state.settings.get("Brightness Freq")),
                width: Float(state.settings.get("Brightness Width")),
                react: Float(state.settings.get("Brightness React")),
                reactMin: Float(SettingsStore.defs["Brightness React"]?.min ?? 0),
                reactMax: Float(SettingsStore.defs["Brightness React"]?.max ?? 8)),
            FilterBand(
                id: "motion", color: PanelTheme.motion,
                freq: Float(state.settings.get("Motion Freq")),
                width: Float(state.settings.get("Motion Width")),
                react: Float(state.settings.get("Motion React")),
                reactMin: Float(SettingsStore.defs["Motion React"]?.min ?? 0),
                reactMax: Float(SettingsStore.defs["Motion React"]?.max ?? 1)),
        ]
    }

    private func begin(at point: CGPoint, size: CGSize) {
        guard size.width > 1, size.height > 1 else { return }
        let band = nearestBand(at: point, size: size)
        dragStart = point
        lastPoint = point
        dragMoved = false
        widthMode = false
        activeBandID = band.id
        widthStart = band.width
        flash(band.id)
        haptic(.alignment)
        onSelect(band.id)

        longPressTask?.cancel()
        longPressTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 650_000_000)
            guard !Task.isCancelled, dragStart != nil, !dragMoved, let band = activeBand else {
                return
            }
            widthMode = true
            widthStart = band.width
            dragStart = lastPoint ?? point
            flash(band.id)
            haptic(.levelChange)
        }
    }

    private func update(at point: CGPoint, size: CGSize) {
        guard size.width > 1, size.height > 1 else { return }
        lastPoint = point

        if let start = dragStart {
            let dx = point.x - start.x
            let dy = point.y - start.y
            if dx * dx + dy * dy > 100 {
                dragMoved = true
                if !widthMode { longPressTask?.cancel(); longPressTask = nil }
            }
        }

        let band = nearestBand(at: point, size: size)
        let f = frequency(at: point, size: size)

        if widthMode, let id = activeBandID {
            // Long press: vertical drag adjusts width instead of reaction.
            let startY = dragStart?.y ?? point.y
            let width = widthStart * exp(Float(startY - point.y) * 0.007)
            setFrequency(id, f)
            setWidth(id, min(4, max(0.25, width)))
            return
        }

        let top: CGFloat = 6
        let plotH = max(size.height - top, 1)
        let ty = 1 - Double(min(1, max(0, (point.y - top) / plotH)))
        setFrequency(band.id, f)
        setReact(band.id, ty)
    }

    private func endDrag() {
        longPressTask?.cancel()
        longPressTask = nil
        dragStart = nil
        lastPoint = nil
        dragMoved = false
        widthMode = false
        activeBandID = nil
    }

    private var activeBand: FilterBand? {
        guard let id = activeBandID else { return nil }
        return bands().first { $0.id == id }
    }

    private func frequency(at point: CGPoint, size: CGSize) -> Float {
        let tx = Float(min(1, max(0, point.x / size.width)))
        return exp(logMin + tx * logSpan)
    }

    private func nearestBand(at point: CGPoint, size: CGSize) -> FilterBand {
        let f = frequency(at: point, size: size)
        let all = bands()
        return all.min {
            abs(log($0.freq / f)) < abs(log($1.freq / f))
        } ?? all[0]
    }

    private func setFrequency(_ id: String, _ f: Float) {
        let key: String
        switch id {
        case "pulse": key = "Pulse Freq"
        case "brightness": key = "Brightness Freq"
        default: key = "Motion Freq"
        }
        state.settings.set(key, (Double(f) / 5).rounded() * 5)
    }

    private func setReact(_ id: String, _ normalized: Double) {
        let key: String
        switch id {
        case "pulse": key = "Pulse React"
        case "brightness": key = "Brightness React"
        default: key = "Motion React"
        }
        let lower = SettingsStore.defs[key]?.min ?? 0
        let upper = SettingsStore.defs[key]?.max ?? 1
        state.settings.set(key, lower + normalized * (upper - lower))
    }

    private func setWidth(_ id: String, _ width: Float) {
        let key: String
        switch id {
        case "pulse": key = "Pulse Width"
        case "brightness": key = "Brightness Width"
        default: key = "Motion Width"
        }
        state.settings.set(key, Double(width))
    }

    private func flash(_ id: String) {
        flashBandID = id
        flashStart = Date()
    }

    private func haptic(_ pattern: NSHapticFeedbackManager.FeedbackPattern) {
        NSHapticFeedbackManager.defaultPerformer.perform(pattern, performanceTime: .now)
    }
}

// MARK: - Tabs

private struct TabHelpItem {
    let name: String
    let text: String
}

private struct TabHelpContent {
    let intro: String
    var items: [TabHelpItem] = []
}

private enum TabHelp {
    static let all: [String: TabHelpContent] = [
        "general": TabHelpContent(
            intro: "The latent vector of the image generator gets pushed by the music. The preamp beside the spectrum sets the input level, and the Hue Shift slider below the spectrum colors the image. Sliding Hue Shift brings you back to this tab.",
            items: [
                TabHelpItem(
                    name: "Hue Shift",
                    text: "How far the colors shift as the music changes. Higher values give stronger color changes."),
            ]),
        "motion": TabHelpContent(
            intro: "Motion moves the structure of the image. The yellow band on the spectrum sets where it listens. Drag left or right for frequency, up or down for strength, or hold the dot for a moment and slide up or down for width."),
        "pulse": TabHelpContent(
            intro: "Pulse makes the image pulsate on the beat. The blue band on the spectrum sets where it listens. Drag left or right for frequency, up or down for strength, or hold the dot for a moment and slide up or down for width."),
        "brightness": TabHelpContent(
            intro: "Brightness makes the image brighter on the beat. The white band on the spectrum sets where it listens. Drag left or right for frequency, up or down for strength, or hold the dot for a moment and slide up or down for width."),
    ]
}

private struct TabsSection: View {
    @Binding var tab: String
    @Binding var openTab: String?
    let tabs: [ControlTab]

    var body: some View {
        VStack(spacing: 16) {
            HStack(spacing: 6) {
                ForEach(tabs) { item in
                    Button {
                        if tab == item.id {
                            openTab = openTab == item.id ? nil : item.id
                        } else {
                            tab = item.id
                            openTab = item.id
                        }
                    } label: {
                        HStack(spacing: 3) {
                            Text(item.label)
                                .font(.system(size: 11))
                            if tab == item.id {
                                Image(systemName: "chevron.down")
                                    .font(.system(size: 8, weight: .semibold))
                                    .rotationEffect(.degrees(openTab == item.id ? 180 : 0))
                            }
                        }
                        .foregroundStyle(item.color)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 7)
                        .background(
                            RoundedRectangle(cornerRadius: 7)
                                .fill(tab == item.id ? item.soft : PanelTheme.surface)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 7)
                                .stroke(tab == item.id ? item.color : PanelTheme.lineStrong,
                                        lineWidth: 1)
                        )
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
            .animation(.easeInOut(duration: 0.18), value: openTab)

            if openTab == tab, let help = TabHelp.all[tab] {
                VStack(alignment: .leading, spacing: 7) {
                    Text(help.intro)
                        .font(.system(size: 11))
                        .lineSpacing(2)
                        .foregroundStyle(PanelTheme.text)
                    ForEach(help.items, id: \.name) { item in
                        HStack(alignment: .firstTextBaseline, spacing: 0) {
                            Text(item.name)
                                .font(.system(size: 11))
                                .foregroundStyle(PanelTheme.text)
                            Text(" — ")
                                .font(.system(size: 11))
                                .foregroundStyle(PanelTheme.textFaint)
                            Text(item.text)
                                .font(.system(size: 11))
                                .foregroundStyle(PanelTheme.textMuted)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

// MARK: - Generator gallery

private struct ModelFamily: Identifiable {
    let base: String
    var variants: [String: HFModel]
    var id: String { base }

    var preferred: HFModel? {
        variants["small"] ?? variants["medium"] ?? variants["big"]
            ?? variants.values.sorted { $0.name < $1.name }.first
    }
}

private struct ModelGallerySection: View {
    @EnvironmentObject var state: AppState
    @Binding var expandedFamily: String?

    private static let sizeOrder = ["small", "medium", "big"]
    private static let sizeLabels = ["small": "Small", "medium": "Medium", "big": "Big"]

    private var families: [ModelFamily] {
        var grouped: [String: ModelFamily] = [:]
        for model in state.modelManager.models {
            var base = model.name
            var size = "small"
            for suffix in Self.sizeOrder where model.name.lowercased().hasSuffix("_" + suffix) {
                base = String(model.name.dropLast(suffix.count + 1))
                size = suffix
                break
            }
            var family = grouped[base] ?? ModelFamily(base: base, variants: [:])
            family.variants[size] = model
            grouped[base] = family
        }
        return grouped.values.sorted { $0.base < $1.base }
    }

    private var selectedID: String? { state.modelManager.selected?.id }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("GENERATOR")
                .font(.system(size: 9.5))
                .tracking(1.3)
                .foregroundStyle(PanelTheme.textFaint)

            if state.modelManager.models.isEmpty {
                Text("Discovering models…")
                    .font(.system(size: 11))
                    .foregroundStyle(PanelTheme.textMuted)
            } else {
                VStack(spacing: 6) {
                    ForEach(families) { family in
                        familyItem(family)
                    }
                }
            }

            VStack(alignment: .leading, spacing: 2) {
                Text("Model distilled using this technique:")
                    .font(.system(size: 10))
                    .foregroundStyle(PanelTheme.textFaint)
                Button {
                    NSWorkspace.shared.open(URL(string: "https://arxiv.org/abs/2104.04767")!)
                } label: {
                    Text("https://arxiv.org/abs/2104.04767")
                        .font(.system(size: 9.5, design: .monospaced))
                        .foregroundStyle(PanelTheme.textMuted)
                        .underline(true, color: PanelTheme.lineStrong)
                }
                .buttonStyle(.plain)
            }
        }
        .onAppear { syncExpanded() }
        .onChange(of: selectedID) { _, _ in syncExpanded() }
    }

    private func syncExpanded() {
        guard let selectedID else { return }
        for family in families where family.variants.values.contains(where: { $0.id == selectedID }) {
            expandedFamily = family.base
            return
        }
    }

    @ViewBuilder
    private func familyItem(_ family: ModelFamily) -> some View {
        let selected = family.variants.values.contains { $0.id == selectedID }
        let open = expandedFamily == family.base

        VStack(spacing: 0) {
            Button {
                if selected {
                    expandedFamily = open ? nil : family.base
                } else {
                    expandedFamily = family.base
                    if let model = family.preferred {
                        Task { await state.load(model) }
                    }
                }
            } label: {
                HStack(spacing: 10) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 6)
                            .fill(PanelTheme.surface3)
                            .overlay(
                                RoundedRectangle(cornerRadius: 6)
                                    .stroke(PanelTheme.line, lineWidth: 1)
                            )
                        Image(systemName: "photo")
                            .font(.system(size: 18))
                            .foregroundStyle(PanelTheme.textFaint)
                    }
                    .frame(width: 48, height: 48)

                    Text(displayName(family.base))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(selected ? PanelTheme.text : PanelTheme.textMuted)
                        .lineLimit(1)
                        .truncationMode(.tail)

                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 9)
                .padding(.vertical, 7)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(state.modelManager.busy)

            if open {
                HStack(spacing: 6) {
                    ForEach(Self.sizeOrder, id: \.self) { size in
                        sizeOption(family.variants[size], size: size)
                    }
                }
                .padding(.horizontal, 9)
                .padding(.bottom, 9)
                .transition(.opacity.combined(with: .move(edge: .top)))

                if let credit = state.modelManager.credits[family.base],
                   credit.author != nil || !credit.links.isEmpty {
                    VStack(alignment: .leading, spacing: 1) {
                        if let author = credit.author {
                            Text("Author: \(author)")
                                .font(.system(size: 9.5))
                                .foregroundStyle(PanelTheme.textMuted)
                        }
                        ForEach(credit.links, id: \.self) { link in
                            if let url = URL(string: link) {
                                Button {
                                    NSWorkspace.shared.open(url)
                                } label: {
                                    Text(link)
                                        .font(.system(size: 9, design: .monospaced))
                                        .foregroundStyle(PanelTheme.textFaint)
                                        .underline(true, color: PanelTheme.lineStrong)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 9)
                    .padding(.top, 7)
                    .padding(.bottom, 9)
                    .overlay(alignment: .top) {
                        Rectangle()
                            .fill(PanelTheme.line)
                            .frame(height: 1)
                            .padding(.horizontal, 9)
                    }
                }
            }
        }
        .background(PanelTheme.surface)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(selected ? PanelTheme.text : PanelTheme.lineStrong, lineWidth: 1)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 7)
                .stroke(selected ? PanelTheme.text : .clear, lineWidth: 1)
                .padding(1)
        )
    }

    @ViewBuilder
    private func sizeOption(_ model: HFModel?, size: String) -> some View {
        let active = model != nil && model?.id == selectedID

        Button {
            guard let model else { return }
            expandedFamily = familyBase(for: model.name)
            Task { await state.load(model) }
        } label: {
            VStack(spacing: 1) {
                Text(Self.sizeLabels[size] ?? size)
                    .font(.system(size: 10.5))
                Text(model.map { $0.size > 0 ? "\(Int(Double($0.size) / 1_048_576)) MB" : "—" } ?? "—")
                    .font(.system(size: 8.5, design: .monospaced))
                    .foregroundStyle(active ? PanelTheme.textMuted : PanelTheme.textFaint)
            }
            .foregroundStyle(active ? PanelTheme.text : PanelTheme.textMuted)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(active ? Color.white.opacity(0.07) : PanelTheme.surface)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(active ? PanelTheme.text : PanelTheme.lineStrong, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(model == nil || state.modelManager.busy)
        .opacity(model == nil ? 0.3 : 1)
    }

    private func displayName(_ base: String) -> String {
        base.replacingOccurrences(of: "_", with: " ")
    }

    private func familyBase(for name: String) -> String {
        for suffix in Self.sizeOrder where name.lowercased().hasSuffix("_" + suffix) {
            return String(name.dropLast(suffix.count + 1))
        }
        return name
    }
}
