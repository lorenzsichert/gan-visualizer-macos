import SwiftUI

struct ContentView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        ZStack(alignment: .topTrailing) {
            HStack(spacing: 0) {
                ZStack {
                    Color(red: 10 / 255, green: 11 / 255, blue: 13 / 255)
                    if let renderer = state.renderer {
                        VisualizerView(renderer: renderer)
                    } else {
                        Text("Metal is unavailable on this system.")
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                if !state.panelCollapsed {
                    ControlPanel()
                        .frame(width: 320)
                        .transition(.move(edge: .trailing))
                }
            }

            if state.panelCollapsed {
                Button {
                    withAnimation(.easeInOut(duration: 0.22)) { state.panelCollapsed = false }
                } label: {
                    Image(systemName: "chevron.left.2")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(PanelTheme.textMuted)
                        .frame(width: 34, height: 34)
                        .background(
                            RoundedRectangle(cornerRadius: 7)
                                .fill(PanelTheme.surface)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 7)
                                .stroke(PanelTheme.lineStrong, lineWidth: 1)
                        )
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .padding(12)
                .help("Show panel")
            }
        }
        .ignoresSafeArea()
    }
}
