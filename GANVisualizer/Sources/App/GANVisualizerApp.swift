import SwiftUI

@main
struct GANVisualizerApp: App {
    @StateObject private var state = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(state)
                .frame(minWidth: 720, minHeight: 480)
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Visualizer") {
                Button("Toggle Fullscreen") { state.toggleFullscreen() }
                    .keyboardShortcut("f", modifiers: [.command, .control])
                Button("Cycle Mirror Mode") { state.cycleMirror() }
                    .keyboardShortcut("m", modifiers: [.command, .shift])
                Button(state.paused ? "Resume Visualizer" : "Pause Visualizer") {
                    state.togglePause()
                }
                .keyboardShortcut("p", modifiers: [.command, .shift])
                Button(state.input == .microphone ? "Disable Microphone" : "Enable Microphone") {
                    state.setInput(state.input == .microphone ? .off : .microphone)
                }
                .keyboardShortcut("u", modifiers: [.command, .shift])
                Button(state.input == .system ? "Disable System Audio" : "Enable System Audio") {
                    state.setInput(state.input == .system ? .off : .system)
                }
                .keyboardShortcut("s", modifiers: [.command, .shift])
                Button("Toggle Demo Signal") {
                    state.setInput(state.input == .demo ? .off : .demo)
                }
                .keyboardShortcut("d", modifiers: [.command, .shift])
            }
        }
    }
}
