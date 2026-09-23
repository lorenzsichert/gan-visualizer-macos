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
                Button(state.audioOn ? "Disable Microphone" : "Enable Microphone") {
                    state.toggleMic()
                }
                .keyboardShortcut("u", modifiers: [.command, .shift])
                Button("Toggle Demo Signal") { state.demo.toggle() }
                    .keyboardShortcut("d", modifiers: [.command, .shift])
            }
        }
    }
}
